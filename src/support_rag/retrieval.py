"""FAISS in-memory retrieval, a fixed bad retriever, and a pgvector store."""

from collections.abc import Sequence
from typing import Protocol

import numpy as np

from support_rag.embeddings import Embedder
from support_rag.models import Chunk, Retrieved

DEFAULT_K = 4
BAD_DOC_IDS: tuple[str, ...] = (
    "mobile-notifications",
    "attachments",
    "outages",
    "workspace-deletion",
)

UPSERT_SQL = (
    "INSERT INTO chunks (chunk_id, doc_id, title, content, embedding) "
    "VALUES (%s, %s, %s, %s, %s::vector) "
    "ON CONFLICT (chunk_id) DO UPDATE SET "
    "doc_id = EXCLUDED.doc_id, title = EXCLUDED.title, "
    "content = EXCLUDED.content, embedding = EXCLUDED.embedding"
)
SEARCH_SQL = (
    "SELECT chunk_id, doc_id, title, content, "
    "(embedding <=> %s::vector) AS distance "
    "FROM chunks ORDER BY embedding <=> %s::vector LIMIT %s"
)


class Retriever(Protocol):
    def retrieve(self, question: str, k: int = DEFAULT_K) -> list[Retrieved]: ...


def vector_literal(values: Sequence[float]) -> str:
    """Format a vector for a parameterized pgvector cast."""
    return "[" + ",".join(f"{value:.8f}" for value in values) + "]"


class FaissRetriever:
    """Cosine search over L2-normalized vectors using FAISS inner product."""

    def __init__(self, chunks: Sequence[Chunk], embedder: Embedder) -> None:
        if not chunks:
            raise ValueError("FAISS retriever needs at least one chunk")
        import faiss

        self.chunks = list(chunks)
        self.embedder = embedder
        texts = [chunk.embedded_text for chunk in self.chunks]
        embedder.fit(texts)
        matrix = np.asarray(embedder.embed_documents(texts), dtype=np.float32)
        index = faiss.IndexFlatIP(matrix.shape[1])
        index.add(matrix)
        self._index = index

    def retrieve(self, question: str, k: int = DEFAULT_K) -> list[Retrieved]:
        query = np.asarray([self.embedder.embed_query(question)], dtype=np.float32)
        scores, ids = self._index.search(query, min(k, len(self.chunks)))
        found: list[Retrieved] = []
        for score, index in zip(scores[0], ids[0], strict=True):
            position = int(index)
            if position < 0:
                continue
            found.append(Retrieved(self.chunks[position], float(score)))
        found.sort(key=lambda item: (-item.score, item.chunk.chunk_id))
        return found


class FixedRetriever:
    """Always returns the same unrelated chunks. Used to prove the eval gate fails."""

    def __init__(self, chunks: Sequence[Chunk], doc_ids: Sequence[str] = BAD_DOC_IDS) -> None:
        grouped: dict[str, list[Chunk]] = {}
        for chunk in chunks:
            grouped.setdefault(chunk.doc_id, []).append(chunk)
        missing = [doc_id for doc_id in doc_ids if doc_id not in grouped]
        if missing:
            raise ValueError(f"bad retriever is missing documents: {', '.join(missing)}")
        self._chunks = [chunk for doc_id in doc_ids for chunk in grouped[doc_id]]

    def retrieve(self, question: str, k: int = DEFAULT_K) -> list[Retrieved]:
        del question
        return [Retrieved(chunk, 0.0) for chunk in self._chunks[:k]]


class PgVectorStore:
    """Postgres + pgvector retriever. The connection is opened by the caller."""

    def __init__(self, connection: object, embedder: Embedder) -> None:
        self._connection = connection
        self.embedder = embedder

    def create_schema(self, dim: int) -> None:
        if dim < 1:
            raise ValueError("dim must be positive")
        with self._connection.cursor() as cursor:  # type: ignore[attr-defined]
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cursor.execute("DROP TABLE IF EXISTS chunks")
            cursor.execute(
                f"CREATE TABLE chunks ("
                f"chunk_id TEXT PRIMARY KEY, "
                f"doc_id TEXT NOT NULL, "
                f"title TEXT NOT NULL, "
                f"content TEXT NOT NULL, "
                f"embedding vector({int(dim)})"
                f")"
            )
        self._connection.commit()  # type: ignore[attr-defined]

    def upsert(self, chunks: Sequence[Chunk]) -> None:
        texts = [chunk.embedded_text for chunk in chunks]
        self.embedder.fit(texts)
        vectors = self.embedder.embed_documents(texts)
        if chunks and not self._table_ready():
            self.create_schema(len(vectors[0]))
        with self._connection.cursor() as cursor:  # type: ignore[attr-defined]
            for chunk, vector in zip(chunks, vectors, strict=True):
                cursor.execute(
                    UPSERT_SQL,
                    (
                        chunk.chunk_id,
                        chunk.doc_id,
                        chunk.title,
                        chunk.text,
                        vector_literal(vector),
                    ),
                )
        self._connection.commit()  # type: ignore[attr-defined]

    def retrieve(self, question: str, k: int = DEFAULT_K) -> list[Retrieved]:
        query = vector_literal(self.embedder.embed_query(question))
        with self._connection.cursor() as cursor:  # type: ignore[attr-defined]
            cursor.execute(SEARCH_SQL, (query, query, k))
            rows = cursor.fetchall()
        found: list[Retrieved] = []
        for row in rows:
            chunk = Chunk(
                chunk_id=str(row[0]),
                doc_id=str(row[1]),
                title=str(row[2]),
                text=str(row[3]),
            )
            distance = float(row[4])
            found.append(Retrieved(chunk, 1.0 - distance))
        return found

    def _table_ready(self) -> bool:
        with self._connection.cursor() as cursor:  # type: ignore[attr-defined]
            cursor.execute("SELECT to_regclass('public.chunks')")
            row = cursor.fetchone()
        return bool(row and row[0])


def connect_pg(dsn: str) -> object:
    import psycopg
    from pgvector.psycopg import register_vector

    try:
        connection = psycopg.connect(dsn, connect_timeout=5)
    except psycopg.OperationalError as exc:
        raise ConnectionError(
            "Could not connect to Postgres. Start it with `docker compose up -d` "
            "or use the default FAISS store."
        ) from exc
    register_vector(connection)
    return connection

import math

from support_rag.chunking import chunk_documents
from support_rag.embeddings import HashingTfidfEmbedder
from support_rag.models import Document
from support_rag.retrieval import (
    SEARCH_SQL,
    UPSERT_SQL,
    FaissRetriever,
    FixedRetriever,
    vector_literal,
)


def _chunks() -> list:
    documents = [
        Document("refunds", "Refunds", "Annual plans can be refunded within 14 days."),
        Document("sync", "Sync", "Install desktop version 4.2 when sync is stuck."),
        Document("notes", "Notes", "Guests can see only the projects shared with them."),
    ]
    return chunk_documents(documents)


def test_hashing_embedder_is_deterministic() -> None:
    embedder = HashingTfidfEmbedder(dim=64)
    embedder.fit(["alpha beta", "gamma delta"])
    assert embedder.embed_query("alpha beta") == embedder.embed_query("alpha beta")
    assert len(embedder.embed_query("alpha")) == 64


def test_faiss_retrieves_the_matching_document() -> None:
    chunks = _chunks()
    retriever = FaissRetriever(chunks, HashingTfidfEmbedder(dim=128))
    found = retriever.retrieve("When can an annual plan be refunded?", k=1)
    assert found[0].chunk.doc_id == "refunds"


def test_fixed_retriever_ignores_the_question() -> None:
    chunks = _chunks()
    retriever = FixedRetriever(chunks, ["sync", "notes"])
    found = retriever.retrieve("When can an annual plan be refunded?", k=2)
    assert [item.chunk.doc_id for item in found] == ["sync", "notes"]


def test_vector_literal_and_sql_use_parameters() -> None:
    literal = vector_literal([0.1, -0.25])
    assert literal == "[0.10000000,-0.25000000]"
    assert "%s" in UPSERT_SQL
    assert "%s::vector" in SEARCH_SQL
    assert math.isfinite(float(literal.strip("[]").split(",")[0]))

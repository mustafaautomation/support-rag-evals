"""Load the help center into the docker-compose pgvector database."""

import argparse
import os
import sys

from support_rag.embeddings import HashingTfidfEmbedder
from support_rag.pipeline import load_chunks
from support_rag.retrieval import PgVectorStore, connect_pg

DEFAULT_DSN = "postgresql://sable:sable@localhost:5432/sableboard"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Index the help center in pgvector.")
    parser.add_argument("--dsn", default=os.environ.get("DATABASE_URL", DEFAULT_DSN))
    args = parser.parse_args(argv)
    chunks = load_chunks()
    embedder = HashingTfidfEmbedder()
    try:
        connection = connect_pg(args.dsn)
    except ConnectionError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    store = PgVectorStore(connection, embedder)
    store.upsert(chunks)
    print(f"Indexed {len(chunks)} chunks into pgvector.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

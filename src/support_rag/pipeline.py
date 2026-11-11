"""Assemble a retriever and a generator into one question-answering call."""

from collections.abc import Sequence
from pathlib import Path

from support_rag.chunking import chunk_documents
from support_rag.corpus import load_corpus
from support_rag.embeddings import Embedder, HashingTfidfEmbedder
from support_rag.guardrails import decide
from support_rag.models import Answer, Chunk
from support_rag.paths import repo_root
from support_rag.providers import Generator, OfflineGenerator
from support_rag.retrieval import DEFAULT_K, FaissRetriever, FixedRetriever, Retriever

VARIANTS: tuple[str, ...] = ("improved", "bad-retriever", "no-guardrail")


def load_chunks(root: Path | None = None) -> list[Chunk]:
    corpus_dir = (root or repo_root()) / "data" / "corpus"
    return chunk_documents(load_corpus(corpus_dir))


def build_pipeline(
    variant: str = "improved",
    *,
    generator: Generator | None = None,
    embedder: Embedder | None = None,
    chunks: Sequence[Chunk] | None = None,
    k: int = DEFAULT_K,
) -> "Pipeline":
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}. Expected one of {', '.join(VARIANTS)}.")
    prepared = list(chunks) if chunks is not None else load_chunks()
    if variant == "bad-retriever":
        retriever: Retriever = FixedRetriever(prepared)
    else:
        retriever = FaissRetriever(prepared, embedder or HashingTfidfEmbedder())
    return Pipeline(
        retriever=retriever,
        generator=generator or OfflineGenerator(),
        guardrails_enabled=variant != "no-guardrail",
        variant=variant,
        k=k,
    )


class Pipeline:
    def __init__(
        self,
        retriever: Retriever,
        generator: Generator,
        *,
        guardrails_enabled: bool,
        variant: str,
        k: int = DEFAULT_K,
    ) -> None:
        self.retriever = retriever
        self.generator = generator
        self.guardrails_enabled = guardrails_enabled
        self.variant = variant
        self.k = k

    def answer(self, question: str) -> Answer:
        retrieved = tuple(self.retriever.retrieve(question, k=self.k))
        decision = decide(question) if self.guardrails_enabled else None
        if decision is not None and decision.response is not None:
            response = decision.response
            category: str | None = decision.category
        else:
            response = self.generator.generate(question, retrieved, guarded=self.guardrails_enabled)
            category = None
        return Answer(
            question=question,
            response=response,
            retrieved=retrieved,
            guardrail=category,
            variant=self.variant,
        )

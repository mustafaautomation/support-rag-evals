"""Shared records for documents, retrieval, and answers."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    body: str


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    text: str

    @property
    def embedded_text(self) -> str:
        return f"{self.title}\n\n{self.text}"


@dataclass(frozen=True)
class Retrieved:
    chunk: Chunk
    score: float


@dataclass(frozen=True)
class Answer:
    question: str
    response: str
    retrieved: tuple[Retrieved, ...]
    guardrail: str | None
    variant: str

"""Labelled questions for the support-bot evaluation."""

import json
from dataclasses import dataclass
from pathlib import Path

from support_rag.models import Chunk
from support_rag.paths import repo_root

BEHAVIORS = frozenset({"answer", "refuse", "escalate", "unknown"})


@dataclass(frozen=True)
class EvalQuestion:
    id: str
    topic: str
    question: str
    reference: str
    reference_doc_ids: tuple[str, ...]
    reference_contexts: tuple[str, ...]
    expected_behavior: str
    presence_phrase: str


def load_questions(path: Path | None = None) -> list[EvalQuestion]:
    question_path = path or (repo_root() / "data" / "eval" / "questions.jsonl")
    questions: list[EvalQuestion] = []
    seen: set[str] = set()
    lines = question_path.read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"Line {line_number} is not an object")
        question = _parse(payload, line_number)
        if question.id in seen:
            raise ValueError(f"Duplicate question id {question.id}")
        seen.add(question.id)
        questions.append(question)
    if not questions:
        raise ValueError(f"No questions in {question_path}")
    return questions


def attach_contexts(questions: list[EvalQuestion], chunks: list[Chunk]) -> list[EvalQuestion]:
    """Fill reference contexts from the chunk that contains the labelled phrase."""
    attached: list[EvalQuestion] = []
    for question in questions:
        contexts = _contexts_for(question, chunks)
        attached.append(
            EvalQuestion(
                id=question.id,
                topic=question.topic,
                question=question.question,
                reference=question.reference,
                reference_doc_ids=question.reference_doc_ids,
                reference_contexts=tuple(contexts),
                expected_behavior=question.expected_behavior,
                presence_phrase=question.presence_phrase,
            )
        )
    return attached


def _contexts_for(question: EvalQuestion, chunks: list[Chunk]) -> list[str]:
    if not question.reference_doc_ids:
        return []
    matching = [
        chunk.embedded_text
        for chunk in chunks
        if chunk.doc_id in question.reference_doc_ids and question.presence_phrase in chunk.text
    ]
    if matching:
        return matching
    fallback = [
        chunk.embedded_text for chunk in chunks if chunk.doc_id in question.reference_doc_ids
    ]
    if not fallback:
        raise ValueError(f"{question.id} names unknown documents: {question.reference_doc_ids}")
    return fallback


def _string_list(value: object, field: str, line_number: int) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"Line {line_number} {field} must be a list of strings")
    items: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError(f"Line {line_number} {field} must be a list of strings")
        items.append(item)
    return items


def _parse(payload: dict[str, object], line_number: int) -> EvalQuestion:
    def text(key: str) -> str:
        value = payload.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Line {line_number} field {key} must be a non-empty string")
        return value.strip()

    behavior = text("expected_behavior")
    if behavior not in BEHAVIORS:
        raise ValueError(f"Line {line_number} has unknown expected_behavior {behavior!r}")
    doc_ids = _string_list(payload.get("reference_doc_ids", []), "reference_doc_ids", line_number)
    contexts = _string_list(
        payload.get("reference_contexts", []),
        "reference_contexts",
        line_number,
    )
    return EvalQuestion(
        id=text("id"),
        topic=text("topic"),
        question=text("question"),
        reference=text("reference"),
        reference_doc_ids=tuple(doc_ids),
        reference_contexts=tuple(contexts),
        expected_behavior=behavior,
        presence_phrase=text("presence_phrase"),
    )

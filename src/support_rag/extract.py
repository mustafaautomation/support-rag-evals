"""Extractive offline answers. Sentences are quoted from retrieved chunks."""

from collections.abc import Sequence

from support_rag.guardrails import UNKNOWN_RESPONSE, template_sentences
from support_rag.models import Retrieved
from support_rag.textutil import content_tokens, split_sentences

MIN_OVERLAP = 0.4

_TEMPLATES = template_sentences()


def overlap_score(question: str, sentence: str) -> float:
    question_tokens = set(content_tokens(question))
    sentence_tokens = set(content_tokens(sentence))
    if not question_tokens or not sentence_tokens:
        return 0.0
    return len(question_tokens & sentence_tokens) / len(question_tokens)


def select_sentences(
    question: str,
    retrieved: Sequence[Retrieved],
    limit: int = 2,
) -> list[str]:
    scored: list[tuple[float, str]] = []
    for item in retrieved:
        for sentence in split_sentences(item.chunk.text):
            if sentence in _TEMPLATES:
                continue
            score = overlap_score(question, sentence)
            if score > 0.0:
                scored.append((score, sentence))
    scored.sort(key=lambda pair: (-pair[0], pair[1]))
    if not scored or scored[0][0] < MIN_OVERLAP:
        return []
    best = scored[0][0]
    picked: list[str] = []
    seen: set[str] = set()
    for score, sentence in scored:
        if sentence in seen:
            continue
        if picked and score < best - 0.15:
            break
        if score < MIN_OVERLAP:
            break
        picked.append(sentence)
        seen.add(sentence)
        if len(picked) >= limit:
            break
    return picked


def extractive_answer(question: str, retrieved: Sequence[Retrieved]) -> str:
    sentences = select_sentences(question, retrieved)
    if not sentences:
        return UNKNOWN_RESPONSE
    return " ".join(sentences)

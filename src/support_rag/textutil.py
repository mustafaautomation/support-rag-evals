"""Token helpers shared by the offline embedder and the extractive generator."""

import re

_TOKEN = re.compile(r"[a-z0-9]+(?:[.][a-z0-9]+)*")
_STOP = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "of",
        "to",
        "for",
        "in",
        "on",
        "my",
        "your",
        "is",
        "are",
        "do",
        "does",
        "how",
        "what",
        "when",
        "where",
        "which",
        "who",
        "can",
        "i",
        "we",
        "it",
        "if",
        "with",
        "that",
        "this",
        "be",
        "you",
        "our",
        "at",
        "from",
        "by",
        "as",
        "me",
        "was",
        "were",
        "been",
        "have",
        "has",
        "had",
        "please",
        "so",
        "am",
        "into",
        "about",
        "its",
        "their",
        "them",
        "than",
        "then",
        "also",
        "only",
        "will",
        "just",
        "any",
        "not",
    }
)


def stem(token: str) -> str:
    """Strip a trailing plural s, leaving words that merely end in s alone."""
    if len(token) > 4 and token.endswith("s") and not token.endswith(("ss", "us", "is", "os")):
        return token[:-1]
    return token


def content_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    for raw in _TOKEN.findall(text.lower()):
        if raw in _STOP:
            continue
        if len(raw) < 2 and not raw.isdigit():
            continue
        tokens.append(stem(raw))
    return tokens


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text.strip())
    return [part.strip() for part in parts if part.strip()]

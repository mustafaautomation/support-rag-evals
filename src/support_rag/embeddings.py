"""Embedders. The offline embedder is a hashed TF-IDF vector, not a neural model."""

import hashlib
import math
from collections import Counter
from collections.abc import Sequence
from typing import Protocol

from support_rag.textutil import content_tokens


class Embedder(Protocol):
    def fit(self, texts: Sequence[str]) -> None: ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


def _features(text: str) -> list[str]:
    tokens = content_tokens(text)
    features = list(tokens)
    features.extend(f"{left}_{right}" for left, right in zip(tokens, tokens[1:], strict=False))
    return features


def _bucket(feature: str, dim: int) -> tuple[int, float]:
    digest = hashlib.sha256(feature.encode("utf-8")).digest()
    index = int.from_bytes(digest[:4], "little") % dim
    sign = 1.0 if digest[4] % 2 == 0 else -1.0
    return index, sign


def _l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return vector
    return [value / norm for value in vector]


class HashingTfidfEmbedder:
    """Deterministic hashing embedder fit on the corpus. No network and no API key."""

    def __init__(self, dim: int = 384) -> None:
        if dim < 8:
            raise ValueError("dim must be at least 8")
        self.dim = dim
        self._idf: dict[str, float] = {}

    def fit(self, texts: Sequence[str]) -> None:
        documents = [_features(text) for text in texts]
        count = max(len(documents), 1)
        doc_freq: Counter[str] = Counter()
        for features in documents:
            doc_freq.update(set(features))
        self._idf = {
            feature: math.log((1.0 + count) / (1.0 + seen)) + 1.0
            for feature, seen in doc_freq.items()
        }

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        counts = Counter(_features(text))
        vector = [0.0] * self.dim
        for feature, term_freq in counts.items():
            index, sign = _bucket(feature, self.dim)
            weight = (1.0 + math.log(term_freq)) * self._idf.get(feature, 1.0)
            vector[index] += sign * weight
        return _l2_normalize(vector)


class OpenAIEmbedder:
    """OpenAI embeddings. `fit` is a no-op; the model is already trained."""

    def __init__(self, client: object, model: str = "text-embedding-3-small") -> None:
        self._client = client
        self.model = model

    def fit(self, texts: Sequence[str]) -> None:
        del texts

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self._client.embeddings.create(model=self.model, input=list(texts))  # type: ignore[attr-defined]
        ordered = sorted(response.data, key=lambda item: item.index)
        return [list(item.embedding) for item in ordered]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

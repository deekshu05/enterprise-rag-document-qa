"""Embedding generation and a FAISS-backed vector index for document retrieval."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

import faiss
import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class Embedder:
    """Minimal interface any embedding provider must implement."""

    dimension: int

    def embed(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError


class HashingEmbedder(Embedder):
    """Deterministic, dependency-free embedder used for local dev and tests.

    Hashes tokens into a fixed-size bag-of-words vector. Not semantically
    rich, but stable and fast -- useful for exercising the retrieval
    pipeline without calling an external embeddings API.
    """

    def __init__(self, dimension: int = 256):
        self.dimension = dimension

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dimension), dtype="float32")
        for i, text in enumerate(texts):
            for token in _TOKEN_RE.findall(text.lower()):
                digest = int(hashlib.sha256(token.encode("utf-8")).hexdigest(), 16)
                vectors[i, digest % self.dimension] += 1.0
            norm = np.linalg.norm(vectors[i])
            if norm > 0:
                vectors[i] /= norm
        return vectors


class OpenAIEmbedder(Embedder):
    """Wraps OpenAI's embeddings API (text-embedding-3-small by default)."""

    def __init__(self, model: str = "text-embedding-3-small", dimension: int = 1536):
        import openai  # imported lazily so the package is optional at import time

        self._client = openai.OpenAI()
        self.model = model
        self.dimension = dimension

    def embed(self, texts: list[str]) -> np.ndarray:
        response = self._client.embeddings.create(model=self.model, input=texts)
        return np.array([item.embedding for item in response.data], dtype="float32")


@dataclass
class VectorIndex:
    """FAISS-backed similarity index over embedded document chunks."""

    dimension: int
    _index: faiss.IndexFlatIP = field(init=False, repr=False)
    _chunks: list[str] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        self._index = faiss.IndexFlatIP(self.dimension)

    def add(self, chunks: list[str], vectors: np.ndarray) -> None:
        self._index.add(vectors)
        self._chunks.extend(chunks)

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[tuple[str, float]]:
        scores, indices = self._index.search(query_vector.reshape(1, -1), top_k)
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx == -1:
                continue
            results.append((self._chunks[idx], float(score)))
        return results

    def __len__(self) -> int:
        return len(self._chunks)

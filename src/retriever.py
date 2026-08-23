"""Semantic retrieval over an embedded document index."""

from __future__ import annotations

from dataclasses import dataclass

from src.embeddings import Embedder, VectorIndex


@dataclass
class RetrievedChunk:
    text: str
    score: float


class Retriever:
    """Retrieves the most relevant chunks for a query using semantic search."""

    def __init__(self, embedder: Embedder, index: VectorIndex):
        self.embedder = embedder
        self.index = index

    def index_documents(self, chunks: list[str]) -> None:
        if not chunks:
            return
        vectors = self.embedder.embed(chunks)
        self.index.add(chunks, vectors)

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        query_vector = self.embedder.embed([query])[0]
        results = self.index.search(query_vector, top_k=top_k)
        return [RetrievedChunk(text=text, score=score) for text, score in results]

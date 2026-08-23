"""End-to-end RAG pipeline: ingest -> embed -> retrieve -> generate."""

from __future__ import annotations

from dataclasses import dataclass

from src.config import RAGConfig
from src.embeddings import Embedder, HashingEmbedder, VectorIndex
from src.ingestion import chunk_documents, load_documents
from src.retriever import Retriever

PROMPT_TEMPLATE = """Answer the question using only the context below. If the
context does not contain the answer, say so.

Context:
{context}

Question: {query}

Answer:
"""


@dataclass
class RAGAnswer:
    query: str
    answer: str
    sources: list[str]


class LLMClient:
    """Minimal interface any LLM provider client must implement."""

    def complete(self, prompt: str) -> str:
        raise NotImplementedError


class MockLLMClient(LLMClient):
    """Deterministic client for local testing without an API key."""

    def complete(self, prompt: str) -> str:
        return "Mock answer grounded in the retrieved context."


class AnthropicLLMClient(LLMClient):
    """Thin wrapper around the Anthropic Claude API."""

    def __init__(self, model: str = "claude-sonnet-4-5", max_tokens: int = 1024):
        import anthropic  # imported lazily so the package is optional at import time

        self._client = anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens

    def complete(self, prompt: str) -> str:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in response.content if hasattr(block, "text"))


class RAGPipeline:
    """Orchestrates ingestion, retrieval, and grounded answer generation."""

    def __init__(self, config: RAGConfig, embedder: Embedder | None = None, llm: LLMClient | None = None):
        self.config = config
        self.embedder = embedder or HashingEmbedder(dimension=config.embedding_dimension)
        self.index = VectorIndex(dimension=self.embedder.dimension)
        self.retriever = Retriever(self.embedder, self.index)
        self.llm = llm or MockLLMClient()

    def ingest(self, directory: str) -> int:
        documents = load_documents(directory)
        chunks = chunk_documents(documents, self.config.chunk_size, self.config.chunk_overlap)
        self.retriever.index_documents(chunks)
        return len(chunks)

    def query(self, question: str) -> RAGAnswer:
        retrieved = self.retriever.retrieve(question, top_k=self.config.top_k)
        context = "\n\n".join(chunk.text for chunk in retrieved)
        prompt = PROMPT_TEMPLATE.format(context=context, query=question)
        answer = self.llm.complete(prompt)
        return RAGAnswer(query=question, answer=answer, sources=[c.text for c in retrieved])

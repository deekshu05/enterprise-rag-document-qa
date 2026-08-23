"""Configuration for the RAG document Q&A pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class RAGConfig:
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "500"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "50"))
    top_k: int = int(os.getenv("TOP_K", "5"))
    embedding_dimension: int = int(os.getenv("EMBEDDING_DIMENSION", "256"))
    embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "hashing")
    llm_provider: str = os.getenv("LLM_PROVIDER", "mock")

"""FastAPI microservice exposing the RAG pipeline over HTTP."""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from src.config import RAGConfig
from src.rag_pipeline import RAGPipeline

app = FastAPI(title="Enterprise RAG Document Q&A")
pipeline = RAGPipeline(config=RAGConfig())


class IngestRequest(BaseModel):
    directory: str


class QueryRequest(BaseModel):
    question: str


@app.post("/ingest")
def ingest(request: IngestRequest) -> dict:
    chunk_count = pipeline.ingest(request.directory)
    return {"chunks_indexed": chunk_count}


@app.post("/query")
def query(request: QueryRequest) -> dict:
    result = pipeline.query(request.question)
    return {"answer": result.answer, "sources": result.sources}


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}

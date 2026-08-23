"""Document loading and chunking for the RAG ingestion pipeline."""

from __future__ import annotations

from pathlib import Path


def load_documents(directory: str | Path, extensions: tuple[str, ...] = (".txt", ".md")) -> dict[str, str]:
    """Load all matching files under directory as {path: text}."""
    documents: dict[str, str] = {}
    for path in Path(directory).rglob("*"):
        if path.is_file() and path.suffix in extensions:
            documents[str(path)] = path.read_text(encoding="utf-8", errors="ignore")
    return documents


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split text into overlapping chunks of roughly chunk_size characters."""
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")

    chunks: list[str] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == length:
            break
        start = end - overlap
    return chunks


def chunk_documents(documents: dict[str, str], chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Chunk every document and return a flat list of chunks."""
    all_chunks: list[str] = []
    for text in documents.values():
        all_chunks.extend(chunk_text(text, chunk_size, overlap))
    return all_chunks

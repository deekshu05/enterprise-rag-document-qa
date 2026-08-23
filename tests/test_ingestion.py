from src.ingestion import chunk_text


def test_chunk_text_respects_size():
    text = "a" * 1200
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    assert all(len(c) <= 500 for c in chunks)
    assert len(chunks) >= 3


def test_chunk_text_overlaps():
    text = "0123456789" * 20
    chunks = chunk_text(text, chunk_size=50, overlap=10)
    assert chunks[0][-10:] == chunks[1][:10]


def test_chunk_text_short_text_single_chunk():
    text = "short text"
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    assert chunks == ["short text"]


def test_chunk_text_rejects_invalid_overlap():
    import pytest

    with pytest.raises(ValueError):
        chunk_text("some text", chunk_size=10, overlap=10)

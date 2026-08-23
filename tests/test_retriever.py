from src.embeddings import HashingEmbedder, VectorIndex
from src.retriever import Retriever


def test_retriever_finds_relevant_chunk():
    embedder = HashingEmbedder(dimension=64)
    index = VectorIndex(dimension=64)
    retriever = Retriever(embedder, index)

    chunks = [
        "The quarterly revenue grew by twelve percent year over year.",
        "The cafeteria menu changed to include more vegetarian options.",
        "Net income increased due to strong revenue growth this quarter.",
    ]
    retriever.index_documents(chunks)

    results = retriever.retrieve("What happened to revenue this quarter?", top_k=2)

    assert len(results) == 2
    assert any("revenue" in r.text for r in results)


def test_retriever_returns_empty_for_empty_index():
    embedder = HashingEmbedder(dimension=32)
    index = VectorIndex(dimension=32)
    retriever = Retriever(embedder, index)

    results = retriever.retrieve("anything", top_k=3)

    assert results == []

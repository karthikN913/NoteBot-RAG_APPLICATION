"""Pytest Configuration and Mock Fixtures for NoteBot Test Suite."""

import pytest
from typing import List
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


class MockEmbeddings(Embeddings):
    """Deterministic, zero-cost mock embeddings for fast offline testing."""

    def __init__(self, dimension: int = 16):
        self.dimension = dimension

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        results = []
        for text in texts:
            # Generate deterministic pseudo-vector based on text length and hash
            val = (len(text) % 10) / 10.0
            vec = [val] * self.dimension
            results.append(vec)
        return results

    def embed_query(self, text: str) -> List[float]:
        val = (len(text) % 10) / 10.0
        return [val] * self.dimension


@pytest.fixture
def mock_embeddings():
    return MockEmbeddings()


@pytest.fixture
def sample_documents() -> List[Document]:
    """Sample documents simulating multi-page, multi-document corpus."""
    return [
        Document(
            page_content="Machine learning is a subfield of artificial intelligence focused on algorithms that learn from data.",
            metadata={"source": "AI_Intro.pdf", "page": 1, "chunk_id": "AI_Intro_p1_c0"}
        ),
        Document(
            page_content="Supervised learning uses labeled training datasets to train models for classification and regression.",
            metadata={"source": "AI_Intro.pdf", "page": 2, "chunk_id": "AI_Intro_p2_c0"}
        ),
        Document(
            page_content="Reinforcement learning involves an agent learning via rewards and penalties in an interactive environment.",
            metadata={"source": "AI_Intro.pdf", "page": 3, "chunk_id": "AI_Intro_p3_c0"}
        ),
        Document(
            page_content="Neural networks are inspired by biological neurons and consist of input, hidden, and output layers.",
            metadata={"source": "Deep_Learning.pdf", "page": 1, "chunk_id": "Deep_Learning_p1_c0"}
        )
    ]

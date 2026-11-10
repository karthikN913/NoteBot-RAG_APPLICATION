"""Unit Tests for Metadata-Preserving Chunking and Deduplication."""

import pytest
from langchain_core.documents import Document
from notebot.chunking.chunker import chunk_documents


def test_chunk_documents_metadata_preservation():
    docs = [
        Document(
            page_content="Sentence one is about neural networks. Sentence two describes activation functions. Sentence three explains backpropagation.",
            metadata={"source": "DeepLearning.pdf", "page": 4}
        )
    ]

    chunks = chunk_documents(docs, chunk_size=70, chunk_overlap=15)
    assert len(chunks) >= 2

    for c in chunks:
        assert c.metadata["source"] == "DeepLearning.pdf"
        assert c.metadata["page"] == 4
        assert "chunk_id" in c.metadata
        assert c.metadata["chunk_id"].startswith("DeepLearning.pdf_p4_c")
        assert len(c.page_content) > 0


def test_chunk_deduplication():
    # Simulate recurring header/footer on different pages
    header_text = "CS101 Introduction to Computer Science — Fall Semester 2026"
    docs = [
        Document(
            page_content=f"{header_text}\n\nToday we will discuss sorting algorithms in depth.",
            metadata={"source": "Lecture.pdf", "page": 1}
        ),
        Document(
            page_content=f"{header_text}\n\nSorting algorithms include Quicksort and Mergesort.",
            metadata={"source": "Lecture.pdf", "page": 2}
        )
    ]

    chunks = chunk_documents(docs, chunk_size=100, chunk_overlap=0, deduplicate=True)
    header_occurrences = [c for c in chunks if header_text in c.page_content]
    # Header should only be retained once, preventing redundant clutter in the vector store
    assert len(header_occurrences) == 1


def test_chunk_empty_input():
    chunks = chunk_documents([], chunk_size=500, chunk_overlap=50)
    assert chunks == []


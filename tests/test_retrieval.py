"""Unit Tests for FAISS, MMR, BM25 Hybrid Search, and Reranking."""

import pytest
from notebot.vectorstore.faiss_manager import FAISSIndexManager
from notebot.retrieval.hybrid_search import BM25Index, reciprocal_rank_fusion
from notebot.retrieval.reranker import Reranker
from notebot.retrieval.retriever import NoteBotRetriever


def test_faiss_index_creation_and_similarity(sample_documents, mock_embeddings):
    manager = FAISSIndexManager.from_documents(sample_documents, mock_embeddings)
    assert manager.vector_store is not None
    assert len(manager.get_all_documents()) == 4

    # Similarity search
    results = manager.search_similarity("machine learning", k=2)
    assert len(results) == 2
    assert results[0][0].metadata["source"] in ["AI_Intro.pdf", "Deep_Learning.pdf"]


def test_faiss_mmr_retrieval(sample_documents, mock_embeddings):
    manager = FAISSIndexManager.from_documents(sample_documents, mock_embeddings)
    mmr_docs = manager.search_mmr("neural networks", k=2, lambda_mult=0.5)
    assert len(mmr_docs) == 2


def test_bm25_sparse_search(sample_documents):
    bm25 = BM25Index(sample_documents)
    # Search for exact keyword unique to one document
    results = bm25.search("reinforcement learning rewards", top_k=2)
    assert len(results) >= 1
    top_doc, score = results[0]
    assert "Reinforcement learning" in top_doc.page_content


def test_reciprocal_rank_fusion(sample_documents):
    dense_res = [sample_documents[0], sample_documents[1]]
    sparse_res = [sample_documents[2], sample_documents[0]]

    # Document 0 appears in both and should rank at the top
    fused = reciprocal_rank_fusion(dense_res, sparse_res, k=60, top_n=3)
    assert len(fused) == 3
    assert fused[0].metadata["chunk_id"] == "AI_Intro_p1_c0"


def test_notebot_unified_retriever(sample_documents, mock_embeddings):
    manager = FAISSIndexManager.from_documents(sample_documents, mock_embeddings)
    bm25 = BM25Index(sample_documents)
    retriever = NoteBotRetriever(index_manager=manager, bm25_index=bm25)

    # Test similarity mode
    sim_docs = retriever.retrieve("supervised learning", top_k=2, mode="similarity")
    assert len(sim_docs) == 2

    # Test mmr mode
    mmr_docs = retriever.retrieve("neural networks", top_k=2, mode="mmr")
    assert len(mmr_docs) == 2

    # Test hybrid mode
    hyb_docs = retriever.retrieve("reinforcement agent", top_k=2, mode="hybrid")
    assert len(hyb_docs) == 2

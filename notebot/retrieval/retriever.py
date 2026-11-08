"""Unified Document Retrieval Orchestrator for NoteBot.
Combines Similarity, MMR, BM25 Hybrid Search, and Cross-Encoder Reranking.
"""

from typing import List, Optional
from langchain_core.documents import Document
from notebot.vectorstore.faiss_manager import FAISSIndexManager
from notebot.retrieval.hybrid_search import BM25Index, reciprocal_rank_fusion
from notebot.retrieval.reranker import Reranker
from notebot.config.logging_config import setup_logger

logger = setup_logger("notebot.retriever")


class NoteBotRetriever:
    """Unified retrieval orchestrator supporting configurable strategies."""

    def __init__(
        self,
        index_manager: FAISSIndexManager,
        bm25_index: Optional[BM25Index] = None,
        reranker: Optional[Reranker] = None
    ):
        self.index_manager = index_manager
        self.bm25_index = bm25_index
        self.reranker = reranker or Reranker()

    def retrieve(
        self,
        query: str,
        top_k: int = 4,
        mode: str = "similarity",
        score_threshold: float = 0.0,
        enable_reranker: bool = False
    ) -> List[Document]:
        """
        Retrieves the most relevant documents according to the chosen strategy.
        
        Modes:
            - "similarity": Standard dense vector search with optional score threshold.
            - "mmr": Maximum Marginal Relevance for diverse, non-redundant passages.
            - "hybrid": BM25 sparse keyword search fused with FAISS dense vector search via RRF.
        """
        if not self.index_manager or not self.index_manager.vector_store:
            return []

        # If reranker is enabled, retrieve a larger initial candidate pool (e.g. 2x top_k)
        fetch_k = top_k * 2 if enable_reranker else top_k

        candidate_docs: List[Document] = []

        if mode == "mmr":
            candidate_docs = self.index_manager.search_mmr(
                query=query,
                k=fetch_k,
                fetch_k=max(20, fetch_k * 2)
            )

        elif mode == "hybrid" and self.bm25_index is not None:
            # 1. Dense search
            dense_results = [
                doc for doc, _ in self.index_manager.search_similarity(
                    query=query,
                    k=fetch_k,
                    score_threshold=score_threshold
                )
            ]
            # 2. Sparse BM25 search
            sparse_pairs = self.bm25_index.search(query=query, top_k=fetch_k)
            sparse_results = [doc for doc, _ in sparse_pairs]

            # 3. Reciprocal Rank Fusion
            candidate_docs = reciprocal_rank_fusion(
                dense_results=dense_results,
                sparse_results=sparse_results,
                top_n=fetch_k
            )

        else:
            # Default: standard similarity search
            results_with_scores = self.index_manager.search_similarity(
                query=query,
                k=fetch_k,
                score_threshold=score_threshold
            )
            candidate_docs = [doc for doc, _ in results_with_scores]

        # Apply optional cross-encoder reranking
        if enable_reranker and candidate_docs:
            return self.reranker.rerank(query=query, documents=candidate_docs, top_n=top_k)

        return candidate_docs[:top_k]

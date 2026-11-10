"""FAISS Vector Store Manager for NoteBot.
Handles building, document addition, similarity search with scores, and MMR retrieval.
"""

from typing import List, Tuple, Optional, Any
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from notebot.config.logging_config import setup_logger

logger = setup_logger("notebot.faiss_manager")


class FAISSIndexManager:
    """Manages creation, document updates, and vector operations on a FAISS vector store."""

    def __init__(self, vector_store: Optional[FAISS] = None, embeddings: Optional[Any] = None):
        self.vector_store = vector_store
        self.embeddings = embeddings

    @classmethod
    def from_documents(cls, documents: List[Document], embeddings: Any) -> "FAISSIndexManager":
        """Builds a new FAISS index from a list of Document objects."""
        if not documents:
            raise ValueError("Cannot create FAISS index from an empty list of documents.")
        store = FAISS.from_documents(documents, embeddings)
        logger.info(f"Built FAISS index with {len(documents)} documents.")
        return cls(vector_store=store, embeddings=embeddings)

    def add_documents(self, documents: List[Document]) -> None:
        """Adds new documents incrementally to the existing index."""
        if not self.vector_store:
            raise RuntimeError("FAISS vector store is not initialized.")
        if not documents:
            return
        self.vector_store.add_documents(documents)
        logger.info(f"Added {len(documents)} new documents to existing FAISS index.")

    def search_similarity(
        self,
        query: str,
        k: int = 4,
        score_threshold: float = 0.0
    ) -> List[Tuple[Document, float]]:
        """
        Executes similarity search returning documents with relevance scores.
        Filters out matches if score_threshold is set (> 0.0).
        """
        if not self.vector_store:
            return []

        results_with_scores = self.vector_store.similarity_search_with_score(query, k=k)
        
        filtered = []
        for doc, score in results_with_scores:
            # Note: FAISS default index uses L2 distance (0 = identical, higher = further)
            # When cosine similarity or normalized vectors are used, score represents distance.
            # If a strict score threshold is specified and score exceeds it, filter out.
            if score_threshold > 0.0 and score > score_threshold:
                continue
            filtered.append((doc, float(score)))

        return filtered

    def search_mmr(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.7
    ) -> List[Document]:
        """
        Executes Maximum Marginal Relevance (MMR) search to optimize for relevance
        while penalizing redundancy and promoting informational diversity.
        """
        if not self.vector_store:
            return []

        try:
            return self.vector_store.max_marginal_relevance_search(
                query=query,
                k=k,
                fetch_k=min(fetch_k, len(self.get_all_documents()) or fetch_k),
                lambda_mult=lambda_mult
            )
        except Exception as e:
            logger.warning(f"MMR search fallback to similarity search due to: {e}")
            return self.vector_store.similarity_search(query, k=k)

    def get_all_documents(self) -> List[Document]:
        """Returns all documents contained in the docstore."""
        if not self.vector_store or not hasattr(self.vector_store, "docstore"):
            return []
        try:
            return list(self.vector_store.docstore._dict.values())
        except Exception:
            return []

    def get_indexed_sources(self) -> List[str]:
        """Returns unique source document filenames present in the vector store."""
        docs = self.get_all_documents()
        sources = set()
        for d in docs:
            src = d.metadata.get("source")
            if src:
                sources.add(src)
        return sorted(list(sources))


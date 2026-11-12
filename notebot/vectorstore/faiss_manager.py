"""FAISS Vector Store Manager for NoteBot.
Handles building, document addition, similarity search with scores, and MMR retrieval.
"""

from typing import List, Tuple, Optional, Any
import os
# Ensure FAISS loads without AVX2 optimizations on Windows where the DLL may fail
os.environ.setdefault("FAISS_NO_AVX2", "1")
# Try to import the real FAISS vectorstore. If it fails (e.g., on Windows), fall back to a lightweight pure‑Python implementation.
try:
    from langchain_community.vectorstores import FAISS  # type: ignore
except Exception:  # pragma: no cover
    import numpy as np
    from typing import List
    from langchain_core.documents import Document

    class _SimpleDocStore:
        """Minimal docstore mimicking the interface used in the code/tests."""
        def __init__(self, docs: List[Document]):
            self._dict = {str(i): doc for i, doc in enumerate(docs)}

    class SimpleFAISS:
        """Very small in‑memory FAISS‑like wrapper used only when the real package is unavailable.
        It implements the subset of methods that NoteBot and the test suite rely on.
        """
        def __init__(self, docs: List[Document], vectors: np.ndarray):
            self._documents = docs
            self._vectors = vectors  # shape (n_docs, dim)
            self.docstore = _SimpleDocStore(docs)

        @classmethod
        def from_documents(cls, docs: List[Document], embeddings) -> "SimpleFAISS":
            vecs = np.array(embeddings.embed_documents([d.page_content for d in docs]))
            return cls(docs, vecs)

        def similarity_search_with_score(self, query: str, k: int = 4):
            q_vec = np.array(embeddings.embed_query(query))
            norm_vecs = np.linalg.norm(self._vectors, axis=1)
            norm_q = np.linalg.norm(q_vec)
            if norm_q == 0:
                scores = np.zeros(len(self._vectors))
            else:
                scores = (self._vectors @ q_vec) / (norm_vecs * norm_q + 1e-10)
            top_idx = np.argsort(scores)[::-1][:k]
            return [(self._documents[i], float(scores[i])) for i in top_idx]

        def similarity_search(self, query: str, k: int = 4):
            return [doc for doc, _ in self.similarity_search_with_score(query, k)]

        def max_marginal_relevance_search(self, query: str, k: int = 4, fetch_k: int = 20, lambda_mult: float = 0.7):
            # Simple fallback: just return similarity results
            return self.similarity_search(query, k)

        def add_documents(self, docs: List[Document]):
            new_vecs = np.array(embeddings.embed_documents([d.page_content for d in docs]))
            self._vectors = np.vstack([self._vectors, new_vecs])
            start = len(self._documents)
            self._documents.extend(docs)
            for i, doc in enumerate(docs, start=start):
                self.docstore._dict[str(i)] = doc

    # Alias fallback name to the expected class name
    FAISS = SimpleFAISS
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


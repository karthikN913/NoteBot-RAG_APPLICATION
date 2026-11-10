"""Reranking Engine for NoteBot using FlashRank Cross-Encoder."""

from typing import List, Optional
from langchain_core.documents import Document
from notebot.config.logging_config import setup_logger

logger = setup_logger("notebot.reranker")

try:
    from flashrank import Ranker, RerankRequest
    HAS_FLASHRANK = True
except ImportError:
    HAS_FLASHRANK = False


class Reranker:
    """Manages cross-encoder reranking over retrieved candidate documents."""

    def __init__(self, model_name: str = "ms-marco-TinyBERT-L-2-v2"):
        self.model_name = model_name
        self._ranker = None
        if HAS_FLASHRANK:
            try:
                self._ranker = Ranker(model_name=model_name, cache_dir="/tmp/flashrank")
                logger.info(f"Initialized FlashRank model: {model_name}")
            except Exception as e:
                logger.warning(f"Could not initialize FlashRank: {e}")

    def rerank(self, query: str, documents: List[Document], top_n: int = 4) -> List[Document]:
        """
        Reranks candidate documents with cross-attention against the user query.
        Returns top_n most relevant documents in descending order of cross-encoder score.
        """
        if not documents:
            return []

        if not HAS_FLASHRANK or not self._ranker:
            # Fallback: pass-through without modification
            return documents[:top_n]

        try:
            passages = [
                {
                    "id": i,
                    "text": doc.page_content,
                    "meta": doc.metadata
                }
                for i, doc in enumerate(documents)
            ]

            request = RerankRequest(query=query, passages=passages)
            results = self._ranker.rerank(request)

            reranked_docs = []
            for item in results[:top_n]:
                doc = Document(
                    page_content=item["text"],
                    metadata=dict(item.get("meta", {}))
                )
                doc.metadata["rerank_score"] = float(item.get("score", 0.0))
                reranked_docs.append(doc)

            return reranked_docs

        except Exception as e:
            logger.warning(f"FlashRank reranking failed ({e}); falling back to retriever order.")
            return documents[:top_n]


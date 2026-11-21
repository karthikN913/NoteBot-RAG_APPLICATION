"""Reranking Engine for NoteBot using FlashRank Cross-Encoder."""

import os
import tempfile
from typing import List, Optional
from langchain_core.documents import Document
from notebot.config.logging_config import setup_logger

logger = setup_logger("notebot.reranker")

try:
    from flashrank import Ranker, RerankRequest
    HAS_FLASHRANK = True
except ImportError:
    HAS_FLASHRANK = False


_CACHED_RANKERS = {}


class Reranker:
    """Manages cross-encoder reranking over retrieved candidate documents with lazy initialization."""

    def __init__(self, model_name: str = "ms-marco-TinyBERT-L-2-v2"):
        self.model_name = model_name

    def _get_ranker(self):
        """Lazily instantiates and caches the FlashRank model singleton."""
        if not HAS_FLASHRANK:
            return None
        if self.model_name not in _CACHED_RANKERS:
            try:
                cache_dir = os.path.join(tempfile.gettempdir(), "flashrank")
                _CACHED_RANKERS[self.model_name] = Ranker(model_name=self.model_name, cache_dir=cache_dir)
                logger.info(f"Initialized FlashRank model: {self.model_name}")
            except Exception as e:
                logger.warning(f"Could not initialize FlashRank: {e}")
                return None
        return _CACHED_RANKERS.get(self.model_name)

    def rerank(self, query: str, documents: List[Document], top_n: int = 4) -> List[Document]:
        """
        Reranks candidate documents with cross-attention against the user query.
        Returns top_n most relevant documents in descending order of cross-encoder score.
        """
        if not documents:
            return []

        ranker = self._get_ranker()
        if not ranker:
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
            results = ranker.rerank(request)

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


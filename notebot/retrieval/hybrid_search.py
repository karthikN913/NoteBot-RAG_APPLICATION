"""BM25 + FAISS Hybrid Search and Reciprocal Rank Fusion (RRF) for NoteBot."""

import re
from typing import List, Dict, Tuple, Optional
from langchain_core.documents import Document
from notebot.config.logging_config import setup_logger

logger = setup_logger("notebot.hybrid_search")

try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False


def tokenize_text(text: str) -> List[str]:
    """Tokenizes text into lowercase alphanumeric terms for BM25 indexing."""
    return re.findall(r'\b[a-zA-Z0-9_]+\b', text.lower())


class BM25Index:
    """Manages sparse lexical BM25 index over a corpus of documents."""

    def __init__(self, documents: List[Document]):
        self.documents = documents
        self.corpus_tokens = [tokenize_text(doc.page_content) for doc in documents]
        if HAS_BM25 and self.corpus_tokens:
            self.bm25 = BM25Okapi(self.corpus_tokens)
        else:
            self.bm25 = None

    def search(self, query: str, top_k: int = 10) -> List[Tuple[Document, float]]:
        """Searches BM25 index and returns top-K documents with BM25 scores."""
        if not HAS_BM25 or not self.bm25 or not self.documents:
            return []

        query_tokens = tokenize_text(query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)
        scored_pairs = list(zip(self.documents, scores))
        # Sort descending by score
        scored_pairs.sort(key=lambda x: x[1], reverse=True)
        return scored_pairs[:top_k]


def reciprocal_rank_fusion(
    dense_results: List[Document],
    sparse_results: List[Document],
    k: int = 60,
    top_n: int = 4
) -> List[Document]:
    """
    Fuses rankings from Dense Vector Search and Sparse BM25 Keyword Search
    using standard Reciprocal Rank Fusion (RRF).
    
    RRF Score formula:
        Score(d) = sum( 1.0 / (k + rank_i(d)) ) for each retriever i
    """
    scores: Dict[str, float] = {}
    doc_map: Dict[str, Document] = {}

    def get_doc_id(doc: Document) -> str:
        # Use chunk_id if available, otherwise hash of content
        return doc.metadata.get("chunk_id", str(hash(doc.page_content)))

    # Process dense rankings
    for rank, doc in enumerate(dense_results, 1):
        doc_id = get_doc_id(doc)
        doc_map[doc_id] = doc
        scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (k + rank))

    # Process sparse rankings
    for rank, doc in enumerate(sparse_results, 1):
        doc_id = get_doc_id(doc)
        doc_map[doc_id] = doc
        scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (k + rank))

    # Sort descending by fused RRF score
    sorted_doc_ids = sorted(scores.keys(), key=lambda did: scores[did], reverse=True)
    return [doc_map[did] for did in sorted_doc_ids[:top_n]]

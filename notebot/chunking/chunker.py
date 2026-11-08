"""Metadata-Preserving Text Chunking Module for NoteBot."""

import hashlib
from typing import List, Set
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from notebot.config.logging_config import setup_logger

logger = setup_logger("notebot.chunker")


def chunk_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    deduplicate: bool = True
) -> List[Document]:
    """
    Splits documents into coherent chunks while strictly preserving source metadata.
    
    Features:
    - Structure-aware separators breaking at paragraphs and sentences.
    - Preserves source filename and page numbers.
    - Generates deterministic chunk_id metadata (e.g., 'document.pdf_p2_c0').
    - Optional deduplication to eliminate repeated header/footer artifacts across pages.
    """
    if not documents:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]
    )

    all_chunks: List[Document] = []
    seen_hashes: Set[str] = set()

    # Track chunk counts per (source, page)
    counter_map = {}

    for doc in documents:
        split_texts = splitter.split_text(doc.page_content)
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", 1)

        key = f"{source}_p{page}"
        if key not in counter_map:
            counter_map[key] = 0

        for text in split_texts:
            clean_chunk = text.strip()
            if not clean_chunk or len(clean_chunk) < 15:
                continue

            # Optional deduplication using content hash
            if deduplicate:
                chunk_hash = hashlib.md5(clean_chunk.encode("utf-8")).hexdigest()
                if chunk_hash in seen_hashes:
                    continue
                seen_hashes.add(chunk_hash)

            c_idx = counter_map[key]
            counter_map[key] += 1

            chunk_meta = dict(doc.metadata)
            chunk_meta.update({
                "chunk_id": f"{source}_p{page}_c{c_idx}",
                "chunk_index": c_idx,
                "char_count": len(clean_chunk)
            })

            all_chunks.append(Document(page_content=clean_chunk, metadata=chunk_meta))

    logger.info(f"Created {len(all_chunks)} chunks from {len(documents)} page documents.")
    return all_chunks

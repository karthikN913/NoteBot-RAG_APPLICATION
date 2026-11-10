"""Multi-PDF Document Ingestion Engine for NoteBot.
Handles text extraction, page metadata preservation, corruption checks, and OCR fallback.
"""

import io
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any

try:
    from pypdf import PdfReader
except ImportError:
    from PyPDF2 import PdfReader

from langchain_core.documents import Document
from notebot.ingestion.text_cleaner import clean_pdf_text
from notebot.ingestion.ocr_processor import is_scanned_page, ocr_page_images
from notebot.config.logging_config import setup_logger

logger = setup_logger("notebot.ingestion")


@dataclass
class IngestionResult:
    """Result of ingesting one or more PDF documents."""
    documents: List[Document] = field(default_factory=list)
    total_pages: int = 0
    total_characters: int = 0
    processed_files: List[str] = field(default_factory=list)
    skipped_pages: Dict[str, List[int]] = field(default_factory=dict)
    scanned_pages: Dict[str, List[int]] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    ocr_applied_pages: Dict[str, List[int]] = field(default_factory=dict)


def load_single_pdf(
    file_bytes: bytes,
    filename: str,
    enable_ocr: bool = False,
    min_page_chars: int = 40
) -> Tuple[List[Document], int, List[int], List[int], List[int], Optional[str]]:
    """
    Parses a single PDF byte stream into LangChain Document objects per page.
    
    Returns:
        (docs, total_pages, skipped_pages, scanned_pages, ocr_pages, error)
    """
    if not file_bytes:
        return [], 0, [], [], [], f"File '{filename}' is empty (0 bytes)."

    try:
        reader = PdfReader(io.BytesIO(file_bytes))
    except Exception as e:
        return [], 0, [], [], [], f"Corrupted or invalid PDF format in '{filename}': {str(e)}"

    total_pages = len(reader.pages)
    if total_pages == 0:
        return [], 0, [], [], [], f"PDF '{filename}' contains 0 pages."

    page_docs: List[Document] = []
    skipped_pages: List[int] = []
    scanned_pages: List[int] = []
    ocr_pages: List[int] = []

    for page_idx, page in enumerate(reader.pages):
        page_num = page_idx + 1
        raw_text = ""
        try:
            raw_text = page.extract_text() or ""
        except Exception as e:
            logger.warning(f"Error extracting text from page {page_num} of {filename}: {e}")

        cleaned = clean_pdf_text(raw_text)

        # Check if page is empty or appears scanned
        if is_scanned_page(cleaned, min_chars=min_page_chars):
            scanned_pages.append(page_num)

            # Optional OCR fallback
            if enable_ocr:
                ocr_text, ocr_err = ocr_page_images(page)
                if ocr_text:
                    cleaned = clean_pdf_text(ocr_text)
                    ocr_pages.append(page_num)
                elif ocr_err:
                    logger.info(f"Page {page_num} OCR note: {ocr_err}")

        if not cleaned:
            skipped_pages.append(page_num)
            continue

        doc = Document(
            page_content=cleaned,
            metadata={
                "source": filename,
                "page": page_num,
                "total_pages": total_pages,
                "char_count": len(cleaned)
            }
        )
        page_docs.append(doc)

    return page_docs, total_pages, skipped_pages, scanned_pages, ocr_pages, None


def ingest_documents(
    files: List[Tuple[str, bytes]],
    enable_ocr: bool = False,
    min_page_chars: int = 40
) -> IngestionResult:
    """
    Ingests multiple PDF documents, producing standardized LangChain Document pages.
    """
    result = IngestionResult()

    for filename, file_bytes in files:
        docs, page_count, skipped, scanned, ocr_done, err = load_single_pdf(
            file_bytes=file_bytes,
            filename=filename,
            enable_ocr=enable_ocr,
            min_page_chars=min_page_chars
        )

        if err:
            result.errors.append(err)
            continue

        result.documents.extend(docs)
        result.total_pages += page_count
        result.processed_files.append(filename)

        if skipped:
            result.skipped_pages[filename] = skipped
        if scanned:
            result.scanned_pages[filename] = scanned
        if ocr_done:
            result.ocr_applied_pages[filename] = ocr_done

    result.total_characters = sum(len(d.page_content) for d in result.documents)
    return result


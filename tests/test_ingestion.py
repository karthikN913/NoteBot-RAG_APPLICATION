"""Unit Tests for Document Ingestion, Text Cleaning, and OCR Detection."""

import pytest
from notebot.ingestion.text_cleaner import clean_pdf_text
from notebot.ingestion.ocr_processor import is_scanned_page
from notebot.ingestion.pdf_loader import load_single_pdf, ingest_documents
from notebot.utils.validators import validate_uploaded_file


def test_clean_pdf_text_hyphenation():
    raw = "The deep learn-\ning model performed well on natural lan-\nguage processing."
    cleaned = clean_pdf_text(raw)
    assert "learning" in cleaned
    assert "language" in cleaned
    assert "learn-\ning" not in cleaned


def test_clean_pdf_text_blank_lines_and_whitespace():
    raw = "Paragraph 1\n\n\n\n\nParagraph 2\r\n\r\nParagraph 3\x0c"
    cleaned = clean_pdf_text(raw)
    assert "\n\n\n" not in cleaned
    assert "Paragraph 1\n\nParagraph 2\n\nParagraph 3" == cleaned


def test_is_scanned_page_detection():
    assert is_scanned_page("", min_chars=40) is True
    assert is_scanned_page("   ", min_chars=40) is True
    assert is_scanned_page("Short title", min_chars=40) is True
    assert is_scanned_page("This page contains a substantial paragraph describing machine learning algorithms in full detail.", min_chars=40) is False


def test_validate_uploaded_file():
    # Empty file
    valid, err = validate_uploaded_file("empty.pdf", b"")
    assert valid is False
    assert "empty" in err.lower()

    # Invalid extension
    valid, err = validate_uploaded_file("notes.docx", b"%PDF-1.4 test content")
    assert valid is False
    assert "extension" in err.lower()

    # Invalid magic header
    valid, err = validate_uploaded_file("fake.pdf", b"NOT_A_PDF_HEADER")
    assert valid is False
    assert "magic" in err.lower() or "%pdf-" in err.lower()

    # Valid PDF signature
    valid, err = validate_uploaded_file("good.pdf", b"%PDF-1.7 valid content")
    assert valid is True
    assert err is None


def test_load_corrupted_pdf():
    docs, total_pages, skipped, scanned, ocr, err = load_single_pdf(
        file_bytes=b"CORRUPTED_BYTES",
        filename="corrupt.pdf"
    )
    assert len(docs) == 0
    assert err is not None
    assert "corrupted" in err.lower() or "invalid" in err.lower()


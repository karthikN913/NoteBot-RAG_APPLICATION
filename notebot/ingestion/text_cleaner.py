"""Text Cleaning and Normalization Utilities for NoteBot Document Ingestion."""

import re


def clean_pdf_text(text: str) -> str:
    """
    Cleans raw text extracted from PDF documents.
    
    1. Normalizes line breaks and removes non-printable / control characters.
    2. Heals hyphenated words split across line breaks (e.g., 'trans-\nformer' -> 'transformer').
    3. Collapses excessive consecutive blank lines into readable paragraphs.
    4. Strips leading and trailing whitespace.
    """
    if not text:
        return ""

    # Normalize line endings and remove null bytes / form feeds
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\x0c", "\n").replace("\x00", "")

    # Heal hyphenated words split across line wraps
    text = re.sub(r'(\b[a-zA-Z]{2,})-\n([a-zA-Z]{2,}\b)', r'\1\2', text)

    # Replace horizontal tabs with spaces
    text = text.replace("\t", " ")

    # Collapse sequences of multiple spaces within a line
    text = re.sub(r'[ ]{2,}', ' ', text)

    # Collapse three or more newlines into double newlines (paragraph separator)
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()

 
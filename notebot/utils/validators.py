"""Input Validation and Security Safeguards for NoteBot."""

import os
from typing import Tuple, Optional

# Security thresholds
MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024  # 100 MB max file size
ALLOWED_EXTENSIONS = {".pdf"}


def validate_uploaded_file(filename: str, file_bytes: bytes) -> Tuple[bool, Optional[str]]:
    """
    Validates uploaded file against security checks:
    - Non-empty byte stream
    - Permitted file extension (.pdf)
    - File size limit (100MB)
    - PDF header magic byte check (%PDF-)
    """
    if not file_bytes or len(file_bytes) == 0:
        return False, f"File '{filename}' is empty (0 bytes)."

    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        mb = len(file_bytes) / (1024 * 1024)
        return False, f"File '{filename}' ({mb:.1f} MB) exceeds maximum allowed size of 100 MB."

    _, ext = os.path.splitext(filename)
    if ext.lower() not in ALLOWED_EXTENSIONS:
        return False, f"Invalid file extension '{ext}'. Only PDF files are supported."

    # Validate PDF magic header
    if not file_bytes.startswith(b"%PDF-"):
        return False, f"File '{filename}' does not appear to be a valid PDF document (missing %PDF- header)."

    return True, None


def sanitize_user_query(query: str) -> str:
    """Sanitizes user input query string."""
    if not query:
        return ""
    # Strip null bytes and non-printable control characters
    cleaned = "".join(ch for ch in query if ch.isprintable() or ch in ("\n", "\t"))
    return cleaned.strip()


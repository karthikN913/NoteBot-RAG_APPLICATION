"""OCR Processing & Scanned Document Detection for NoteBot."""

import io
from typing import Optional, List, Tuple
from notebot.config.logging_config import setup_logger

logger = setup_logger("notebot.ocr")

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import pytesseract
    HAS_PYTESSERACT = True
except ImportError:
    HAS_PYTESSERACT = False


def is_scanned_page(text: str, min_chars: int = 40) -> bool:
    """Returns True if the extracted text length is suspiciously low (scanned or image-based)."""
    return len(text.strip()) < min_chars


def ocr_page_images(page_obj) -> Tuple[str, Optional[str]]:
    """
    Attempts to perform OCR on any embedded images inside a PDF page object.
    
    Returns:
        (extracted_text, error_message)
    """
    if not HAS_PIL:
        return "", "Pillow (PIL) is not installed; OCR image decoding unavailable."
    if not HAS_PYTESSERACT:
        return "", "pytesseract is not installed; OCR text extraction unavailable."

    ocr_texts = []
    try:
        images = getattr(page_obj, "images", [])
        if not images:
            return "", "No embedded images found on this scanned page."

        for img in images:
            try:
                pil_image = Image.open(io.BytesIO(img.data))
                text = pytesseract.image_to_string(pil_image)
                if text and text.strip():
                    ocr_texts.append(text.strip())
            except Exception as e:
                logger.warning(f"Failed to OCR individual image: {e}")
                continue

        if not ocr_texts:
            return "", "No readable text could be recognized from page images."

        return "\n".join(ocr_texts), None

    except Exception as e:
        err_msg = str(e)
        if "tesseract is not installed" in err_msg.lower() or "tesseract_cmd" in err_msg.lower():
            return "", "Tesseract-OCR engine binary is not installed or not found in system PATH."
        return "", f"OCR extraction error: {err_msg}"


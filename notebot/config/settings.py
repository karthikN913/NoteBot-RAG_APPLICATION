"""Application Configuration and Provider Settings for NoteBot."""

import os
from dataclasses import dataclass, field
from typing import Optional, Dict, List

# Supported LLM Providers
PROVIDER_GOOGLE = "google"
PROVIDER_OPENAI = "openai"
PROVIDER_GROQ = "groq"
PROVIDER_OPENROUTER = "openrouter"

ALL_LLM_PROVIDERS = [
    PROVIDER_GOOGLE,
    PROVIDER_OPENAI,
    PROVIDER_GROQ,
    PROVIDER_OPENROUTER,
]

# Supported Embedding Providers
EMBED_GOOGLE = "google"
EMBED_OPENAI = "openai"
EMBED_FASTEMBED = "fastembed"

ALL_EMBED_PROVIDERS = [
    EMBED_GOOGLE,
    EMBED_OPENAI,
    EMBED_FASTEMBED,
]

# Dedicated Environment Variable Names per Provider
PROVIDER_ENV_VARS: Dict[str, List[str]] = {
    PROVIDER_GOOGLE: ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
    PROVIDER_OPENAI: ["OPENAI_API_KEY"],
    PROVIDER_GROQ: ["GROQ_API_KEY"],
    PROVIDER_OPENROUTER: ["OPENROUTER_API_KEY"],
}

# Provider Display Names & Helper Metadata
PROVIDER_DISPLAY_NAMES: Dict[str, str] = {
    PROVIDER_GOOGLE: "Google Gemini (gemini-2.5-flash / gemini-flash-latest)",
    PROVIDER_OPENAI: "OpenAI (gpt-4o-mini)",
    PROVIDER_GROQ: "Groq (llama-3.3-70b-versatile)",
    PROVIDER_OPENROUTER: "OpenRouter (meta-llama/llama-3.3-70b)",
}

# Default Model Hierarchies for LLMs
DEFAULT_LLM_MODELS: Dict[str, List[str]] = {
    PROVIDER_GOOGLE: [
        "gemini-2.5-flash",
        "gemini-flash-latest",
        "gemini-2.5-flash-lite",
        "gemini-3.5-flash",
        "gemini-2.5-pro",
    ],
    PROVIDER_OPENAI: [
        "gpt-4o-mini",
        "gpt-4o",
        "gpt-3.5-turbo",
    ],
    PROVIDER_GROQ: [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "mixtral-8x7b-32768",
    ],
    PROVIDER_OPENROUTER: [
        "meta-llama/llama-3.3-70b-instruct:free",
        "google/gemini-2.0-flash-exp:free",
        "mistralai/mistral-7b-instruct:free",
    ],
}

# Embedding Model Defaults
DEFAULT_EMBED_MODELS: Dict[str, List[str]] = {
    EMBED_GOOGLE: [
        "gemini-embedding-001",
        "embedding-001",
        "models/gemini-embedding-001",
        "models/embedding-001",
        "gemini-embedding-2",
        "text-embedding-004",
    ],
    EMBED_OPENAI: [
        "text-embedding-3-small",
        "text-embedding-ada-002",
    ],
    EMBED_FASTEMBED: [
        "BAAI/bge-small-en-v1.5",
    ],
}


def get_provider_api_key(provider: str, user_override_key: Optional[str] = None) -> Optional[str]:
    """
    Retrieves the API key strictly designated for the specified provider.
    Never falls back to an unrelated provider's key.
    
    Order of precedence:
    1. Direct user runtime input (if non-empty)
    2. Specific environment variable for this provider
    3. Streamlit secrets (if available)
    """
    if user_override_key and user_override_key.strip():
        return user_override_key.strip()

    env_var_names = PROVIDER_ENV_VARS.get(provider.lower(), [])
    for var in env_var_names:
        val = os.getenv(var)
        if val and val.strip():
            return val.strip()

    # Attempt to load from streamlit secrets safely without crashing
    try:
        import streamlit as st
        for var in env_var_names:
            if hasattr(st, "secrets") and var in st.secrets:
                val = st.secrets[var]
                if val and str(val).strip():
                    return str(val).strip()
    except Exception:
        pass

    return None


def detect_provider_from_key(key: str) -> str:
    """Detects provider by key structural prefix."""
    if not key:
        return PROVIDER_GOOGLE
    k = key.strip()
    if k.startswith("AIza") or k.startswith("AQ."):
        return PROVIDER_GOOGLE
    elif k.startswith("gsk_"):
        return PROVIDER_GROQ
    elif k.startswith("sk-or-"):
        return PROVIDER_OPENROUTER
    elif k.startswith("sk-") or k.startswith("sk-proj-"):
        return PROVIDER_OPENAI
    return PROVIDER_GOOGLE if "." in k else PROVIDER_OPENAI


@dataclass
class AppConfig:
    """Runtime Configuration State for NoteBot."""
    # LLM Settings
    llm_provider: str = PROVIDER_GOOGLE
    llm_model: Optional[str] = None
    temperature: float = 0.3
    max_output_tokens: int = 2048

    # Embedding Settings (Decoupled from LLM Provider)
    embedding_provider: str = EMBED_GOOGLE
    embedding_model: Optional[str] = None

    # Chunking Settings
    chunk_size: int = 1000
    chunk_overlap: int = 150

    # Retrieval Settings
    retrieval_mode: str = "similarity"  # "similarity", "mmr", "hybrid"
    top_k: int = 4
    score_threshold: float = 0.0
    enable_reranker: bool = False

    # OCR Settings
    enable_ocr_fallback: bool = False
    ocr_min_char_count: int = 40  # If page text < 40 chars, consider scanned

    # Persistence Settings
    index_storage_dir: str = "data/indexes"


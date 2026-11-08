"""Independent Embedding Model Factory for NoteBot.
Supports Google Gemini, OpenAI, and local offline FastEmbed.
"""

from typing import Optional, Any
from notebot.config.settings import (
    EMBED_GOOGLE,
    EMBED_OPENAI,
    EMBED_FASTEMBED,
    DEFAULT_EMBED_MODELS
)
from notebot.config.logging_config import setup_logger

logger = setup_logger("notebot.embeddings")

try:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    HAS_GOOGLE_EMBED = True
except ImportError:
    HAS_GOOGLE_EMBED = False

try:
    from langchain_openai import OpenAIEmbeddings
    HAS_OPENAI_EMBED = True
except ImportError:
    HAS_OPENAI_EMBED = False

try:
    from langchain_community.embeddings import FastEmbedEmbeddings
    HAS_FASTEMBED = True
except ImportError:
    HAS_FASTEMBED = False


def get_embedding_model(
    provider: str = EMBED_GOOGLE,
    key: Optional[str] = None,
    model_name: Optional[str] = None,
    allow_offline_fallback: bool = True
) -> Any:
    """
    Instantiates an embedding model for vector generation and similarity search.
    Completely decoupled from the chosen LLM provider.
    """
    prov = provider.lower()
    last_err: Optional[Exception] = None

    # 1. Google Gemini Embeddings
    if prov == EMBED_GOOGLE:
        if not HAS_GOOGLE_EMBED:
            raise ImportError("Package 'langchain-google-genai' is required for Google Gemini embeddings.")
        if not key:
            raise ValueError("Google Gemini API Key is required for Gemini embeddings.")

        models_to_try = [model_name] if model_name else DEFAULT_EMBED_MODELS[EMBED_GOOGLE]
        for m in models_to_try:
            try:
                embed_instance = GoogleGenerativeAIEmbeddings(
                    model=m,
                    google_api_key=key
                )
                logger.info(f"Initialized Google Gemini embedding model: {m}")
                return embed_instance
            except Exception as e:
                last_err = e
                err_str = str(e).lower()
                if "api_key_invalid" in err_str or "unregistered" in err_str or "400" in err_str:
                    break

        if allow_offline_fallback and HAS_FASTEMBED:
            logger.warning(f"Google embedding failed ({last_err}); falling back to local FastEmbed.")
            return FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")

        raise RuntimeError(f"Failed to initialize Google Gemini embeddings: {last_err}")

    # 2. OpenAI Embeddings
    elif prov == EMBED_OPENAI:
        if not HAS_OPENAI_EMBED:
            raise ImportError("Package 'langchain-openai' is required for OpenAI embeddings.")
        if not key:
            raise ValueError("OpenAI API Key is required for OpenAI embeddings.")

        models_to_try = [model_name] if model_name else DEFAULT_EMBED_MODELS[EMBED_OPENAI]
        for m in models_to_try:
            try:
                embed_instance = OpenAIEmbeddings(
                    api_key=key,
                    model=m
                )
                logger.info(f"Initialized OpenAI embedding model: {m}")
                return embed_instance
            except Exception as e:
                last_err = e
                err_str = str(e).lower()
                if "invalid_api_key" in err_str or "401" in err_str:
                    break

        if allow_offline_fallback and HAS_FASTEMBED:
            logger.warning(f"OpenAI embedding failed ({last_err}); falling back to local FastEmbed.")
            return FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")

        raise RuntimeError(f"Failed to initialize OpenAI embeddings: {last_err}")

    # 3. Local Offline FastEmbed
    elif prov == EMBED_FASTEMBED:
        if not HAS_FASTEMBED:
            raise ImportError("Package 'fastembed' is required for local offline embeddings.")
        chosen_model = model_name or DEFAULT_EMBED_MODELS[EMBED_FASTEMBED][0]
        logger.info(f"Initialized local FastEmbed model: {chosen_model}")
        return FastEmbedEmbeddings(model_name=chosen_model)

    else:
        raise ValueError(f"Unsupported embedding provider: '{provider}'. Supported: google, openai, fastembed.")

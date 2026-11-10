"""Independent Embedding Model Factory for NoteBot.
Supports Google Gemini, OpenAI, and local offline FastEmbed with automatic fallback.
"""

from typing import Optional, Any, List
from langchain_core.embeddings import Embeddings
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


class ResilientGoogleEmbeddings(Embeddings):
    """
    Self-healing Google Gemini embeddings wrapper.
    Iterates through supported Gemini embedding models dynamically.
    If a model returns 404 NOT_FOUND, deprecated, or quota exhausted,
    it automatically tries the next supported model, and if all remote
    models fail, seamlessly falls back to FastEmbed so ingestion never crashes.
    """

    def __init__(
        self,
        key: str,
        models: Optional[List[str]] = None,
        allow_offline_fallback: bool = True
    ):
        self.key = key
        self.models = models or DEFAULT_EMBED_MODELS.get(EMBED_GOOGLE, [])
        self.allow_offline_fallback = allow_offline_fallback
        self._active_embedder: Optional[Embeddings] = None
        self.active_model_name: Optional[str] = None

    def _resolve_active_embedder(self, sample_text: str = "Validation probe") -> Embeddings:
        """Determines and caches the best functioning embedder."""
        if self._active_embedder is not None:
            return self._active_embedder

        last_err: Optional[Exception] = None

        if HAS_GOOGLE_EMBED and self.key:
            for mod in self.models:
                try:
                    candidate = GoogleGenerativeAIEmbeddings(
                        model=mod,
                        google_api_key=self.key
                    )
                    # Verify model endpoint accepts embedContent with a 1-item probe
                    candidate.embed_query(sample_text)
                    self._active_embedder = candidate
                    self.active_model_name = mod
                    logger.info(f"Verified working Google Gemini embedding model: {mod}")
                    return candidate
                except Exception as e:
                    last_err = e
                    logger.warning(f"Google embedding model '{mod}' probe failed ({e}); testing next candidate.")
                    continue

        if self.allow_offline_fallback and HAS_FASTEMBED:
            logger.warning(
                f"All Google Gemini embedding models failed or unavailable ({last_err}). "
                "Engaging local offline FastEmbed (BAAI/bge-small-en-v1.5) fallback."
            )
            fallback = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
            self._active_embedder = fallback
            self.active_model_name = "fastembed-offline-fallback"
            return fallback

        raise RuntimeError(
            f"Google Gemini embedding failed: {last_err}. "
            "Please select FastEmbed in the sidebar for offline embeddings."
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        embedder = self._resolve_active_embedder(texts[0])
        try:
            return embedder.embed_documents(texts)
        except Exception as e:
            logger.warning(f"Batch embedding failed with active model ({e}); re-resolving fallback...")
            self._active_embedder = None
            embedder = self._resolve_active_embedder(texts[0])
            return embedder.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        embedder = self._resolve_active_embedder(text)
        try:
            return embedder.embed_query(text)
        except Exception as e:
            logger.warning(f"Query embedding failed with active model ({e}); re-resolving fallback...")
            self._active_embedder = None
            embedder = self._resolve_active_embedder(text)
            return embedder.embed_query(text)


class ResilientOpenAIEmbeddings(Embeddings):
    """
    Self-healing OpenAI embeddings wrapper with graceful FastEmbed fallback.
    """

    def __init__(
        self,
        key: str,
        models: Optional[List[str]] = None,
        allow_offline_fallback: bool = True
    ):
        self.key = key
        self.models = models or DEFAULT_EMBED_MODELS.get(EMBED_OPENAI, [])
        self.allow_offline_fallback = allow_offline_fallback
        self._active_embedder: Optional[Embeddings] = None
        self.active_model_name: Optional[str] = None

    def _resolve_active_embedder(self, sample_text: str = "Validation probe") -> Embeddings:
        if self._active_embedder is not None:
            return self._active_embedder

        last_err: Optional[Exception] = None

        if HAS_OPENAI_EMBED and self.key:
            for mod in self.models:
                try:
                    candidate = OpenAIEmbeddings(
                        model=mod,
                        api_key=self.key
                    )
                    candidate.embed_query(sample_text)
                    self._active_embedder = candidate
                    self.active_model_name = mod
                    logger.info(f"Verified working OpenAI embedding model: {mod}")
                    return candidate
                except Exception as e:
                    last_err = e
                    logger.warning(f"OpenAI embedding model '{mod}' probe failed ({e}); testing next candidate.")
                    continue

        if self.allow_offline_fallback and HAS_FASTEMBED:
            logger.warning(f"OpenAI embedding failed ({last_err}); falling back to local FastEmbed.")
            fallback = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
            self._active_embedder = fallback
            self.active_model_name = "fastembed-offline-fallback"
            return fallback

        raise RuntimeError(f"OpenAI embedding failed: {last_err}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        embedder = self._resolve_active_embedder(texts[0])
        try:
            return embedder.embed_documents(texts)
        except Exception as e:
            logger.warning(f"Batch embedding failed with active model ({e}); re-resolving fallback...")
            self._active_embedder = None
            embedder = self._resolve_active_embedder(texts[0])
            return embedder.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        embedder = self._resolve_active_embedder(text)
        try:
            return embedder.embed_query(text)
        except Exception as e:
            logger.warning(f"Query embedding failed with active model ({e}); re-resolving fallback...")
            self._active_embedder = None
            embedder = self._resolve_active_embedder(text)
            return embedder.embed_query(text)


def get_embedding_model(
    provider: str = EMBED_GOOGLE,
    key: Optional[str] = None,
    model_name: Optional[str] = None,
    allow_offline_fallback: bool = True
) -> Embeddings:
    """
    Instantiates an embedding model for vector generation and similarity search.
    Completely decoupled from the chosen LLM provider.
    """
    prov = provider.lower()

    # 1. Google Gemini Embeddings (Resilient & Self-Healing)
    if prov == EMBED_GOOGLE:
        if not HAS_GOOGLE_EMBED:
            raise ImportError("Package 'langchain-google-genai' is required for Google Gemini embeddings.")
        if not key:
            raise ValueError("Google Gemini API Key is required for Gemini embeddings.")

        models_to_try = [model_name] if model_name else DEFAULT_EMBED_MODELS[EMBED_GOOGLE]
        return ResilientGoogleEmbeddings(
            key=key,
            models=models_to_try,
            allow_offline_fallback=allow_offline_fallback
        )

    # 2. OpenAI Embeddings (Resilient & Self-Healing)
    elif prov == EMBED_OPENAI:
        if not HAS_OPENAI_EMBED:
            raise ImportError("Package 'langchain-openai' is required for OpenAI embeddings.")
        if not key:
            raise ValueError("OpenAI API Key is required for OpenAI embeddings.")

        models_to_try = [model_name] if model_name else DEFAULT_EMBED_MODELS[EMBED_OPENAI]
        return ResilientOpenAIEmbeddings(
            key=key,
            models=models_to_try,
            allow_offline_fallback=allow_offline_fallback
        )

    # 3. Local Offline FastEmbed
    elif prov == EMBED_FASTEMBED:
        if not HAS_FASTEMBED:
            raise ImportError("Package 'fastembed' is required for local offline embeddings.")
        chosen_model = model_name or DEFAULT_EMBED_MODELS[EMBED_FASTEMBED][0]
        logger.info(f"Initialized local FastEmbed model: {chosen_model}")
        return FastEmbedEmbeddings(model_name=chosen_model)

    else:
        raise ValueError(f"Unsupported embedding provider: '{provider}'. Supported: google, openai, fastembed.")

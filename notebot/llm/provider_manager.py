"""Provider Configuration, Validation, and Model Management for NoteBot."""

import os
from typing import Optional, List, Tuple, Any
from notebot.config.settings import (
    PROVIDER_GOOGLE,
    PROVIDER_OPENAI,
    PROVIDER_GROQ,
    PROVIDER_OPENROUTER,
    DEFAULT_LLM_MODELS,
    get_provider_api_key
)
from notebot.config.logging_config import setup_logger

logger = setup_logger("notebot.provider_manager")

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    HAS_GOOGLE = True
except ImportError:
    HAS_GOOGLE = False

try:
    from langchain_openai import ChatOpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    from langchain_groq import ChatGroq
    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False


def validate_provider_key(provider: str, key: Optional[str]) -> Tuple[bool, Optional[str]]:
    """
    Validates that the provided key format matches the expected provider.
    Returns (is_valid, error_reason).
    """
    if not key or not key.strip():
        return False, f"API Key for {provider.title()} is missing or empty."

    k = key.strip()
    prov = provider.lower()

    if prov == PROVIDER_GOOGLE:
        if not (k.startswith("AIza") or k.startswith("AQ.") or "." in k or len(k) >= 30):
            return False, "Key format appears invalid for Google Gemini (expected prefix 'AIza...')."
    elif prov == PROVIDER_OPENAI:
        if not (k.startswith("sk-") or k.startswith("sk-proj-")):
            return False, "Key format appears invalid for OpenAI (expected prefix 'sk-...')."
    elif prov == PROVIDER_GROQ:
        if not k.startswith("gsk_"):
            return False, "Key format appears invalid for Groq (expected prefix 'gsk_...')."
    elif prov == PROVIDER_OPENROUTER:
        if not k.startswith("sk-or-"):
            return False, "Key format appears invalid for OpenRouter (expected prefix 'sk-or-...')."

    return True, None


def get_llm_instance(
    provider: str,
    model_name: str,
    key: str,
    temperature: float = 0.3,
    max_tokens: int = 2048,
    streaming: bool = True
) -> Any:
    """
    Instantiates an LLM instance for inference.
    """
    prov = provider.lower()

    if prov == PROVIDER_GOOGLE:
        if not HAS_GOOGLE:
            raise ImportError("Package 'langchain-google-genai' is required for Google Gemini.")
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=key,
            temperature=temperature,
            max_output_tokens=max_tokens,
            streaming=streaming
        )

    elif prov == PROVIDER_GROQ:
        if not HAS_GROQ:
            raise ImportError("Package 'langchain-groq' is required for Groq models.")
        return ChatGroq(
            model=model_name,
            groq_api_key=key,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=streaming
        )

    elif prov == PROVIDER_OPENROUTER:
        if not HAS_OPENAI:
            raise ImportError("Package 'langchain-openai' is required for OpenRouter.")
        return ChatOpenAI(
            api_key=key,
            base_url="https://openrouter.ai/api/v1",
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=streaming
        )

    elif prov == PROVIDER_OPENAI:
        if not HAS_OPENAI:
            raise ImportError("Package 'langchain-openai' is required for OpenAI models.")
        return ChatOpenAI(
            api_key=key,
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=streaming
        )

    else:
        raise ValueError(f"Unsupported provider: '{provider}'")


def classify_api_error(error: Exception) -> str:
    """
    Translates raw exception messages into user-friendly, actionable diagnostic guidance.
    """
    err_str = str(error).lower()
    
    if "429" in err_str or "resource_exhausted" in err_str or "quota" in err_str:
        return "❌ **API Quota Exceeded (429)**: Your rate limit or free tier credits for this provider have been exhausted. Please check your provider account limits or switch to an alternate provider in the sidebar."
    
    if "401" in err_str or "invalid_api_key" in err_str or "api_key_invalid" in err_str or "authentication" in err_str:
        return "❌ **Invalid API Key (401)**: The provided key was rejected by the provider. Please verify your API key in the sidebar."
    
    if "404" in err_str or "not_found" in err_str:
        return f"❌ **Model Unavailable (404)**: The requested model could not be accessed. Details: {error}"
    
    if "connection" in err_str or "getaddrinfo" in err_str or "timed out" in err_str:
        return "❌ **Network Connection Error**: Unable to reach the provider servers. Please verify your internet connection."

    return f"❌ **Inference Error**: {str(error)}"


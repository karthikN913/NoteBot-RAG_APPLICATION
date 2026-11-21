"""Unit Tests for Provider Validation, Key Mapping, and Error Handling."""

import os
import pytest
from notebot.config.settings import (
    PROVIDER_GOOGLE,
    PROVIDER_OPENAI,
    PROVIDER_GROQ,
    PROVIDER_OPENROUTER,
    get_provider_api_key
)
from notebot.llm.provider_manager import (
    validate_provider_key,
    classify_api_error
)


def test_provider_key_mapping(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaSyTestGeminiKey123456789012345")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-TestOpenAIKey123456789012345")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_TestGroqKey123456789012345")

    # Strictly fetches designated key for Google, ignoring OpenAI or Groq
    assert get_provider_api_key(PROVIDER_GOOGLE) == "AIzaSyTestGeminiKey123456789012345"
    assert get_provider_api_key(PROVIDER_OPENAI) == "sk-proj-TestOpenAIKey123456789012345"
    assert get_provider_api_key(PROVIDER_GROQ) == "gsk_TestGroqKey123456789012345"

    # User runtime override takes priority
    assert get_provider_api_key(PROVIDER_GOOGLE, user_override_key="custom_runtime_key") == "custom_runtime_key"


def test_validate_provider_key():
    # Valid formats
    assert validate_provider_key(PROVIDER_GOOGLE, "AIzaSyTest1234567890123456789012345")[0] is True
    assert validate_provider_key(PROVIDER_OPENAI, "sk-proj-1234567890123456789012345678")[0] is True
    assert validate_provider_key(PROVIDER_GROQ, "gsk_1234567890123456789012345678")[0] is True
    assert validate_provider_key(PROVIDER_OPENROUTER, "sk-or-1234567890123456789012345678")[0] is True

    # Empty
    assert validate_provider_key(PROVIDER_GOOGLE, "")[0] is False

    # Mismatched formats
    assert validate_provider_key(PROVIDER_GROQ, "AIzaSyGoogleKey")[0] is False
    assert validate_provider_key(PROVIDER_OPENAI, "gsk_GroqKey")[0] is False


def test_classify_api_error():
    # 429 Rate limit
    msg = classify_api_error(Exception("429 RESOURCE_EXHAUSTED: Quota exceeded for project"))
    assert "429" in msg or "quota" in msg.lower()

    # 401 Invalid key
    msg = classify_api_error(Exception("401 Unauthorized: Invalid API key provided"))
    assert "401" in msg or "invalid" in msg.lower()

    # 404 Model not found
    msg = classify_api_error(Exception("404 NOT_FOUND: models/gemini-1.5-pro is not found"))
    assert "404" in msg or "model" in msg.lower()


def test_get_available_gemini_models():
    from notebot.llm.provider_manager import get_available_gemini_models
    # Without key, returns static hierarchy
    models = get_available_gemini_models()
    assert len(models) >= 3
    assert any("flash" in m for m in models)


def test_stream_rag_response_resilient_fallback_on_404(monkeypatch):
    """Verifies that if the preferred model throws a 404 NOT_FOUND,
    stream_rag_response automatically tries the next candidate in the hierarchy.
    """
    from notebot.llm.response_generator import stream_rag_response
    from langchain_core.documents import Document
    from langchain_core.messages import AIMessage
    from langchain_core.runnables import RunnableLambda

    attempted_models = []

    def mock_get_llm_instance(provider, model_name, key, **kwargs):
        attempted_models.append(model_name)
        if model_name == "gemini-1.5-flash":
            def failing_func(prompt_val):
                raise Exception("404 NOT_FOUND: models/gemini-1.5-flash is not found")
            return RunnableLambda(failing_func)
        else:
            def working_func(prompt_val):
                yield AIMessage(content="Grounded pedagogical explanation")
            return RunnableLambda(working_func)

    monkeypatch.setattr("notebot.llm.response_generator.get_llm_instance", mock_get_llm_instance)

    sample_doc = Document(page_content="Photosynthesis converts light into chemical energy.", metadata={"source": "bio.pdf", "page": 1})
    generator = stream_rag_response(
        provider=PROVIDER_GOOGLE,
        api_key="AIzaSyFakeKey12345",
        query="What is photosynthesis?",
        documents=[sample_doc],
        chat_history=[],
        model_override="gemini-1.5-flash"
    )

    tokens = list(generator)
    assert len(tokens) > 0
    assert "Grounded" in tokens[0]
    # Verify gemini-1.5-flash was attempted first, failed with 404, and automatically fell back to the next model
    assert attempted_models[0] == "gemini-1.5-flash"
    assert len(attempted_models) >= 2



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


"""Unit Tests for Resilient Embedding Manager and FastEmbed Fallback."""

import pytest
from unittest.mock import MagicMock, patch
from notebot.embeddings.embedding_manager import (
    get_embedding_model,
    ResilientGoogleEmbeddings,
    ResilientOpenAIEmbeddings
)
from notebot.config.settings import EMBED_GOOGLE, EMBED_FASTEMBED, EMBED_OPENAI


def test_get_embedding_model_fastembed():
    embedder = get_embedding_model(provider=EMBED_FASTEMBED)
    assert embedder is not None
    # Verify local offline embedding works
    vecs = embedder.embed_documents(["Hello world", "FastEmbed test document"])
    assert len(vecs) == 2
    assert len(vecs[0]) > 0


def test_get_embedding_model_missing_key():
    with pytest.raises(ValueError, match="API Key is required"):
        get_embedding_model(provider=EMBED_GOOGLE, key="")


def test_resilient_google_embeddings_fallback_on_404():
    # Simulate all Google models failing with 404 NOT_FOUND
    resilient = ResilientGoogleEmbeddings(
        key="fake_key_12345",
        models=["gemini-embedding-001", "embedding-001"],
        allow_offline_fallback=True
    )

    with patch("langchain_google_genai.GoogleGenerativeAIEmbeddings.embed_query") as mock_embed:
        mock_embed.side_effect = Exception("404 NOT_FOUND: models/text-embedding-004 is not found")

        # When remote fails with 404, resilient wrapper should fall back to FastEmbed
        res = resilient.embed_documents(["Document testing fallback"])
        assert len(res) == 1
        assert len(res[0]) > 0
        assert resilient.active_model_name == "fastembed-offline-fallback"

"""Unit Tests for Conversational Query Understanding and Rewriting."""

import pytest
from notebot.retrieval.query_rewriter import (
    is_followup_query,
    resolve_search_query_heuristic
)


def test_is_followup_query_pronouns():
    assert is_followup_query("What are its applications?") is True
    assert is_followup_query("Can you elaborate on that?") is True
    assert is_followup_query("Why does this happen?") is True
    assert is_followup_query("Tell me more") is True
    assert is_followup_query("Give an example") is True


def test_is_followup_query_independent():
    assert is_followup_query("What is gradient descent optimization?") is False
    assert is_followup_query("Explain photosynthesis in green plants.") is False
    assert is_followup_query("Define the central limit theorem in statistics.") is False


def test_resolve_search_query_heuristic():
    history = [
        {"role": "user", "content": "What is machine learning?"},
        {"role": "assistant", "content": "Machine learning is a field of AI..."},
        {"role": "user", "content": "What are its applications?"}
    ]

    resolved = resolve_search_query_heuristic(history, "What are its applications?")
    # Should attach prior substantive query as semantic anchor
    assert "What are its applications?" in resolved
    assert "machine learning" in resolved.lower()


def test_resolve_search_query_heuristic_independent_query():
    history = [
        {"role": "user", "content": "What is machine learning?"},
        {"role": "assistant", "content": "Machine learning is a field of AI..."},
        {"role": "user", "content": "Explain cellular respiration in biology."}
    ]

    # Independent query should NOT be contaminated with prior chat history
    resolved = resolve_search_query_heuristic(history, "Explain cellular respiration in biology.")
    assert resolved == "Explain cellular respiration in biology."

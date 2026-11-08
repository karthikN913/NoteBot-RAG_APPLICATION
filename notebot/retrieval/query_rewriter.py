"""Conversational Query Understanding and Contextual Rewriting for NoteBot."""

import re
from typing import List, Dict, Optional, Any
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from notebot.config.logging_config import setup_logger

logger = setup_logger("notebot.query_rewriter")

BACK_REFERENCE_WORDS = {
    'it', 'that', 'these', 'those', 'they', 'them', 'he', 'she',
    'above', 'previous', 'earlier', 'latter', 'former', 'aforementioned',
    'its', 'their', 'this'
}

BACK_REFERENCE_PHRASES = (
    'tell me more', 'how come', 'what about', 'the first point',
    'the second point', 'the third point', 'point 1', 'point 2',
    'you mentioned', 'as you said', 'in that case', 'elaborate on that',
    'give an example', 'can you expand', 'why is that'
)


def is_followup_query(query: str) -> bool:
    """
    Determines whether a query is a dependent follow-up referencing prior conversational context.
    """
    if not query:
        return False
    q_lower = query.lower().strip()
    words = re.findall(r'\b[a-z0-9]+\b', q_lower)

    # Ultra-short queries are almost always follow-ups (e.g., "Why?", "Explain more", "And then?")
    if len(words) <= 2:
        return True

    # Check for known conversational follow-up phrases
    if any(p in q_lower for p in BACK_REFERENCE_PHRASES):
        return True

    # Check for pronoun and back-reference words
    intersection = set(words) & BACK_REFERENCE_WORDS
    if intersection:
        if len(words) <= 6 or any(w in ['it', 'that', 'they', 'them', 'its', 'this'] for w in words):
            return True

    return False


def resolve_search_query_heuristic(history: List[Dict[str, Any]], query: str) -> str:
    """
    Zero-latency heuristic query resolution.
    If the query is a follow-up, enriches it with the semantic anchor of the previous user query (0ms).
    """
    user_msgs = [m for m in history if m.get("role") == "user"]
    if len(user_msgs) <= 1 or not is_followup_query(query):
        return query

    # Find the most recent substantive user query
    last_user_query = ""
    for m in reversed(user_msgs[:-1]):
        content = m.get("content", "").strip()
        if content and len(content.split()) >= 3:
            last_user_query = content
            break

    if not last_user_query:
        return query

    return f"{query} {last_user_query}"


def rewrite_query_with_llm(
    query: str,
    history: List[Dict[str, Any]],
    llm: Any
) -> str:
    """
    Contextualizes ambiguous follow-up questions using the active LLM.
    Returns a single concise standalone search query for vector retrieval.
    """
    if not is_followup_query(query) or not llm:
        return query

    # Extract immediate past turns (max 4 turns)
    prior_turns = [m for m in history if m.get("role") in ("user", "assistant")][-4:]
    if not prior_turns:
        return query

    lc_history = []
    for m in prior_turns:
        if m.get("role") == "user":
            lc_history.append(HumanMessage(content=m.get("content", "")))
        elif m.get("role") == "assistant":
            # Truncate assistant messages to avoid token bloat
            short_content = m.get("content", "")[:300]
            lc_history.append(AIMessage(content=short_content))

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Given the conversational exchange and a follow-up query containing pronouns or references, "
            "reformulate it into one concise, clear standalone search query for document retrieval. "
            "Return ONLY the search query string with NO explanations, quotes, or conversational preamble."
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

    try:
        chain = prompt | llm
        res = chain.invoke({"chat_history": lc_history, "input": query})
        content = res.content if hasattr(res, "content") else str(res)
        rewritten = content.strip().strip('"').strip("'")
        if rewritten and len(rewritten) < 300:
            logger.info(f"Rewrote query '{query}' -> '{rewritten}'")
            return rewritten
    except Exception as e:
        logger.warning(f"LLM query rewriting failed ({e}); falling back to heuristic.")

    return resolve_search_query_heuristic(history, query)

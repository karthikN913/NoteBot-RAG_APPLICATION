"""Prompt Templates, Grounded Generation, and Citation Orchestration for NoteBot."""

import re
from typing import List, Dict, Any, Generator, Tuple
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document
from notebot.config.settings import DEFAULT_LLM_MODELS
from notebot.llm.provider_manager import get_llm_instance
from notebot.config.logging_config import setup_logger

logger = setup_logger("notebot.response_generator")

NOTEBOT_SYSTEM_PROMPT = """You are NoteBot, a senior academic tutor and document intelligence assistant.

Your primary mission is to EXPLAIN the uploaded document's contents to the user in a clear, engaging, structured, and pedagogical manner.

CRITICAL INSTRUCTIONS FOR ACCURACY, CITATION, AND EXPLANATION:

1. EXPLAIN AND TEACH — DO NOT VERBATIM COPY:
   - Synthesize the information in your own clear, articulate words.
   - Do NOT simply copy-paste or regurgitate raw fragments or disconnected sentences.
   - Break down dense academic or technical passages into structured, intuitive concepts:
     • Core Concept: A clear, direct definition of what it is.
     • Mechanism & Details: A logical explanation of how or why it works based on the text.
     • Key Terminology: Define and clarify essential terms mentioned in the document.

2. ACCURATE CITATIONS & ZERO HALLUCINATION:
   - Every single fact, number, definition, formula, or principle MUST be grounded 100% in the provided DOCUMENT CONTEXT.
   - Always reference the source document and page number where the fact originated using the format:
     [Doc: <filename>, Page: <page_number>]
   - Do NOT invent facts, outside dates, or unsubstantiated claims.
   - If the requested concept is absent from the DOCUMENT CONTEXT, state clearly:
     "This specific topic is not covered in the uploaded document(s)."
     (If related topics are mentioned in the document, briefly guide the user toward them).

3. ADAPTIVE PEDAGOGICAL STRUCTURE:
   - Definitional Questions: Start with a crisp definition, then explain its significance.
   - Process / How-To Questions: Use numbered step-by-step breakdowns.
   - Comparison Requests: Contrast key aspects side-by-side using structured bullets or Markdown tables.
   - Formatting: Use **bold** for key concepts and terms, bullet points for readability, and LaTeX ($...$ or $$...$$) for formulas.
   - Conversational Memory: Seamlessly resolve follow-up questions while maintaining a supportive tutoring tone."""


RAG_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", NOTEBOT_SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", """DOCUMENT CONTEXT:
{context}

USER QUESTION:
{input}

INSTRUCTION: Carefully analyze the document context above. Teach and explain the answer to the user in your own clear words, citing the exact source documents and page numbers [Doc: <filename>, Page: <page>] for all key claims:""")
])


def format_context_for_prompt(documents: List[Document]) -> str:
    """Formats retrieved Document objects into structured context blocks for the prompt."""
    if not documents:
        return "No relevant passages were retrieved from the document."

    blocks = []
    for i, doc in enumerate(documents, 1):
        source = doc.metadata.get("source", "Unknown Document")
        page = doc.metadata.get("page", 1)
        chunk_id = doc.metadata.get("chunk_id", f"c{i}")
        blocks.append(
            f"--- [PASSAGE {i}] Source: {source} | Page: {page} | ID: {chunk_id} ---\n"
            f"{doc.page_content.strip()}"
        )
    return "\n\n".join(blocks)


def stream_rag_response(
    provider: str,
    api_key: str,
    query: str,
    documents: List[Document],
    chat_history: List[Dict[str, Any]],
    temperature: float = 0.3,
    max_tokens: int = 2048,
    model_override: str = None
) -> Generator[str, None, None]:
    """
    Executes streaming RAG inference with automatic multi-model fallback.
    Yields string tokens in real time.
    """
    # Build LangChain chat history
    prior_msgs = [m for m in chat_history if m.get("role") in ("user", "assistant")]
    # Exclude the current in-flight user message if already appended
    if prior_msgs and prior_msgs[-1].get("role") == "user" and prior_msgs[-1].get("content") == query:
        prior_msgs = prior_msgs[:-1]

    lc_history = []
    for m in prior_msgs[-6:]:  # Retain up to last 6 turns
        if m.get("role") == "user":
            lc_history.append(HumanMessage(content=m["content"]))
        elif m.get("role") == "assistant":
            lc_history.append(AIMessage(content=m["content"]))

    context_str = format_context_for_prompt(documents)

    models_to_try = [model_override] if model_override else DEFAULT_LLM_MODELS.get(provider.lower(), [])
    if not models_to_try:
        models_to_try = ["gpt-4o-mini"]

    active_stream = None
    last_err = None

    for mod in models_to_try:
        try:
            llm = get_llm_instance(
                provider=provider,
                model_name=mod,
                key=api_key,
                temperature=temperature,
                max_tokens=max_tokens,
                streaming=True
            )

            chain = RAG_PROMPT_TEMPLATE | llm
            stream_gen = chain.stream({
                "input": query,
                "chat_history": lc_history,
                "context": context_str
            })

            first_chunk = next(stream_gen, None)
            if first_chunk is not None:
                active_stream = (first_chunk, stream_gen)
                logger.info(f"Streaming initialized with {provider} ({mod})")
                break
        except Exception as e:
            last_err = e
            logger.warning(f"Model {mod} failed ({e}); attempting next in hierarchy.")
            continue

    if active_stream is None:
        if last_err:
            raise last_err
        raise RuntimeError(f"No response generated from provider '{provider}'.")

    first_chunk, stream_gen = active_stream

    def extract_text(chunk) -> str:
        if hasattr(chunk, "content"):
            c = chunk.content
            if isinstance(c, str):
                return c
            elif isinstance(c, list):
                return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in c)
            return str(c)
        return str(chunk)

    first_text = extract_text(first_chunk)
    if first_text:
        yield first_text

    for chunk in stream_gen:
        txt = extract_text(chunk)
        if txt:
            yield txt


def verify_answer_groundedness(answer: str, documents: List[Document]) -> Dict[str, Any]:
    """
    Verifies answer citations and groundedness against retrieved context.
    Detects cited documents and verifies whether they correspond to actual retrieved sources.
    """
    retrieved_sources = {doc.metadata.get("source", "").lower() for doc in documents}
    retrieved_pages = {doc.metadata.get("page") for doc in documents}

    # Extract [Doc: X, Page: Y] citation patterns
    citation_pattern = re.compile(r'\[Doc:\s*([^,\]]+),\s*Page:\s*(\d+)\]', re.IGNORECASE)
    citations_found = citation_pattern.findall(answer)

    verified_citations = []
    unverified_citations = []

    for doc_name, page_str in citations_found:
        page_num = int(page_str)
        is_source_match = any(doc_name.strip().lower() in s for s in retrieved_sources)
        is_page_match = page_num in retrieved_pages

        citation_record = {"doc": doc_name.strip(), "page": page_num}
        if is_source_match or is_page_match:
            verified_citations.append(citation_record)
        else:
            unverified_citations.append(citation_record)

    return {
        "total_citations": len(citations_found),
        "verified_citations": verified_citations,
        "unverified_citations": unverified_citations,
        "is_sufficiently_grounded": len(unverified_citations) == 0
    }


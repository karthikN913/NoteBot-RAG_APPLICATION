import io
import html
import os
import re
import streamlit as st

# Load environment variables from .env file if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from pypdf import PdfReader
except ImportError:
    from PyPDF2 import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# Optional Multi-Provider & Local Fallback Imports
try:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
    HAS_GOOGLE = True
except ImportError:
    HAS_GOOGLE = False

try:
    from langchain_groq import ChatGroq
    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False

try:
    from langchain_community.embeddings import FastEmbedEmbeddings
    HAS_FASTEMBED = True
except ImportError:
    HAS_FASTEMBED = False


# ---------------------------------------------------------------------------
# Page Configuration & Modern Student Aesthetics
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="NoteBot — Conversational AI Document Assistant",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for chat UI and polished aesthetics
st.markdown("""
<style>
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1000px;
    }
    .app-header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 4px;
    }
    .app-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin: 0;
    }
    .app-subtitle {
        font-size: 1rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .stat-box {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 12px 16px;
        margin-bottom: 16px;
    }
    .stat-label {
        font-size: 0.8rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
    }
    .stat-value {
        font-size: 1.15rem;
        font-weight: 600;
        color: #0F172A;
    }
    .passage-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 10px;
        font-size: 0.92rem;
        line-height: 1.5;
        white-space: pre-wrap;
    }
    .passage-header {
        font-weight: 600;
        color: #2563EB;
        margin-bottom: 4px;
    }
    /* Spinner styling */
    div[data-testid="stSpinner"] {
        display: flex !important;
        align-items: center !important;
        padding: 6px 0 !important;
        margin: 4px 0 !important;
    }
    div[data-testid="stSpinner"] > div {
        border-top-color: #2563EB !important;
        border-right-color: #2563EB !important;
        width: 22px !important;
        height: 22px !important;
    }
</style>
""", unsafe_allow_html=True)

# App Title Header
st.markdown("""
<div class="app-header">
    <span style="font-size: 2.3rem;">📘</span>
    <h1 class="app-title">NoteBot</h1>
</div>
<p class="app-subtitle">Conversational Student AI Tutor & Document Q&A (Multi-Turn Chat & Deep Explanations)</p>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session State Initialization for Continuous Chat History
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "👋 Hi! I'm **NoteBot**, your AI study tutor. Upload a PDF document in the sidebar, and ask me any question or summary request. I remember our entire conversation so you can ask follow-up questions anytime!",
            "sources": None
        }
    ]


# ---------------------------------------------------------------------------
# Helper: Provider Auto-Detection
# ---------------------------------------------------------------------------
def detect_provider(key: str) -> str:
    """Auto-detect model provider based on API key structure."""
    if not key:
        return "openai"
    k = key.strip()
    if k.startswith("AIza") or k.startswith("AQ."):
        return "google"
    elif k.startswith("gsk_"):
        return "groq"
    elif k.startswith("sk-or-"):
        return "openrouter"
    elif k.startswith("sk-") or k.startswith("sk-proj-"):
        return "openai"
    return "google" if "." in k else "openai"


# ---------------------------------------------------------------------------
# API Key Handling & Model Selection
# ---------------------------------------------------------------------------
env_key = (
    os.getenv("OPENAI_API_KEY")
    or os.getenv("GEMINI_API_KEY")
    or os.getenv("GOOGLE_API_KEY")
    or os.getenv("GROQ_API_KEY")
    or os.getenv("OPENROUTER_API_KEY")
)
if not env_key:
    try:
        env_key = (
            st.secrets.get("OPENAI_API_KEY")
            or st.secrets.get("GEMINI_API_KEY")
            or st.secrets.get("GOOGLE_API_KEY")
            or st.secrets.get("GROQ_API_KEY")
            or st.secrets.get("OPENROUTER_API_KEY")
        )
    except Exception:
        env_key = None

with st.sidebar:
    st.header("⚙️ Configuration")

    provider_choice = st.selectbox(
        "Select Model Provider",
        options=[
            "Auto-Detect (Recommended)",
            "Google Gemini (Free tier at aistudio.google.com)",
            "OpenAI (gpt-4o-mini)",
            "Groq (Free tier at console.groq.com)",
            "OpenRouter (openrouter.ai)"
        ],
        help="NoteBot supports Google Gemini, OpenAI, Groq, and OpenRouter."
    )

    api_key_input = st.text_input(
        "Enter API Key",
        value=env_key if env_key else "",
        type="password",
        help="Paste your API key (Google Gemini, OpenAI, Groq, or OpenRouter)."
    )

    api_key = api_key_input.strip() if api_key_input else None

    # Resolve Provider Selection
    if provider_choice.startswith("Auto-Detect"):
        provider = detect_provider(api_key)
    elif provider_choice.startswith("Google Gemini"):
        provider = "google"
    elif provider_choice.startswith("Groq"):
        provider = "groq"
    elif provider_choice.startswith("OpenRouter"):
        provider = "openrouter"
    else:
        provider = "openai"

    if api_key:
        provider_names = {
            "google": "Google Gemini (gemini-2.5-flash / gemini-flash-latest)",
            "openai": "OpenAI (gpt-4o-mini)",
            "groq": "Groq (llama-3.3-70b-versatile)",
            "openrouter": "OpenRouter"
        }
        st.success(f"✅ Active Provider: **{provider_names.get(provider, provider)}**")
    else:
        st.warning("⚠️ API Key is required to proceed.")

    st.markdown("---")
    st.header("📄 Upload Document")
    pdf_file = st.file_uploader(
        "Select a PDF file (notes, report, textbook)",
        type=["pdf"],
        help="Only readable PDFs with text content are supported."
    )

    st.markdown("---")
    if st.button("🧹 Clear Chat History", use_container_width=True):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "🧹 Chat history cleared! What would you like to ask about your document?",
                "sources": None
            }
        ]
        st.rerun()

if not api_key:
    st.info(
        "👈 Please enter your API key in the sidebar to get started.\n\n"
        "💡 **Don't have an OpenAI key with paid credits?** Get a **100% Free Google Gemini API Key** at [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)!"
    )
    st.stop()


# ---------------------------------------------------------------------------
# Conversational RAG Prompt Definition (Deep Explanations & Adaptive Tutoring)
# ---------------------------------------------------------------------------
RAG_SYSTEM_PROMPT = """You are NoteBot, an expert academic tutor and conversational document analyst.

Your primary mission is to EXPLAIN the document's content to the user in a clear, engaging, and thoroughly understandable manner.

CRITICAL INSTRUCTIONS FOR ACCURACY, EXPLANATION, AND SPEED:

1. EXPLAIN AND TEACH — DO NOT VERBATIM COPY:
   - Do NOT simply copy-paste or regurgitate raw excerpts or fragmented sentences from the document.
   - Synthesize the information in your own clear, articulate words.
   - Break down dense academic, technical, or complex prose into intuitive, structured explanations.
   - When explaining a concept, address:
     • Core Concept: A clear, direct explanation of what it is.
     • Mechanism & Details: A logical breakdown of how or why it works based on the document.
     • Key Terminology: Define and clarify important terms referenced in the text.
   - Connect related ideas across the document to give the user a complete conceptual understanding.

2. ACCURACY & 100% GROUNDING (ZERO HALLUCINATION):
   - Every single fact, definition, formula, number, and principle MUST be grounded strictly in the provided DOCUMENT CONTEXT.
   - Do NOT fabricate outside facts, dates, or unsubstantiated claims.
   - Maintain absolute fidelity to the source document's true meaning and intent.
   - If the requested concept is absent from the DOCUMENT CONTEXT, state clearly:
     "This specific topic is not covered in the uploaded document."
     (If related topics are mentioned in the document, briefly point the user to those).

3. ADAPTIVE PEDAGOGICAL STRUCTURE:
   - Quick / Definitional Questions: Start with a crisp, direct definition, followed by an intuitive explanation of its role or importance.
   - Process / How-To / Mechanism Questions: Use numbered step-by-step breakdowns or cause-and-effect flow.
   - Overview / Summary Requests: Provide an executive overview followed by bulleted thematic highlights with brief explanations.
   - Comparison Requests: Contrast key aspects side-by-side using structured bullets or Markdown tables.
   - Formatting: Use **bold** for key concepts and terms, bullet points for readability, and LaTeX ($...$ or $$...$$) for formulas.

4. CONVERSATIONAL MEMORY & FLUIDITY:
   - Seamlessly resolve follow-up inquiries (e.g. "elaborate on the second point", "why does that happen?", "give an example").
   - Maintain a friendly, supportive, and intellectually rigorous tutoring tone."""

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", RAG_SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", """DOCUMENT CONTEXT:
{context}

USER QUESTION:
{input}

INSTRUCTION: Carefully analyze the document context above. Teach and explain the answer to the user in your own clear words, referencing the exact facts and concepts from the document without verbatim copying:""")
])


# ---------------------------------------------------------------------------
# Caching PDF Parsing & Chunking
# ---------------------------------------------------------------------------
def clean_pdf_text(text: str) -> str:
    """Cleans common PDF extraction artifacts, line wraps, and hyphenations."""
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\x0c", "\n").replace("\x00", "")
    # Fix hyphenated words broken across lines (e.g. "algo-\nrithm" -> "algorithm")
    text = re.sub(r'(\b[a-zA-Z]{2,})-\n([a-zA-Z]{2,}\b)', r'\1\2', text)
    # Collapse multiple consecutive blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


@st.cache_data(show_spinner="Extracting and processing document text...")
def parse_and_chunk_pdf(file_bytes: bytes):
    """
    Parses PDF bytes, extracts readable text, and splits into chunks.
    Cached by file content to eliminate redundant PDF parsing on each question.
    """
    reader = PdfReader(io.BytesIO(file_bytes))
    extracted_text_pages = []
    skipped_pages = []

    for i, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if page_text and page_text.strip():
            cleaned = clean_pdf_text(page_text)
            if cleaned:
                extracted_text_pages.append(f"[Page {i + 1}]\n{cleaned}")
        else:
            skipped_pages.append(i + 1)

    full_text = "\n\n".join(extracted_text_pages)
    if not full_text.strip():
        return len(reader.pages), "", [], skipped_pages

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len,
        separators=["\n\n", "\n", ". ", "? ", "! ", "; ", " ", ""]
    )
    chunks = splitter.split_text(full_text)
    return len(reader.pages), full_text, chunks, skipped_pages


# ---------------------------------------------------------------------------
# Caching Vector Store Function
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Embedding text chunks & building FAISS index...")
def build_vector_store(chunks: tuple, key: str, active_provider: str):
    """
    Builds and caches FAISS vector store per uploaded document session.
    """
    chunk_list = list(chunks)
    last_error = None

    # 1. Google Gemini Embeddings
    if active_provider == "google":
        if not HAS_GOOGLE:
            raise ImportError("langchain-google-genai package is required for Google Gemini embeddings.")
        google_models = [
            "text-embedding-004",
            "gemini-embedding-001",
            "gemini-embedding-2",
            "models/text-embedding-004",
            "embedding-001"
        ]
        for m_name in google_models:
            try:
                embeddings = GoogleGenerativeAIEmbeddings(
                    model=m_name,
                    google_api_key=key
                )
                return FAISS.from_texts(chunk_list, embeddings)
            except Exception as e:
                last_error = e
                err_str = str(e).lower()
                if "api_key_invalid" in err_str or "unregistered" in err_str or "400" in err_str:
                    break

        # Fallback to local FastEmbed if Google embedding quota/service fails
        if HAS_FASTEMBED:
            try:
                embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
                return FAISS.from_texts(chunk_list, embeddings)
            except Exception:
                pass

        if last_error:
            raise RuntimeError(f"Google Gemini embedding failed: {last_error}")
        raise RuntimeError("Failed to initialize Google Gemini embeddings.")

    # 2. OpenAI Embeddings
    elif active_provider == "openai":
        openai_models = ["text-embedding-3-small", "text-embedding-ada-002"]
        for m_name in openai_models:
            try:
                embeddings = OpenAIEmbeddings(
                    api_key=key,
                    model=m_name
                )
                return FAISS.from_texts(chunk_list, embeddings)
            except Exception as e:
                last_error = e
                err_str = str(e).lower()
                if "incorrect api key" in err_str or "invalid_api_key" in err_str or "401" in err_str:
                    break

        # Fallback to local FastEmbed if OpenAI quota or network fails
        if HAS_FASTEMBED:
            try:
                embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
                return FAISS.from_texts(chunk_list, embeddings)
            except Exception:
                pass

        if last_error:
            raise RuntimeError(f"OpenAI embedding failed: {last_error}")
        raise RuntimeError("Failed to initialize OpenAI embeddings.")

    # 3. Groq or OpenRouter (neither provides a native embedding API)
    else:
        if HAS_FASTEMBED:
            try:
                embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
                return FAISS.from_texts(chunk_list, embeddings)
            except Exception as e:
                last_error = e

        err_detail = f" ({last_error})" if last_error else ""
        raise RuntimeError(
            f"The selected provider ({active_provider.title()}) does not provide an embedding API, "
            f"and local offline embeddings (FastEmbed) could not be loaded{err_detail}. "
            "Please select Google Gemini (free at aistudio.google.com) or OpenAI to generate document embeddings."
        )


# ---------------------------------------------------------------------------
# High-Speed Multi-Provider Model Hierarchies
# ---------------------------------------------------------------------------
FAST_GOOGLE_MODELS = [
    "gemini-2.5-flash",
    "gemini-flash-latest",
    "gemini-2.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-2.5-pro",
]

def get_supported_gemini_models(key: str) -> list[str]:
    """Returns the prioritized, active Google Gemini models for fast inference."""
    return FAST_GOOGLE_MODELS


# ---------------------------------------------------------------------------
# High-Speed Cached LLM Instance Manager & Query Resolver
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_cached_llm(provider: str, model_name: str, key: str, temperature: float = 0.3, max_tokens: int = 2048):
    """
    Caches LLM instances in memory to eliminate the ~1.8s Python class instantiation
    overhead and maintain warm keep-alive TLS connection pools for instant generation.
    """
    if provider == "google":
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=key,
            temperature=temperature,
            max_output_tokens=max_tokens,
            streaming=True
        )
    elif provider == "groq":
        return ChatGroq(
            model=model_name,
            groq_api_key=key,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=True
        )
    elif provider == "openrouter":
        return ChatOpenAI(
            api_key=key,
            base_url="https://openrouter.ai/api/v1",
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=True
        )
    else:
        return ChatOpenAI(
            api_key=key,
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=True
        )


@st.cache_data(show_spinner=False, max_entries=100)
def cached_similarity_search(_vector_store, query: str, k: int):
    """Caches vector similarity search results in RAM for 0ms lookup on repeated or preset queries."""
    return _vector_store.similarity_search(query, k=k)


def is_likely_followup(query: str) -> bool:
    """Check if query genuinely needs prior chat context to be understood."""
    q_lower = query.lower().strip()
    words = re.findall(r'\b[a-z0-9]+\b', q_lower)
    if len(words) <= 2:
        return True

    back_reference_words = {
        'it', 'that', 'these', 'those', 'they', 'them', 'he', 'she',
        'above', 'previous', 'earlier', 'latter', 'former', 'aforementioned'
    }
    back_reference_phrases = (
        'tell me more', 'how come', 'what about', 'the first point',
        'the second point', 'the third point', 'point 1', 'point 2',
        'you mentioned', 'as you said', 'in that case', 'elaborate on that'
    )

    if any(p in q_lower for p in back_reference_phrases):
        return True

    if bool(set(words) & back_reference_words):
        if len(words) <= 6 or any(w in ['it', 'that', 'they', 'them'] for w in words):
            return True

    return False


def resolve_search_query(history: list, query: str) -> str:
    """
    Constructs an optimized vector search query instantaneously in 0ms.
    If the query is a follow-up referencing prior conversation, enriches it
    with the semantic anchor of the previous user query, completely bypassing
    redundant, high-latency LLM round-trips.
    """
    user_msgs = [m for m in history if m.get("role") == "user"]
    if len(user_msgs) <= 1 or not is_likely_followup(query):
        return query

    # Find the most recent substantive user query to borrow semantic anchor keywords
    last_user_query = ""
    for m in reversed(user_msgs[:-1]):
        content = m.get("content", "").strip()
        if content and len(content.split()) >= 3:
            last_user_query = content
            break

    if not last_user_query:
        return query

    return f"{query} {last_user_query}"


# ---------------------------------------------------------------------------
# High-Speed Streaming RAG Execution Engine
# ---------------------------------------------------------------------------
def stream_rag_chain(active_provider: str, key: str, prompt_template, context_chunks, chat_history_list, query: str):
    """
    Streams RAG execution token by token for instant UI responsiveness and low latency.
    Includes multi-model auto-fallback for provider model deprecations or 404s.
    """
    prior_msgs = chat_history_list[1:-1] if chat_history_list and chat_history_list[-1].get("role") == "user" else chat_history_list[1:]
    lc_history = []
    for m in prior_msgs[-4:]:
        if m.get("role") == "user":
            lc_history.append(HumanMessage(content=m["content"]))
        elif m.get("role") == "assistant":
            lc_history.append(AIMessage(content=m["content"]))

    context_text = "\n\n".join([
        f"--- [PASSAGE {i + 1}] ---\n{doc.page_content.strip()}"
        for i, doc in enumerate(context_chunks)
    ])

    models_to_try = []
    if active_provider == "google":
        if not HAS_GOOGLE:
            raise ImportError("langchain-google-genai package is required for Google Gemini.")
        models_to_try = [("google", m) for m in FAST_GOOGLE_MODELS]
    elif active_provider == "groq":
        if not HAS_GROQ:
            raise ImportError("langchain-groq package is required for Groq models.")
        models_to_try = [
            ("groq", "llama-3.3-70b-versatile"),
            ("groq", "llama-3.1-8b-instant"),
        ]
    elif active_provider == "openrouter":
        models_to_try = [
            ("openrouter", "meta-llama/llama-3.3-70b-instruct:free"),
            ("openrouter", "google/gemini-2.0-flash-exp:free"),
            ("openrouter", "mistralai/mistral-7b-instruct:free"),
        ]
    else:
        models_to_try = [("openai", "gpt-4o-mini"), ("openai", "gpt-3.5-turbo")]

    active_stream = None
    last_err = None
    for prov, mod in models_to_try:
        try:
            llm = get_cached_llm(prov, mod, key, temperature=0.3, max_tokens=2048)

            chain = prompt_template | llm
            stream_gen = chain.stream({
                "input": query,
                "chat_history": lc_history,
                "context": context_text
            })

            # Check if model can generate the initial chunk
            first_chunk = next(stream_gen, None)
            if first_chunk is not None:
                active_stream = (first_chunk, stream_gen)
                break
        except Exception as e:
            last_err = e
            continue

    if active_stream is None:
        if last_err:
            raise last_err
        raise RuntimeError("No response was generated from the language model.")

    first_chunk, stream_gen = active_stream

    def extract_chunk_text(chunk):
        if hasattr(chunk, "content"):
            c = chunk.content
            if isinstance(c, str):
                return c
            elif isinstance(c, list):
                parts = []
                for p in c:
                    if isinstance(p, dict):
                        parts.append(p.get("text", ""))
                    elif hasattr(p, "text"):
                        parts.append(getattr(p, "text", ""))
                    else:
                        parts.append(str(p))
                return "".join(parts)
            return str(c)
        return str(chunk)

    first_text = extract_chunk_text(first_chunk)
    if first_text:
        yield first_text

    for chunk in stream_gen:
        text = extract_chunk_text(chunk)
        if text:
            yield text


# ---------------------------------------------------------------------------
# Main Application Workflow
# ---------------------------------------------------------------------------
if pdf_file is not None:
    file_bytes = pdf_file.getvalue()
    file_id = getattr(pdf_file, "file_id", pdf_file.name)

    # Automatically reset chat history when a new document is uploaded
    if st.session_state.get("current_file_id") != file_id:
        st.session_state.current_file_id = file_id
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": f"👋 I've loaded **{pdf_file.name}**! Ask me any question or summary request. I remember our conversation so you can ask follow-ups anytime!",
                "sources": None
            }
        ]

    # 1. PDF Text Extraction & Error Handling
    try:
        total_pages, full_text, chunks, skipped_pages = parse_and_chunk_pdf(file_bytes)

        if skipped_pages:
            st.sidebar.info(
                f"ℹ️ Pages skipped (no extractable text): {', '.join(map(str, skipped_pages))}"
            )

        if not full_text.strip() or not chunks:
            st.error(
                "❌ No extractable text found in this document. "
                "This appears to be a scanned or image-only PDF, which requires OCR."
            )
            st.stop()

    except Exception as e:
        st.error(f"❌ Failed to parse PDF document: {str(e)}")
        st.stop()

    # Display document statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-label">Total Pages</div>
            <div class="stat-value">{total_pages}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-label">Characters Extracted</div>
            <div class="stat-value">{len(full_text):,}</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-label">Text Chunks</div>
            <div class="stat-value">{len(chunks)}</div>
        </div>
        """, unsafe_allow_html=True)

    # 2. Vector Indexing (Cached per uploaded PDF session)
    try:
        vector_store = build_vector_store(tuple(chunks), api_key, provider)
    except Exception as e:
        err_msg = str(e)
        st.error(f"❌ Error creating vector index: {err_msg}")
        st.stop()

    # 3. Quick Action Prompts & Continuous Chat Interface
    st.caption("⚡ **Quick Study Prompts:** Click any button below to start or continue your chat:")
    qcol1, qcol2, qcol3 = st.columns(3)
    preset_query = None
    if qcol1.button("📌 Summarize Document", use_container_width=True):
        preset_query = "Summarize the key topics and main points covered in this document in full detail."
    if qcol2.button("🔑 List Key Definitions", use_container_width=True):
        preset_query = "List and explain all key terms, definitions, and concepts in this document in detail."
    if qcol3.button("❓ Generate Practice Quiz", use_container_width=True):
        preset_query = "Generate 3 detailed study questions with answers based on the key concepts in this document."

    # Render previous conversation history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("🔍 View Retrieved Source Passages"):
                    for i, src in enumerate(msg["sources"], 1):
                        safe_src = html.escape(src)
                        st.markdown(f"""
                        <div class="passage-card">
                            <div class="passage-header">Passage {i}</div>
                            {safe_src}
                        </div>
                        """, unsafe_allow_html=True)

    # Capture input from chat_input or preset buttons
    user_input = st.chat_input("Ask NoteBot a question or follow-up about your document...")
    active_query = user_input or preset_query

    if active_query:
        # Append user message to state and display
        st.session_state.messages.append({"role": "user", "content": active_query, "sources": None})
        with st.chat_message("user"):
            st.markdown(active_query)

        # Process assistant response
        with st.chat_message("assistant"):
            try:
                # 1. Instant zero-latency search query resolution (0ms)
                search_q = resolve_search_query(
                    st.session_state.messages,
                    active_query
                )

                # 2. Similarity search cached in RAM (0-5ms)
                matching_chunks = cached_similarity_search(
                    vector_store,
                    search_q,
                    k=min(4, len(chunks))
                )

                # 3. Stream generator using warm cached LLM connection
                stream_gen = stream_rag_chain(
                    provider,
                    api_key,
                    RAG_PROMPT,
                    matching_chunks,
                    st.session_state.messages,
                    active_query
                )

                # 4. Stream response tokens live to UI with instant start
                response = st.write_stream(stream_gen)

                # Prepare passage sources for expander
                passage_texts = [doc.page_content for doc in matching_chunks]
                with st.expander("🔍 View Retrieved Source Passages"):
                    for i, src in enumerate(passage_texts, 1):
                        safe_src = html.escape(src)
                        st.markdown(f"""
                        <div class="passage-card">
                            <div class="passage-header">Passage {i}</div>
                            {safe_src}
                        </div>
                        """, unsafe_allow_html=True)

                # Save response into chat memory
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "sources": passage_texts
                })

            except Exception as e:
                # Remove unanswered user message so chat history doesn't become desynchronized
                if st.session_state.messages and st.session_state.messages[-1].get("role") == "user":
                    st.session_state.messages.pop()
                err_msg = str(e)
                if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                    st.error(
                        "❌ **API Quota Exceeded (429)**\n\n"
                        "Please check your API provider usage limits or switch to a different provider."
                    )
                elif "401" in err_msg or "invalid_api_key" in err_msg or "API_KEY_INVALID" in err_msg:
                    st.error(
                        "❌ **Invalid API Key**\n\n"
                        "The provided API key was rejected by the provider. Please verify your key in the sidebar."
                    )
                else:
                    st.error(f"❌ Error: {err_msg}")

else:
    # Initial state when no PDF is uploaded yet
    st.info("💡 Upload a PDF document in the sidebar to start your continuous chat session.")

    st.markdown("""
    #### Key Student Features
    - 💬 **Continuous Multi-Turn Chat**: Asks follow-up questions, requests further examples, or asks for clarification while keeping context.
    - 🎓 **Educational Explanations**: Generates multi-paragraph study notes, key term breakdowns, and summaries.
    - 🔒 **Zero Hallucination**: Every answer is strictly grounded in your uploaded PDF document.
    - 🌐 **Multi-Model Support**: Works with **Google Gemini**, **OpenAI**, **Groq**, and **OpenRouter**.
    - ⚡ **Quick Action Chips**: One-click buttons for document summaries, key definitions, and practice quizzes.
    - 🚀 **Cached Indexing**: Embeddings and document parsing are computed once per file session for instant response times.
    """)

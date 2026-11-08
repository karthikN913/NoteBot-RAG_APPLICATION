"""NoteBot — Conversational AI Document Assistant.
Production RAG Application with Multi-PDF Support, Hybrid Search, MMR, Reranking & Persistence.
"""

import io
import html
import os
import streamlit as st

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from notebot.config.settings import (
    AppConfig,
    PROVIDER_GOOGLE,
    PROVIDER_OPENAI,
    PROVIDER_GROQ,
    PROVIDER_OPENROUTER,
    EMBED_GOOGLE,
    EMBED_OPENAI,
    EMBED_FASTEMBED,
    PROVIDER_DISPLAY_NAMES,
    get_provider_api_key
)
from notebot.config.logging_config import setup_logger
from notebot.utils.validators import validate_uploaded_file, sanitize_user_query
from notebot.ingestion.pdf_loader import ingest_documents
from notebot.chunking.chunker import chunk_documents
from notebot.embeddings.embedding_manager import get_embedding_model
from notebot.vectorstore.faiss_manager import FAISSIndexManager
from notebot.vectorstore.persistence import (
    compute_dataset_hash,
    save_faiss_index,
    load_faiss_index,
    list_saved_indexes,
    delete_faiss_index
)
from notebot.retrieval.hybrid_search import BM25Index
from notebot.retrieval.reranker import Reranker
from notebot.retrieval.query_rewriter import resolve_search_query_heuristic
from notebot.retrieval.retriever import NoteBotRetriever
from notebot.llm.provider_manager import (
    validate_provider_key,
    classify_api_error
)
from notebot.llm.response_generator import (
    stream_rag_response,
    verify_answer_groundedness
)
from notebot.conversation.chat_history import ChatHistoryManager, DEFAULT_WELCOME_MESSAGE

logger = setup_logger("notebot.app")

# ---------------------------------------------------------------------------
# Streamlit Page Setup & Custom Styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="NoteBot — Advanced Conversational RAG",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1050px;
    }
    .app-header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 2px;
    }
    .app-title {
        font-size: 2.3rem;
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
        margin-bottom: 12px;
    }
    .stat-label {
        font-size: 0.78rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
    }
    .stat-value {
        font-size: 1.2rem;
        font-weight: 700;
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
        display: flex;
        justify-content: space-between;
        font-weight: 600;
        color: #2563EB;
        margin-bottom: 6px;
        font-size: 0.85rem;
    }
    .source-badge {
        background: #EFF6FF;
        color: #1D4ED8;
        border: 1px solid #BFDBFE;
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    div[data-testid="stSpinner"] {
        display: flex !important;
        align-items: center !important;
        padding: 6px 0 !important;
    }
</style>
""", unsafe_allow_html=True)

# Application Header
st.markdown("""
<div class="app-header">
    <span style="font-size: 2.4rem;">📘</span>
    <h1 class="app-title">NoteBot</h1>
</div>
<p class="app-subtitle">Advanced Conversational RAG & Academic Intelligence (Multi-Doc Ingestion, Hybrid Search, MMR & Persistent FAISS)</p>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------
if "chat_manager" not in st.session_state:
    st.session_state.chat_manager = ChatHistoryManager()

if "indexed_dataset_hash" not in st.session_state:
    st.session_state.indexed_dataset_hash = None

if "retriever_instance" not in st.session_state:
    st.session_state.retriever_instance = None

if "active_documents" not in st.session_state:
    st.session_state.active_documents = []


# ---------------------------------------------------------------------------
# Sidebar: Provider, Embeddings & RAG Configuration
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Model Configuration")

    # 1. LLM Provider Selection
    llm_choice = st.selectbox(
        "LLM Provider",
        options=[
            "Google Gemini (Free tier available)",
            "OpenAI (gpt-4o-mini)",
            "Groq (High-speed LLaMA-3.3)",
            "OpenRouter (Open-source models)"
        ],
        index=0,
        help="Select language model provider for conversational inference."
    )

    provider_map = {
        "Google Gemini": PROVIDER_GOOGLE,
        "OpenAI": PROVIDER_OPENAI,
        "Groq": PROVIDER_GROQ,
        "OpenRouter": PROVIDER_OPENROUTER
    }
    selected_llm_provider = next(v for k, v in provider_map.items() if llm_choice.startswith(k))

    # Strict key retrieval: only designated key for this provider
    default_env_key = get_provider_api_key(selected_llm_provider) or ""
    llm_api_key_input = st.text_input(
        f"Enter {selected_llm_provider.title()} API Key",
        value=default_env_key,
        type="password",
        help=f"Paste your API key designated strictly for {selected_llm_provider.title()}."
    )
    llm_api_key = llm_api_key_input.strip() if llm_api_key_input else None

    # Key validation
    if llm_api_key:
        is_valid, err_reason = validate_provider_key(selected_llm_provider, llm_api_key)
        if not is_valid:
            st.warning(f"⚠️ {err_reason}")
        else:
            st.success(f"✅ {selected_llm_provider.title()} API Key Ready")
    else:
        st.warning(f"⚠️ API Key required for {selected_llm_provider.title()}.")

    st.markdown("---")
    st.header("🧠 Embedding Configuration")

    # 2. Independent Embedding Provider Selection
    embed_choice = st.selectbox(
        "Embedding Provider",
        options=[
            "Google Gemini (text-embedding-004)",
            "OpenAI (text-embedding-3-small)",
            "FastEmbed (Local Offline - Zero Cost)"
        ],
        index=0 if selected_llm_provider == PROVIDER_GOOGLE else (2 if selected_llm_provider in [PROVIDER_GROQ, PROVIDER_OPENROUTER] else 1),
        help="Embeddings can be selected completely independently of your LLM provider."
    )

    embed_provider_map = {
        "Google Gemini": EMBED_GOOGLE,
        "OpenAI": EMBED_OPENAI,
        "FastEmbed": EMBED_FASTEMBED
    }
    selected_embed_provider = next(v for k, v in embed_provider_map.items() if embed_choice.startswith(k))

    embed_api_key = None
    if selected_embed_provider != EMBED_FASTEMBED:
        embed_default_key = get_provider_api_key(selected_embed_provider) or (llm_api_key if selected_llm_provider == selected_embed_provider else "")
        embed_key_input = st.text_input(
            f"{selected_embed_provider.title()} Embedding Key",
            value=embed_default_key,
            type="password",
            help="API Key used for generating document embeddings."
        )
        embed_api_key = embed_key_input.strip() if embed_key_input else None

    st.markdown("---")
    st.header("🔍 Retrieval & Ingestion Settings")

    retrieval_mode = st.selectbox(
        "Retrieval Strategy",
        options=[
            "Similarity Search (Fastest)",
            "MMR - Diverse Results (Reduces Redundancy)",
            "Hybrid Search (BM25 Keyword + FAISS Dense)"
        ],
        index=0,
        help="Select how NoteBot searches and retrieves document passages."
    )
    mode_key = "similarity"
    if "MMR" in retrieval_mode:
        mode_key = "mmr"
    elif "Hybrid" in retrieval_mode:
        mode_key = "hybrid"

    top_k = st.slider("Top-K Retrieved Passages", min_value=2, max_value=8, value=4, step=1)
    enable_reranker = st.checkbox("Enable FlashRank Cross-Encoder Reranker", value=False, help="Reranks candidate chunks with a local neural model for higher precision.")
    enable_ocr = st.checkbox("Enable OCR Fallback for Scanned Pages", value=False, help="Attempts OCR extraction if a PDF page contains no extractable text.")

    with st.expander("🛠️ Advanced Chunking Parameters"):
        chunk_size = st.slider("Chunk Size (characters)", min_value=400, max_value=2000, value=1000, step=100)
        chunk_overlap = st.slider("Chunk Overlap (characters)", min_value=0, max_value=400, value=150, step=25)

    st.markdown("---")
    st.header("📁 Document Management")
    uploaded_files = st.file_uploader(
        "Upload PDF Documents",
        type=["pdf"],
        accept_multiple_files=True,
        help="Select one or multiple PDF documents (notes, textbooks, reports)."
    )

    if st.button("🧹 Clear Chat History", use_container_width=True):
        st.session_state.chat_manager.clear()
        st.rerun()


# ---------------------------------------------------------------------------
# Check Missing API Keys
# ---------------------------------------------------------------------------
if not llm_api_key:
    st.info(
        "👈 Please provide your API key in the sidebar to get started.\n\n"
        "💡 **Need a free key?** You can get a free Google Gemini key at [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)."
    )
    st.stop()


# ---------------------------------------------------------------------------
# Ingestion, Chunking & Persistent Vector Store Pipeline
# ---------------------------------------------------------------------------
if uploaded_files:
    # 1. File Validation
    valid_file_payloads = []
    for f in uploaded_files:
        f_bytes = f.getvalue()
        is_valid, val_err = validate_uploaded_file(f.name, f_bytes)
        if not is_valid:
            st.error(f"❌ {val_err}")
            st.stop()
        valid_file_payloads.append((f.name, f_bytes))

    # Compute dataset hash for persistence
    files_info = [{"name": name, "size": len(b)} for name, b in valid_file_payloads]
    embed_signature = f"{selected_embed_provider}_{chunk_size}_{chunk_overlap}"
    dataset_hash = compute_dataset_hash(files_info, embed_signature)

    storage_dir = os.path.join(os.path.dirname(__file__), "data", "indexes")

    # Check if index needs rebuilding or can be loaded from persistence/memory
    if st.session_state.indexed_dataset_hash != dataset_hash:
        with st.spinner("Processing documents & building vector store..."):
            try:
                # Initialize Embedding Model
                embedding_model = get_embedding_model(
                    provider=selected_embed_provider,
                    key=embed_api_key,
                    allow_offline_fallback=True
                )

                # Attempt to load persisted index from disk
                loaded_faiss = load_faiss_index(storage_dir, dataset_hash, embedding_model)

                if loaded_faiss is not None:
                    manager = FAISSIndexManager(vector_store=loaded_faiss, embeddings=embedding_model)
                    all_docs = manager.get_all_documents()
                    st.toast(f"⚡ Loaded persistent index from disk ({len(all_docs)} chunks)", icon="💾")
                else:
                    # Ingest and extract text
                    ingest_res = ingest_documents(
                        valid_file_payloads,
                        enable_ocr=enable_ocr
                    )

                    if ingest_res.errors:
                        for err in ingest_res.errors:
                            st.warning(f"⚠️ {err}")

                    if not ingest_res.documents:
                        st.error("❌ No readable text could be extracted from the uploaded document(s). If this is a scanned document, enable OCR in the sidebar.")
                        st.stop()

                    # Chunk documents preserving metadata
                    all_docs = chunk_documents(
                        ingest_res.documents,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        deduplicate=True
                    )

                    if not all_docs:
                        st.error("❌ No valid text chunks were created.")
                        st.stop()

                    # Build FAISS vector store
                    manager = FAISSIndexManager.from_documents(all_docs, embedding_model)

                    # Persist index to disk
                    save_metadata = {
                        "dataset_hash": dataset_hash,
                        "files": [f.name for f in uploaded_files],
                        "total_chunks": len(all_docs),
                        "chunk_size": chunk_size,
                        "chunk_overlap": chunk_overlap,
                        "embedding_provider": selected_embed_provider
                    }
                    save_faiss_index(manager.vector_store, save_metadata, storage_dir, dataset_hash)

                # Build BM25 index for Hybrid Search
                bm25_idx = BM25Index(all_docs)

                # Initialize unified NoteBotRetriever
                reranker_instance = Reranker() if enable_reranker else None
                st.session_state.retriever_instance = NoteBotRetriever(
                    index_manager=manager,
                    bm25_index=bm25_idx,
                    reranker=reranker_instance
                )

                st.session_state.active_documents = all_docs
                st.session_state.indexed_dataset_hash = dataset_hash

            except Exception as e:
                st.error(f"❌ Error during document ingestion or index creation: {classify_api_error(e)}")
                st.stop()

    # Document Metrics Display
    docs_count = len(uploaded_files)
    total_chunks = len(st.session_state.active_documents)
    unique_sources = list({d.metadata.get("source", "Doc") for d in st.session_state.active_documents})

    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-label">Active Documents</div>
            <div class="stat-value">{docs_count} file(s)</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-label">Indexed Chunks</div>
            <div class="stat-value">{total_chunks:,}</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-label">Retrieval Mode</div>
            <div class="stat-value">{mode_key.upper()}</div>
        </div>
        """, unsafe_allow_html=True)

    # Quick Study Action Buttons
    st.caption("⚡ **Quick Study Prompts:** Click any button below to ask NoteBot instantly:")
    qcol1, qcol2, qcol3 = st.columns(3)
    preset_query = None
    if qcol1.button("📌 Comprehensive Summary", use_container_width=True):
        preset_query = "Provide a comprehensive pedagogical summary of the key concepts and topics in these documents."
    if qcol2.button("🔑 Key Definitions & Terms", use_container_width=True):
        preset_query = "List and thoroughly define all core technical terms and concepts found in the document(s)."
    if qcol3.button("❓ Practice Study Quiz", use_container_width=True):
        preset_query = "Generate 3 detailed exam-style study questions with complete answers based strictly on the text."

    # Render Chat History
    for msg in st.session_state.chat_manager.get_messages():
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("🔍 View Retrieved Source Citations & Passages"):
                    for i, src in enumerate(msg["sources"], 1):
                        source_file = html.escape(src.get("source", "Unknown"))
                        page_num = src.get("page", 1)
                        chunk_id = html.escape(src.get("chunk_id", f"c{i}"))
                        safe_text = html.escape(src.get("content", ""))
                        st.markdown(f"""
                        <div class="passage-card">
                            <div class="passage-header">
                                <span>Passage {i} &bull; Page {page_num}</span>
                                <span class="source-badge">{source_file}</span>
                            </div>
                            {safe_text}
                        </div>
                        """, unsafe_allow_html=True)

    # Chat Input Box
    user_query = st.chat_input("Ask NoteBot a question or follow-up about your documents...")
    active_query = user_query or preset_query

    if active_query:
        clean_query = sanitize_user_query(active_query)

        # Append user message
        st.session_state.chat_manager.add_user_message(clean_query)
        with st.chat_message("user"):
            st.markdown(clean_query)

        # Generate Assistant Response
        with st.chat_message("assistant"):
            try:
                # 1. Resolve search query (zero-latency heuristic anchor)
                retrieval_query = resolve_search_query_heuristic(
                    st.session_state.chat_manager.get_messages(),
                    clean_query
                )

                # 2. Retrieve relevant documents using chosen strategy
                retriever: NoteBotRetriever = st.session_state.retriever_instance
                retrieved_chunks = retriever.retrieve(
                    query=retrieval_query,
                    top_k=top_k,
                    mode=mode_key,
                    enable_reranker=enable_reranker
                )

                # 3. Stream Response
                stream_gen = stream_rag_response(
                    provider=selected_llm_provider,
                    api_key=llm_api_key,
                    query=clean_query,
                    documents=retrieved_chunks,
                    chat_history=st.session_state.chat_manager.get_messages()
                )

                full_response = st.write_stream(stream_gen)

                # 4. Source Attribution & Grounding Verification
                source_records = [
                    {
                        "source": doc.metadata.get("source", "Document"),
                        "page": doc.metadata.get("page", 1),
                        "chunk_id": doc.metadata.get("chunk_id", f"c{i}"),
                        "content": doc.page_content
                    }
                    for i, doc in enumerate(retrieved_chunks)
                ]

                with st.expander("🔍 View Retrieved Source Citations & Passages"):
                    for i, src in enumerate(source_records, 1):
                        source_file = html.escape(src["source"])
                        page_num = src["page"]
                        safe_text = html.escape(src["content"])
                        st.markdown(f"""
                        <div class="passage-card">
                            <div class="passage-header">
                                <span>Passage {i} &bull; Page {page_num}</span>
                                <span class="source-badge">{source_file}</span>
                            </div>
                            {safe_text}
                        </div>
                        """, unsafe_allow_html=True)

                # Grounding check
                grounding_check = verify_answer_groundedness(full_response, retrieved_chunks)
                if grounding_check["verified_citations"]:
                    verified_count = len(grounding_check["verified_citations"])
                    st.caption(f"🛡️ **Grounded Answer**: Verified {verified_count} document citation(s).")

                # Store response in chat memory
                st.session_state.chat_manager.add_assistant_message(full_response, sources=source_records)

            except Exception as e:
                st.session_state.chat_manager.pop_last_user_message()
                st.error(classify_api_error(e))

else:
    # Landing instructions
    st.info("💡 Upload one or more PDF documents in the sidebar to start your continuous study session.")

    st.markdown("""
    ### 🌟 What's New in NoteBot 2.0?
    - 📚 **Multi-Document Ingestion**: Upload multiple textbooks, research papers, or study notes simultaneously.
    - 🧠 **Decoupled Embeddings**: Choose between Google Gemini embeddings, OpenAI embeddings, or **100% Free Offline FastEmbed**.
    - 🔍 **Advanced Retrieval Strategies**:
      - **Standard Vector Similarity**: Ultra-fast cosine/L2 semantic search.
      - **MMR (Maximum Marginal Relevance)**: Promotes informational diversity and eliminates repetitive chunks.
      - **Hybrid Search (BM25 + FAISS)**: Fuses keyword precision with dense neural semantic understanding via Reciprocal Rank Fusion (RRF).
      - **Cross-Encoder Reranker**: Optional neural reranking powered by FlashRank.
    - 💾 **Persistent FAISS Storage**: Automatically saves vector stores to disk with content hashes for 0ms reloading on future sessions.
    - 🛡️ **Zero Hallucination & Grounded Citations**: Automatic source citation validation referencing exact document names and page numbers.
    """)

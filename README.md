# NoteBot — Conversational AI Document Assistant

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B.svg)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-1.0+-green.svg)](https://www.langchain.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

NoteBot is a production-grade Streamlit web application designed for high-precision, zero-hallucination document question answering, summarization, and interactive tutoring. Users can upload a PDF (study notes, textbook chapters, research papers) and chat with it in natural language across multiple turns.

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/karthikN913/NoteBot---RAG-APPLICATION.git
cd NoteBot---RAG-APPLICATION
```

### 2. Install Dependencies
Ensure you have Python 3.10+ installed:
```bash
pip install -r requirements.txt
```

### 3. Configure Your API Key
Duplicate the `.env.example` template:
```bash
cp .env.example .env
```
Open `.env` and paste your preferred API key. NoteBot supports **Google Gemini**, **OpenAI**, **Groq**, and **OpenRouter**:

```bash
# Recommended: Google Gemini (Free tier available at https://aistudio.google.com/app/apikey)
GEMINI_API_KEY="your-actual-api-key"

# Or OpenAI
OPENAI_API_KEY="your-actual-api-key"

# Or Groq (Free tier available at https://console.groq.com)
GROQ_API_KEY="your-actual-api-key"
```

*Alternatively, you can skip creating a `.env` file and enter your API key directly into the secure sidebar input field when the app starts.*

### 4. Run the Application
Launch NoteBot using Streamlit:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 💡 RAG Pipeline Architecture & Key Features

NoteBot implements an end-to-end conversational Retrieval-Augmented Generation (RAG) architecture engineered for precision, zero hallucination, and student tutoring:

- **Cached Document Parsing**: Page-by-page extraction using `pypdf`/`PyPDF2`, cached by file bytes with `@st.cache_data`. Unparsable or scanned pages are cleanly reported without crashing.
- **Text Normalization**: Automatic cleaning of soft-hyphen line breaks, carriage returns, and extraction artifacts with `[Page X]` citation tagging.
- **Recursive Sentence Chunking**: `RecursiveCharacterTextSplitter` configured with a chunk size of **1,000 characters** and a **150-character overlap** breaking cleanly on sentence boundaries (`. `, `? `, `! `, `; `).
- **Vector Indexing & Fast Retrieval**: In-memory `FAISS` vector store cached with `@st.cache_resource`. Supports Google Gemini (`text-embedding-004`, `gemini-embedding-001`), OpenAI (`text-embedding-3-small`), and local offline fallback via `FastEmbed`.
- **Intelligent Query Contextualization**: Follow-up questions referencing earlier context (e.g. *"Can you explain that in more detail?"*) are reformulated into standalone retrieval queries using conversational history.
- **Pedagogical Explanation Engine**: Formatted to teach and explain concepts thoroughly rather than quoting raw text from the document.
- **Sub-Second Token Streaming**: Real-time streaming generation via `st.write_stream` with multi-model auto-fallback.
- **Grounded Source Verification**: Expandable source viewer displays the exact source passages retrieved for full transparency.

---

## 🔒 Security & Privacy

- **No Hardcoded Secrets**: NoteBot never hardcodes or logs API keys. All keys are loaded from system environment variables, `.env`, or secure runtime session input.
- **Git Protection**: The `.gitignore` is strictly configured to prevent accidental commits of `.env` files, `.streamlit/secrets.toml`, local caches, virtual environments, and private uploaded `*.pdf` documents.
- **Sandboxed Execution**: Documents are processed in-memory and held in session state only for the duration of the active browser session.

---

## 🛠️ Technology Stack
- **Frontend / UI**: Streamlit
- **PDF Extraction**: PyPDF2 / pypdf
- **Orchestration**: LangChain (`langchain-core`, `langchain-openai`, `langchain-google-genai`, `langchain-groq`, `langchain-community`, `langchain-text-splitters`)
- **Embeddings**: Google Gemini `text-embedding-004`, `gemini-embedding-001`, OpenAI `text-embedding-3-small`, or FastEmbed
- **LLMs**: Google Gemini (`gemini-2.5-flash`, `gemini-flash-latest`, `gemini-2.5-flash-lite`), OpenAI (`gpt-4o-mini`), Groq (`llama-3.3-70b-versatile`), OpenRouter
- **Vector Index**: FAISS (in-memory, cached per session)

# NoteBot — Conversational AI Document Assistant

NoteBot is a production-grade Streamlit web application designed for high-precision, zero-hallucination document question answering, summarization, and interactive tutoring. Users can upload a PDF (study notes, textbook chapters, research papers) and chat with it in natural language across multiple turns.

---

## 🚀 Quick Start

### 1. Prerequisites
Ensure you have Python 3.10+ installed.

### 2. Installation
Clone or navigate to the repository directory and install dependencies:
```bash
pip install -r requirements.txt
```

### 3. Set API Key (Optional)
You can configure your API key in a `.env` file or as an environment variable. NoteBot supports **Google Gemini**, **OpenAI**, **Groq**, and **OpenRouter**:

```bash
# In a .env file (or terminal):
# For Google Gemini (Free tier available at https://aistudio.google.com/app/apikey)
GEMINI_API_KEY="your-actual-api-key"

# For OpenAI
OPENAI_API_KEY="your-actual-api-key"

# For Groq (Free tier available at https://console.groq.com)
GROQ_API_KEY="your-actual-api-key"
```

*Alternatively, you can enter your API key directly into the secure sidebar input field when the app starts.*

### 4. Run the Application
Launch NoteBot using Streamlit:
```bash
streamlit run app.py
```

---

## 💡 RAG Pipeline Architecture & Key Features

NoteBot implements an end-to-end conversational Retrieval-Augmented Generation (RAG) architecture engineered for precision, zero hallucination, and student tutoring:

- **Cached Document Parsing**: Page-by-page extraction using `pypdf`/`PyPDF2`, cached by file bytes with `@st.cache_data`. Unparsable or scanned pages are cleanly reported without crashing.
- **Recursive Chunking**: `RecursiveCharacterTextSplitter` configured with a chunk size of **1,000 characters** and a **150-character overlap**, preserving paragraph boundaries and sentence semantics.
- **Vector Indexing & Fast Retrieval**: In-memory `FAISS` vector store cached with `@st.cache_resource`. Supports Google Gemini (`text-embedding-004`), OpenAI (`text-embedding-3-small`), and local offline fallback via `FastEmbed`.
- **Intelligent Query Contextualization**: Follow-up questions referencing earlier context (e.g. *"Can you explain that in more detail?"*) are reformulated into standalone retrieval queries using conversational history.
- **Token Streaming**: Real-time streaming generation via `st.write_stream` with multi-model auto-fallback.
- **Grounded Verification**: Expandable source viewer displays the exact source passages retrieved for full transparency.

---

## 🛠️ Technology Stack
- **Frontend / UI**: Streamlit
- **PDF Extraction**: PyPDF2 / pypdf
- **Orchestration**: LangChain (`langchain-core`, `langchain-openai`, `langchain-google-genai`, `langchain-groq`, `langchain-community`, `langchain-text-splitters`)
- **Embeddings**: Google Gemini `text-embedding-004`, OpenAI `text-embedding-3-small`, or FastEmbed
- **LLMs**: Google Gemini (`gemini-2.0-flash`, `gemini-1.5-flash`), OpenAI (`gpt-4o-mini`), Groq (`llama-3.3-70b-versatile`), OpenRouter
- **Vector Index**: FAISS (in-memory, cached per session)

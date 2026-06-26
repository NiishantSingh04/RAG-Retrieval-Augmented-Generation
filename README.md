# 🧠 Retrieval-Augmented Generation (RAG) Pipeline

[![Python Version](https://img.shields.io/badge/Python-3.12%2B-blue.svg?style=flat-square&logo=python)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-1.3%2B-orange.svg?style=flat-square)](https://github.com/langchain-ai/langchain)
[![FAISS](https://img.shields.io/badge/Vector%20DB-FAISS-green.svg?style=flat-square)](https://github.com/facebookresearch/faiss)
[![LLM Provider](https://img.shields.io/badge/LLM-Groq-red.svg?style=flat-square)](https://groq.com/)
[![License](https://img.shields.io/badge/license-MIT-lightgrey.svg?style=flat-square)](LICENSE)

A production-ready, modular **Retrieval-Augmented Generation (RAG)** pipeline designed to ingest, chunk, embed, index, and query various document types. The project integrates local sentence embeddings and vector search with state-of-the-art LLMs accessed via the Groq API to provide context-aware, summarized answers to user queries.

---

## 🗺️ Architecture Overview

The system uses a standard RAG pattern, dividing operations into two key phases: **Ingestion & Indexing** and **Query & Generation**.

```mermaid
flowchart TD
    subgraph Ingestion & Indexing
        A[Raw Documents<br/>PDF, TXT, CSV, Word, Excel, JSON] --> B[data_loader.py<br/>Langchain Document Loaders]
        B --> C[embedding.py<br/>RecursiveCharacterTextSplitter]
        C --> D[embedding.py<br/>SentenceTransformer]
        D --> E[vectorstore.py<br/>FaissVectorStore]
        E -->|Persist Index & Metadata| F[(faiss_store/)]
    end

    subgraph Query & Generation
        G[User Query] --> H[search.py<br/>RAGSearch Engine]
        F -->|Load Index| H
        H -->|1. Vector Similarity Search| E
        E -->|2. Retrieve top-k context| H
        H -->|3. Construct Prompt with Context| I[Groq ChatGroq LLM<br/>qwen/qwen3-32b]
        I -->|4. Generate Response| J[Structured Summary / Answer]
    end

    style Ingestion & Indexing fill:#eef2f7,stroke:#3b5998,stroke-width:2px
    style Query & Generation fill:#fdf6ec,stroke:#e6a23c,stroke-width:2px
    style F fill:#f0f9eb,stroke:#67c23a,stroke-width:2px
    style J fill:#fef0f0,stroke:#f56c6c,stroke-width:2px
```

---

## ✨ Features

- **Multi-Format Document Ingestion**: Seamlessly load and process **PDF**, **TXT**, **CSV**, **Excel** (`.xlsx`), **Word** (`.docx`), and **JSON** files using LangChain loaders.
- **Semantic Text Chunking**: Utilizes `RecursiveCharacterTextSplitter` with optimal paragraph/sentence separators and configured chunk overlap to preserve document context boundaries.
- **Local Embedding Generation**: Computes embeddings locally using HuggingFace's `sentence-transformers` library (specifically optimized with `all-MiniLM-L6-v2` for speed and accuracy).
- **Fast Similarity Search**: Employs **FAISS** (Facebook AI Similarity Search) to index vector representations and perform rapid L2 similarity queries.
- **Groq LLM Synthesis**: Connects to the Groq API via `ChatGroq` (using Qwen-32B or other high-performance models) to synthesize retrieved documents and produce concise summaries.
- **Experimental Playgrounds**: Contains interactive Jupyter Notebooks for testing alternative databases (such as **Typesense**) and tracking data ingestion experiments.

---

## 📂 Project Structure

```directory
├── Data/                       # Source document files for ingestion
│   ├── pdf/                    # PDF files (e.g., C Programming, Distributed Systems)
│   ├── text_files/             # Plaintext files (e.g., Machine Learning, Python guides)
│   └── VectorStore/            # (Optional) Pre-computed vector indices
├── NoteBook/                   # Jupyter Notebooks for analysis & prototyping
│   ├── document.ipynb          # Step-by-step data ingestion flow
│   └── pdf_loader.ipynb        # In-depth PDF loading and splitting tests
├── src/                        # Core Application Source Code
│   ├── __init__.py             # Module initialization
│   ├── data_loader.py          # Unified data loaders for multiple formats
│   ├── embedding.py            # Recursive chunking & sentence embeddings pipeline
│   ├── search.py               # Main RAG querying and Groq LLM orchestration
│   └── vectorstore.py          # FAISS index persistence, management, and similarity search
├── app.py                      # Main entrypoint script for RAG execution
├── main.py                     # Project entrypoint utility
├── pyproject.toml              # Modern Python packaging configuration (uv/pep621 compatible)
├── requirements.txt            # Python dependencies lists
├── typesense.ipynb             # Experimental notebook exploring Typesense search engine integration
└── .env                        # Local environment variables (API keys, etc.)
```

---

## 🚀 Getting Started

### 📋 Prerequisites

- **Python 3.12+**
- A **Groq API Key** (Get one from the [Groq Console](https://console.groq.com/))
- (Optional) [uv](https://github.com/astral-sh/uv) - Fast Python package installer and resolver

---

### ⚙️ Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/RAG-Retrieval-Augmented-Generation.git
   cd RAG-Retrieval-Augmented-Generation
   ```

2. **Set up a Virtual Environment**:
   - Using standard `venv`:
     ```bash
     python -m venv .venv
     source .venv/bin/activate  # On Windows: .venv\Scripts\activate
     ```
   - Using `uv` (recommended):
     ```bash
     uv venv
     source .venv/bin/activate  # On Windows: .venv\Scripts\activate
     ```

3. **Install Dependencies**:
   - Using `pip`:
     ```bash
     pip install -r requirements.txt
     ```
   - Using `uv`:
     ```bash
     uv pip install -r requirements.txt
     ```

4. **Configure Environment Variables**:
   Create a `.env` file in the root directory and add your Groq API key:
   ```env
   GROQ_API_KEY="your-groq-api-key-here"
   ```

---

## 💻 Usage

### 1. Preparing Data
Place your documents in the `Data/` folder (supported structures inside `Data/pdf/`, `Data/text_files/`, etc.). 

### 2. Building the Index & Running Queries
You can run the full RAG search pipeline using `app.py`:

```bash
python app.py
```

Inside `app.py`, the system:
1. Resolves document paths under `Data/` and ingests all text.
2. Chunk-splits the text and computes embedding vectors.
3. Builds and persists the FAISS database index into a directory named `faiss_store/`.
4. Executes similarity searches on FAISS and retrieves the most relevant paragraphs.
5. Feeds the query + context to Groq to generate a final answer.

### 📝 Example Usage snippet

```python
from src.data_loader import load_all_documents
from src.vectorstore import FaissVectorStore
from src.search import RAGSearch

# 1. Ingest Documents
docs = load_all_documents("Data")

# 2. Build and Save Vector Index
store = FaissVectorStore("faiss_store")
store.build_from_documents(docs)

# 3. Initialize search engine and query
rag_search = RAGSearch(persist_dir="faiss_store")
query = "What are the types of Machine Learning?"
summary = rag_search.search_and_summarize(query, top_k=3)

print("Answer:", summary)
```

---

## 🛠️ Technologies & Libraries

- **LangChain & LangChain Community**: Orchestrates document loading, text splitting (`RecursiveCharacterTextSplitter`), and wrappers around the LLM.
- **HuggingFace Sentence-Transformers**: Encodes text chunks into dense 384-dimensional vector representations locally.
- **FAISS**: CPU-optimized vector indexing for lightning-fast similarity searching.
- **ChatGroq**: Interfaces with Groq's high-speed inference engine to utilize state-of-the-art open-weights models (such as Qwen).
- **PyMuPDF & PyPDF**: Extracts clean text layouts from complex PDF structures.
- **Unstructured / Docx2txt / CSV**: Specialized tools to extract data from tables, word processing documents, and spreadsheets.

---

## 🧪 Experimental Notebooks

- **[document.ipynb](file:///Users/nishantsingh04/Documents/RAG-Retrieval-Augmented%20Generation/NoteBook/document.ipynb)**: Walkthrough of data loading, document schemas, and parsing techniques.
- **[pdf_loader.ipynb](file:///Users/nishantsingh04/Documents/RAG-Retrieval-Augmented%20Generation/NoteBook/pdf_loader.ipynb)**: Detailed tests comparing `PyPDFLoader` vs. `PyMuPDFLoader` chunk distributions.
- **[typesense.ipynb](file:///Users/nishantsingh04/Documents/RAG-Retrieval-Augmented%20Generation/typesense.ipynb)**: Experiments on indexing metadata and searching documents via **Typesense Cloud**, providing an alternative to FAISS for full-text search combined with vector capabilities.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

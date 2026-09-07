# 🔍 HYBRID-RAG

> A production-grade Retrieval-Augmented Generation (RAG) system over SEC 10-K filings.
> Ask questions in plain English — get grounded, cited answers with zero hallucination.

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green.svg)](https://fastapi.tiangolo.com)
[![LangChain](https://img.shields.io/badge/LangChain-0.3-orange.svg)](https://langchain.com)
[![RAGAS](https://img.shields.io/badge/RAGAS-0.2-purple.svg)](https://docs.ragas.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🗂️ Project Structure

```
HYBRID-RAG/
│
├── app.py                        # Streamlit frontend (pure HTTP client to FastAPI)
├── config.py                     # Central settings — Pydantic BaseSettings + .env
├── logger.py                     # Centralised get_logger() with lru_cache
├── setup.py                      # Editable install — enables clean package imports
├── requirements.txt              # Pinned dependencies
├── .env                          # API keys and settings (gitignored)
├── .gitignore
│
├── api/
│   ├── __init__.py
│   ├── main.py                   # FastAPI factory + lifespan startup + rate limiter
│   ├── routes.py                 # /health, /query, /upload route handlers
│   └── schemas.py                # Pydantic request/response contracts
│
├── ingestion/
│   ├── __init__.py
│   ├── loader.py                 # load_pdf() — PyPDFLoader, one Document per page
│   ├── chunker.py                # chunk_documents() — splitter + artifact filter
│   └── embedder.py               # embed_and_store() / load_store() — ChromaDB
│
├── retrieval/
│   ├── __init__.py
│   ├── bm25_retriever.py         # BM25Retriever — in-memory keyword ranking
│   ├── semantic_retriever.py     # SemanticRetriever — ChromaDB vector similarity
│   └── hybrid.py                 # hybrid_retrieve() — Reciprocal Rank Fusion (RRF)
│
├── generation/
│   ├── __init__.py
│   ├── prompts.py                # SYSTEM_PROMPT + build_user_message()
│   └── generator.py              # generate_answer() — GPT-4o-mini via LangChain
│
├── evaluation/
│   ├── __init__.py
│   └── ragas_eval.py             # run_evaluation() — 4 RAGAS quality metrics
│
├── tests/
│   ├── __init__.py
│   ├── test_ingestion.py         # Unit tests for loader and chunker
│   ├── test_retrieval.py         # Unit tests for BM25 and RRF fusion
│   └── test_api.py               # Integration tests for all API endpoints
│
└── data/
    ├── raw/                      # Place your PDF files here (gitignored)
    │   └── aapl-10k-2024.pdf     # Default document loaded on startup
    └── chroma/                   # ChromaDB vector store (gitignored, auto-created)
```

One folder = one concern. One file = one job.

---

## ⚙️ How It Works

The pipeline has five clean layers, each in its own module:

```
PDF  →  [Ingestion]  →  [Retrieval]  →  [Generation]  →  Answer + Citations
                             ↑
                     BM25 + Semantic
                      fused via RRF
```

**1. Ingestion** — `loader.py` reads the PDF page by page. `chunker.py` splits pages into 500-character overlapping chunks, then filters out browser-print artifact chunks (SEC EDGAR header/footer noise). `embedder.py` embeds clean chunks with `text-embedding-3-small` and persists them in ChromaDB.

**2. Retrieval** — Every query runs two retrievers in parallel. `bm25_retriever.py` does exact keyword matching (good for tickers, figures, legal terms). `semantic_retriever.py` does vector similarity search (good for meaning and paraphrase). `hybrid.py` fuses both ranked lists using **Reciprocal Rank Fusion** (RRF, k=60). Both retrievers are built **once on startup** and reused across all requests — no per-request index rebuilding.

**3. Generation** — `prompts.py` builds a strict PERMISSION / PROHIBITION / FALLBACK prompt. The LLM is only allowed to use the retrieved passages — no training knowledge, no extrapolation. `generator.py` calls GPT-4o-mini and returns a structured `RAGResponse` with the answer, source citations, model name, and token count.

**4. API** — `main.py` wires FastAPI with a lifespan manager that builds both BM25 and ChromaDB retrievers on startup and stores them in `app.state`. `routes.py` exposes three endpoints with rate limiting. `schemas.py` validates every request and response with Pydantic.

**5. Evaluation** — `ragas_eval.py` scores the pipeline on four RAGAS metrics (Faithfulness, Answer Relevancy, Context Precision, Context Recall) against curated question/answer pairs.

---

## 🛡️ Production Features

| Feature | Implementation |
|---------|---------------|
| **Rate Limiting** | `slowapi` — 100 req/min on `/query`, 10 req/min on `/upload` |
| **Path Traversal Protection** | `Path(filename).name` sanitization on uploads |
| **BM25 built once** | Index built at startup, stored in `app.state` — not per-request |
| **Single eval module** | One `ragas_eval.py` — no duplicates |
| **Structured logging** | `get_logger(__name__)` in every module |
| **Pydantic validation** | All API contracts validated on input and output |
| **Test suite** | `pytest` — unit + integration tests across all layers |

---

## 🚀 Quick Start (Local)

### 1. Clone and set up environment

```powershell
git clone https://github.com/aravind00006/HYBRID-RAG.git
cd HYBRID-RAG

python -m venv venv
.\venv\Scripts\Activate.ps1

pip install -r requirements.txt
pip install -e .
```

### 2. Configure environment variables

Create a `.env` file in the project root:

```env
# Required
OPENAI_API_KEY=sk-...

# Optional — LangSmith observability
LANGCHAIN_API_KEY=ls__...
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=hybrid-rag
```

### 3. Add your PDF

```
data/raw/aapl-10k-2024.pdf   ← default document (update DEFAULT_PDF_PATH in .env to change)
```

### 4. Start the API

```powershell
uvicorn api.main:app --reload
```

On first run the server embeds the PDF and builds ChromaDB — takes ~30 seconds. Every run after that loads from disk instantly.

```
API      → http://localhost:8000
API Docs → http://localhost:8000/docs
```

### 5. Start the Streamlit frontend

Open a **second terminal** and run:

```powershell
streamlit run app.py
```

```
UI → http://localhost:8501
```

### 6. Run the tests

```powershell
pytest tests/ -v
```

---

## 🌐 API Reference

Base URL: `http://localhost:8000/api/v1`

Interactive docs: `http://localhost:8000/docs`

### `GET /health`

```json
{
  "status": "ok",
  "retriever_ready": true,
  "model": "gpt-4o-mini"
}
```

### `POST /query`

**Rate limit:** 100 requests/minute

**Request**
```json
{
  "question": "What was Apple's total revenue in fiscal year 2024?",
  "top_k": 5
}
```

**Response**
```json
{
  "answer": "Apple's total net sales for fiscal year 2024 were $391.035 billion...",
  "sources": [
    {
      "source": "aapl-10k-2024.pdf",
      "page": 25,
      "preview": "Total net sales $391,035 $383,285 $394,328..."
    }
  ],
  "model": "gpt-4o-mini",
  "tokens_used": 842,
  "question": "What was Apple's total revenue in fiscal year 2024?"
}
```

**Error codes**

| Code | Meaning |
|------|---------|
| 429  | Rate limit exceeded |
| 503  | Retriever not ready — server still starting up |
| 500  | Retrieval or generation pipeline error |

### `POST /upload`

**Rate limit:** 10 requests/minute

Upload a PDF and set it as the active document. Filename is sanitized server-side to prevent path traversal attacks.

**Request** — `multipart/form-data`, field name `file`.

**Response**
```json
{
  "filename": "msft-10k-2024.pdf",
  "chunks_created": 512,
  "message": "'msft-10k-2024.pdf' loaded successfully. You can now ask questions from it."
}
```

---

## 📐 Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | **Required.** OpenAI API key. |
| `LANGCHAIN_API_KEY` | `""` | LangSmith API key (optional). |
| `LANGCHAIN_TRACING_V2` | `true` | Enable LangSmith tracing. |
| `LANGCHAIN_PROJECT` | `hybrid-rag` | LangSmith project name. |
| `LLM_MODEL` | `gpt-4o-mini` | OpenAI chat model. |
| `LLM_TEMPERATURE` | `0.0` | LLM temperature (0.0–2.0). |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model. |
| `CHROMA_PATH` | `./data/chroma` | ChromaDB persistence directory. |
| `COLLECTION_NAME` | `hybrid_rag` | ChromaDB collection name. |
| `CHUNK_SIZE` | `500` | Characters per chunk. |
| `CHUNK_OVERLAP` | `100` | Overlap between consecutive chunks. |
| `TOP_K` | `5` | Chunks retrieved per query. |
| `DEFAULT_PDF_PATH` | `data/raw/aapl-10k-2024.pdf` | PDF loaded on server startup. |
| `RATE_LIMIT_QUERY` | `100/minute` | Rate limit for /query endpoint. |
| `RATE_LIMIT_UPLOAD` | `10/minute` | Rate limit for /upload endpoint. |

---

## 📊 Evaluation

The pipeline is evaluated with [RAGAS](https://docs.ragas.io/) on four metrics:

| Metric | What it measures | Pass threshold |
|--------|-----------------|----------------|
| **Faithfulness** | Answer claims traceable to context — anti-hallucination | ≥ 0.80 |
| **Answer Relevancy** | Answer actually addresses the question asked | ≥ 0.80 |
| **Context Precision** | Retrieved chunks are relevant (no noise) | ≥ 0.80 |
| **Context Recall** | Retrieval found all information needed to answer | ≥ 0.80 |

Run evaluation:

```powershell
python -m evaluation.ragas_eval
```

Sample output:

```
RAGAS Evaluation — Status: PASS
--------------------------------------------------
  faithfulness           0.9200   [##################  ]  PASS
  answer_relevancy       0.8800   [#################   ]  PASS
  context_precision      0.7500   [###############     ]  PASS
  context_recall         1.0000   [####################]  PASS
```

---

## 🔭 Observability

LangSmith tracing is active when `LANGCHAIN_TRACING_V2=true` and `LANGCHAIN_API_KEY` are set in `.env`. Every LLM call, retrieval step, and token count is logged automatically. View traces at [smith.langchain.com](https://smith.langchain.com).

---

## ☁️ Deployment — Hugging Face Spaces

This project is deployed on [Hugging Face Spaces](https://huggingface.co/spaces) as a **Docker** Space.

### How it works on Spaces

Hugging Face Spaces runs a Docker container. We expose the **Streamlit frontend** on the public URL, while the FastAPI backend runs as an internal process inside the same container.

### Files needed for deployment

```
HYBRID-RAG/
├── Dockerfile           ← builds the container
├── README.md            ← top section becomes the Space card (title, emoji, SDK)
└── .env.example         ← shows required env vars (never commit real .env)
```

### Step 1 — Add the Space metadata block to the top of README.md

Hugging Face reads a YAML block at the very top of `README.md` to configure the Space card:

```yaml
---
title: HYBRID-RAG
emoji: 🔍
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---
```

> This block is already included at the top of this README in the deployed version.

### Step 2 — Create the Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .
RUN pip install -e .

# Expose Streamlit port (Spaces default)
EXPOSE 7860

# Start both FastAPI (background) and Streamlit (foreground)
CMD uvicorn api.main:app --host 0.0.0.0 --port 8000 & \
    streamlit run app.py --server.port 7860 --server.address 0.0.0.0
```

### Step 3 — Set Secrets on Hugging Face

Go to your Space → **Settings → Repository secrets** and add:

| Secret name | Value |
|-------------|-------|
| `OPENAI_API_KEY` | Your OpenAI key |
| `LANGCHAIN_API_KEY` | Your LangSmith key (optional) |
| `LANGCHAIN_TRACING_V2` | `true` |
| `LANGCHAIN_PROJECT` | `hybrid-rag` |

These are injected as environment variables at runtime — never hardcoded.

### Step 4 — Update `app.py` for Spaces

On Spaces, FastAPI runs on port 8000 internally. Update the `API_BASE` in `app.py`:

```python
import os
# Locally: http://localhost:8000 | On Spaces: same (internal process)
API_BASE = os.getenv("API_BASE", "http://localhost:8000/api/v1")
```

### Step 5 — Push to Hugging Face

```powershell
# Install huggingface_hub CLI
pip install huggingface_hub

# Login
huggingface-cli login

# Create a new Space (do this once)
huggingface-cli repo create HYBRID-RAG --type space --space_sdk docker

# Add HF as a remote and push
git remote add hf https://huggingface.co/spaces/<your-username>/HYBRID-RAG
git push hf main
```

Every `git push hf main` triggers a rebuild and redeploy automatically.

### Important Notes for Spaces

- **ChromaDB persistence** — Spaces has ephemeral storage. The ChromaDB index is rebuilt on every cold start (the PDF re-embeds on startup). For persistent storage, use Hugging Face Datasets or an external vector DB.
- **Default PDF** — Include `aapl-10k-2024.pdf` in the repo under `data/raw/` or the startup will fail. Add it via Git LFS if it's large.
- **Port** — Spaces expects port `7860` for the public URL. Streamlit binds to this; FastAPI runs internally on `8000`.

---

## 🧰 Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | OpenAI GPT-4o-mini |
| Embeddings | OpenAI text-embedding-3-small |
| Vector Store | ChromaDB (local) |
| Keyword Search | BM25Okapi (rank-bm25) |
| Fusion | Reciprocal Rank Fusion (RRF, k=60) |
| Orchestration | LangChain 0.3 |
| API | FastAPI + Uvicorn |
| Rate Limiting | slowapi |
| Frontend | Streamlit |
| Evaluation | RAGAS 0.2.x |
| Observability | LangSmith |
| Config | Pydantic BaseSettings |
| Testing | pytest + httpx |
| Deployment | Hugging Face Spaces (Docker) |

---

## 🗺️ Roadmap

- [x] Hybrid BM25 + Semantic retrieval via RRF
- [x] FastAPI backend with rate limiting
- [x] Streamlit frontend with PDF upload
- [x] RAGAS evaluation suite
- [x] LangSmith observability
- [x] Hugging Face Spaces deployment
- [ ] GitHub Actions CI pipeline with RAGAS quality gates
- [ ] Streaming responses via Server-Sent Events
- [ ] Multi-document support with per-session isolated collections

---

## 📄 License

MIT
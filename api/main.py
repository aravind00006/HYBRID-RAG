"""
FastAPI application factory and lifespan manager.
Single responsibility: app wiring only.

"""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router
from config import get_settings
from ingestion.chunker import chunk_documents
from ingestion.embedder import embed_and_store, load_store
from ingestion.loader import load_pdf
from retrieval.bm25_retriever import BM25Retriever
from logger import get_logger

logger = get_logger(__name__)

#  Lifespan 

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Manage startup and shutdown of shared resources.

    """
    settings = get_settings()
    logger.info("Starting HYBRID-RAG API — loading default document.")

    # Step 1 — Load and chunk the default PDF.
    pages  = load_pdf(settings.default_pdf_path)
    chunks = chunk_documents(pages)

    # Step 2 — Load ChromaDB from disk or build it fresh.
    chroma_path = settings.chroma_path
    if Path(chroma_path).exists() and any(Path(chroma_path).iterdir()):
        logger.info("ChromaDB found at '%s' — loading from disk.", chroma_path)
        store = load_store()
    else:
        logger.info("ChromaDB not found — embedding %d chunks.", len(chunks))
        store = embed_and_store(chunks)

    # Step 3 — Build full BM25Retriever ONCE at startup.
    # BM25 is always in-memory — must be rebuilt on each server start.
    bm25 = BM25Retriever(chunks)

    # Step 4 — Store everything in app.state.
    app.state.chunks = chunks
    app.state.store  = store
    app.state.bm25   = bm25

    logger.info("HYBRID-RAG API startup complete — ready to serve requests.")

    yield  # Server is live here — handling requests.

    # Shutdown
    logger.info("HYBRID-RAG API shutting down.")


#  App factory 

def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    """
    app = FastAPI(
        title="HYBRID-RAG API",
        description=(
            "Production-grade Hybrid RAG system over SEC 10-K filings. "
            "Combines BM25 keyword search and semantic vector search via "
            "Reciprocal Rank Fusion (RRF). Ask questions in plain English, "
            "get grounded answers with citations."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    #  CORS 
    # Allows the Streamlit frontend (localhost:8501) to call this API.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    #  Router 
    app.include_router(router, prefix="/api/v1")

    return app


#  Entry point
app = create_app()
"""
api/main.py — FastAPI application factory and lifespan manager.
Single responsibility: app wiring only.
"""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from rank_bm25 import BM25Okapi
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from api.routes import router
from config import get_settings
from ingestion.chunker import chunk_documents
from ingestion.embedder import embed_and_store, load_store
from ingestion.loader import load_pdf
from logger import get_logger

logger = get_logger(__name__)

# Rate limiter 
# get_remote_address extracts the client IP from the request for per-IP limits.
# The limiter instance is imported by routes.py to decorate individual endpoints.
limiter = Limiter(key_func=get_remote_address)

# Lifespan 

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

    # Step 3 — Build BM25 index ONCE at startup.
    # BM25 is always in-memory — must be rebuilt on each server start.
    logger.info("Building BM25 index over %d chunks.", len(chunks))
    tokenized  = [c.page_content.lower().split() for c in chunks]
    bm25_index = BM25Okapi(tokenized)
    logger.info("BM25 index built successfully.")

    # Step 4 — Store everything in app.state.
    app.state.chunks     = chunks
    app.state.store      = store
    app.state.bm25_index = bm25_index

    logger.info("HYBRID-RAG API startup complete — ready to serve requests.")

    yield  # Server is live here — handling requests.

    # Shutdown
    logger.info("HYBRID-RAG API shutting down.")
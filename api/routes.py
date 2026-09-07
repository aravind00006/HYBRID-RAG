"""
api/routes.py — FastAPI route handlers.
Single responsibility: HTTP request handling only.

"""

import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, UploadFile, File, status

from api.schemas import (
    ErrorResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    SourceReference,
    UploadResponse,
)
from config import get_settings
from generation.generator import generate_answer
from ingestion.chunker import chunk_documents
from ingestion.embedder import embed_and_store
from ingestion.loader import load_pdf
from retrieval.bm25_retriever import BM25Retriever
from retrieval.hybrid import hybrid_retrieve
from retrieval.semantic_retriever import SemanticRetriever
from logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


# Health 

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness check",
)
async def health(request: Request) -> HealthResponse:
    """
    Return service status and whether retrievers are ready.

    """
    settings = get_settings()
    ready = (
        getattr(request.app.state, "chunks", None) is not None
        and getattr(request.app.state, "store", None) is not None
    )
    logger.info("Health check — retriever_ready: %s.", ready)
    return HealthResponse(
        status="ok",
        retriever_ready=ready,
        model=settings.llm_model,
    )
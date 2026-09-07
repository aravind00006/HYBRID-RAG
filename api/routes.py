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

# Query 

@router.post(
    "/query",
    response_model=QueryResponse,
    responses={
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
        503: {"model": ErrorResponse, "description": "Retrievers not ready"},
        500: {"model": ErrorResponse, "description": "Pipeline error"},
    },
    summary="Ask a question from the loaded document",
)
async def query(body: QueryRequest, request: Request) -> QueryResponse:
    """
    Run the full RAG pipeline for the given question.

    """
    chunks = getattr(request.app.state, "chunks", None)
    store  = getattr(request.app.state, "store",  None)

    if chunks is None or store is None:
        logger.warning("Query received but retrievers are not ready.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Retrievers are not ready. The service may still be starting up.",
        )

    logger.info("Query received: '%s' (top_k=%d).", body.question[:60], body.top_k)

    # Load pre-built retrievers from app.state — no index rebuilding.
    bm25            = BM25Retriever.__new__(BM25Retriever)
    bm25._documents = chunks
    bm25._bm25      = request.app.state.bm25_index

    semantic = SemanticRetriever(store)

    try:
        results = hybrid_retrieve(
            query=body.question,
            bm25=bm25,
            semantic=semantic,
            top_k=body.top_k,
        )
    except Exception as exc:
        logger.error("Retrieval failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Retrieval failed: {exc}",
        )

    try:
        response = generate_answer(body.question, results)
    except Exception as exc:
        logger.error("Generation failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Generation failed: {exc}",
        )

    logger.info(
        "Query complete — tokens: %d, sources: %d.",
        response.tokens_used,
        len(response.sources),
    )

    return QueryResponse(
        answer=response.answer,
        sources=[
            SourceReference(
                source=s["source"],
                page=s["page"],
                preview=s["preview"],
            )
            for s in response.sources
        ],
        model=response.model,
        tokens_used=response.tokens_used,
        question=body.question,
    )
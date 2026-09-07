"""
Integration tests for the FastAPI API layer.
Tests /health, /query, and /upload endpoints via httpx TestClient.
"""

import io
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from api.main import create_app
from retrieval.bm25_retriever import BM25Retriever
from retrieval.semantic_retriever import SemanticRetriever


#  Fixtures 

def make_mock_chunk(text: str = "Apple revenue was $391 billion.", page: int = 0):
    """Helper — build a mock chunk dict as returned by hybrid_retrieve()."""
    return {
        "text":     text,
        "score":    0.95,
        "metadata": {"source": "aapl-10k-2024.pdf", "page": page},
        "rank":     1,
    }


@pytest.fixture(scope="function")
def client():
    """
    Create a test FastAPI client with mocked app.state.

    """
    mock_chunks = [make_mock_chunk()]
    mock_store  = MagicMock()

    # Patch all startup I/O so tests run without real files or API keys.
    with (
        patch("api.main.load_pdf",         return_value=mock_chunks),
        patch("api.main.chunk_documents",  return_value=mock_chunks),
        patch("api.main.load_store",       return_value=mock_store),
        patch("api.main.embed_and_store",  return_value=mock_store),
        patch("api.main.BM25Okapi"),
        patch("pathlib.Path.exists",       return_value=True),
        patch("pathlib.Path.iterdir",      return_value=iter(["file"])),
    ):
        app = create_app()
        app.state.chunks     = mock_chunks
        app.state.store      = mock_store
        app.state.bm25_index = MagicMock()

        with TestClient(app) as c:
            yield c


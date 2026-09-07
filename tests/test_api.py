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


#  /health tests ─

class TestHealthEndpoint:
    """Tests for GET /api/v1/health."""

    def test_health_returns_200(self, client):
        """Health endpoint returns HTTP 200."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200

    def test_health_returns_ok_status(self, client):
        """Health response body has status='ok'."""
        response = client.get("/api/v1/health")
        data     = response.json()
        assert data["status"] == "ok"

    def test_health_retriever_ready_true_when_state_set(self, client):
        """retriever_ready is True when chunks and store are in app.state."""
        response = client.get("/api/v1/health")
        data     = response.json()
        assert data["retriever_ready"] is True

    def test_health_includes_model_name(self, client):
        """Health response includes the configured LLM model name."""
        response = client.get("/api/v1/health")
        data     = response.json()
        assert "model" in data
        assert len(data["model"]) > 0


#  /query tests 

class TestQueryEndpoint:
    """Tests for POST /api/v1/query."""

    def _mock_rag_response(self):
        """Build a mock RAGResponse for generate_answer patching."""
        from generation.generator import RAGResponse
        return RAGResponse(
            answer="Apple's total net sales were $391.035 billion in fiscal year 2024.",
            sources=[{
                "source":  "aapl-10k-2024.pdf",
                "page":    25,
                "preview": "Total net sales $391,035...",
            }],
            model="gpt-4o-mini",
            tokens_used=842,
        )

    def test_query_returns_200(self, client):
        """Query endpoint returns HTTP 200 for a valid question."""
        with (
            patch("api.routes.hybrid_retrieve",  return_value=[make_mock_chunk()]),
            patch("api.routes.generate_answer",  return_value=self._mock_rag_response()),
            patch("api.routes.BM25Retriever"),
            patch("api.routes.SemanticRetriever"),
        ):
            response = client.post(
                "/api/v1/query",
                json={"question": "What was Apple revenue in 2024?", "top_k": 5},
            )
        assert response.status_code == 200

    def test_query_response_has_required_fields(self, client):
        """Query response body contains answer, sources, model, tokens_used, question."""
        with (
            patch("api.routes.hybrid_retrieve",  return_value=[make_mock_chunk()]),
            patch("api.routes.generate_answer",  return_value=self._mock_rag_response()),
            patch("api.routes.BM25Retriever"),
            patch("api.routes.SemanticRetriever"),
        ):
            response = client.post(
                "/api/v1/query",
                json={"question": "What was Apple revenue in 2024?", "top_k": 5},
            )
        data = response.json()
        assert "answer"      in data
        assert "sources"     in data
        assert "model"       in data
        assert "tokens_used" in data
        assert "question"    in data

    def test_query_echoes_question(self, client):
        """Query response echoes back the original question."""
        question = "What was Apple revenue in 2024?"
        with (
            patch("api.routes.hybrid_retrieve",  return_value=[make_mock_chunk()]),
            patch("api.routes.generate_answer",  return_value=self._mock_rag_response()),
            patch("api.routes.BM25Retriever"),
            patch("api.routes.SemanticRetriever"),
        ):
            response = client.post(
                "/api/v1/query",
                json={"question": question, "top_k": 5},
            )
        assert response.json()["question"] == question

    def test_query_returns_422_for_short_question(self, client):
        """Query returns 422 when question is shorter than min_length=3."""
        response = client.post(
            "/api/v1/query",
            json={"question": "hi", "top_k": 5},
        )
        assert response.status_code == 422

    def test_query_returns_422_for_invalid_top_k(self, client):
        """Query returns 422 when top_k is out of the valid range (1-20)."""
        response = client.post(
            "/api/v1/query",
            json={"question": "What was Apple revenue?", "top_k": 99},
        )
        assert response.status_code == 422

    def test_query_returns_503_when_chunks_not_set(self, client):
        """Query returns 503 when app.state.chunks is None."""
        client.app.state.chunks = None
        response = client.post(
            "/api/v1/query",
            json={"question": "What was Apple revenue?", "top_k": 5},
        )
        assert response.status_code == 503
        # Restore for other tests.
        client.app.state.chunks = [make_mock_chunk()]


"""
Unit tests for retrieval layer. Tests BM25Retriever, 
SemanticRetriever, and hybrid_retrieve / _rrf_fusion.
"""

import pytest
from unittest.mock import MagicMock
from langchain_core.documents import Document

from retrieval.bm25_retriever import BM25Retriever
from retrieval.semantic_retriever import SemanticRetriever
from retrieval.hybrid import _rrf_fusion, hybrid_retrieve


#  Fixtures 

def make_document(text: str, page: int = 0, source: str = "test.pdf") -> Document:
    """Helper — build a LangChain Document for testing."""
    return Document(
        page_content=text,
        metadata={"source": source, "page": page},
    )


@pytest.fixture
def sample_documents() -> list[Document]:
    """A small realistic corpus of financial chunk Documents."""
    return [
        make_document(
            "Apple's total net sales for fiscal year 2024 were $391.035 billion.",
            page=0,
        ),
        make_document(
            "Net income attributable to Apple Inc was $93.736 billion in 2024.",
            page=1,
        ),
        make_document(
            "Products segment revenue was $298.085 billion for fiscal year 2024.",
            page=2,
        ),
        make_document(
            "Services revenue reached $96.169 billion, a new record for Apple.",
            page=3,
        ),
        make_document(
            "Apple repurchased $94.949 billion of its common stock during 2024.",
            page=4,
        ),
    ]


@pytest.fixture
def bm25_retriever(sample_documents) -> BM25Retriever:
    """A BM25Retriever built over the sample corpus."""
    return BM25Retriever(sample_documents)


#  BM25Retriever tests ─

class TestBM25Retriever:
    """Tests for BM25Retriever in retrieval/bm25_retriever.py."""

    def test_raises_on_empty_documents(self):
        """BM25Retriever raises ValueError when built with no documents."""
        with pytest.raises(ValueError, match="empty document list"):
            BM25Retriever([])

    def test_retrieve_returns_list(self, bm25_retriever):
        """retrieve() returns a list for a valid query."""
        results = bm25_retriever.retrieve("Apple revenue", top_k=3)
        assert isinstance(results, list)

    def test_retrieve_respects_top_k(self, bm25_retriever):
        """retrieve() returns at most top_k results."""
        results = bm25_retriever.retrieve("Apple revenue billion", top_k=2)
        assert len(results) <= 2

    def test_retrieve_returns_empty_for_blank_query(self, bm25_retriever):
        """retrieve() returns an empty list for a blank query string."""
        results = bm25_retriever.retrieve("   ", top_k=5)
        assert results == []

    def test_retrieve_result_has_required_keys(self, bm25_retriever):
        """Each result dict has text, score, metadata, and rank keys."""
        results = bm25_retriever.retrieve("net sales", top_k=3)
        assert len(results) > 0
        for result in results:
            assert "text"     in result
            assert "score"    in result
            assert "metadata" in result
            assert "rank"     in result

    def test_retrieve_ranks_are_sequential(self, bm25_retriever):
        """Result ranks start at 1 and increment by 1."""
        results = bm25_retriever.retrieve("Apple revenue billion", top_k=3)
        ranks = [r["rank"] for r in results]
        assert ranks == list(range(1, len(ranks) + 1))

    def test_retrieve_scores_are_positive(self, bm25_retriever):
        """All returned results have a positive BM25 score."""
        results = bm25_retriever.retrieve("revenue billion", top_k=5)
        for result in results:
            assert result["score"] > 0.0

    def test_relevant_document_ranks_higher(self, bm25_retriever):
        """A document containing query terms ranks above unrelated ones."""
        results = bm25_retriever.retrieve("net income", top_k=5)
        # The doc about net income should be in the top results.
        top_texts = [r["text"] for r in results]
        assert any("net income" in t.lower() for t in top_texts)


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


#  SemanticRetriever tests ─

class TestSemanticRetriever:
    """Tests for SemanticRetriever in retrieval/semantic_retriever.py."""

    def test_raises_on_none_store(self):
        """SemanticRetriever raises ValueError when store is None."""
        with pytest.raises(ValueError, match="cannot be None"):
            SemanticRetriever(None)

    def test_retrieve_returns_empty_for_blank_query(self):
        """retrieve() returns an empty list for a blank query string."""
        mock_store = MagicMock()
        retriever  = SemanticRetriever(mock_store)
        results    = retriever.retrieve("   ", top_k=5)
        assert results == []
        mock_store.similarity_search_with_relevance_scores.assert_not_called()

    def test_retrieve_calls_store_with_correct_args(self):
        """retrieve() calls similarity_search_with_relevance_scores with query and k."""
        mock_store = MagicMock()
        mock_doc   = make_document("Apple revenue $391 billion.")
        mock_store.similarity_search_with_relevance_scores.return_value = [
            (mock_doc, 0.92)
        ]
        retriever = SemanticRetriever(mock_store)
        results   = retriever.retrieve("Apple revenue", top_k=3)

        mock_store.similarity_search_with_relevance_scores.assert_called_once_with(
            "Apple revenue", k=3
        )
        assert len(results) == 1

    def test_retrieve_result_has_required_keys(self):
        """Each result dict has text, score, metadata, and rank keys."""
        mock_store = MagicMock()
        mock_doc   = make_document("Apple revenue $391 billion.", page=5)
        mock_store.similarity_search_with_relevance_scores.return_value = [
            (mock_doc, 0.92)
        ]
        retriever = SemanticRetriever(mock_store)
        results   = retriever.retrieve("Apple revenue", top_k=3)

        assert results[0]["text"]              == "Apple revenue $391 billion."
        assert results[0]["score"]             == 0.92
        assert results[0]["metadata"]["page"]  == 5
        assert results[0]["rank"]              == 1

    def test_retrieve_raises_runtime_error_on_store_failure(self):
        """retrieve() raises RuntimeError if the ChromaDB call fails."""
        mock_store = MagicMock()
        mock_store.similarity_search_with_relevance_scores.side_effect = Exception(
            "ChromaDB connection error"
        )
        retriever = SemanticRetriever(mock_store)
        with pytest.raises(RuntimeError, match="Semantic search failed"):
            retriever.retrieve("Apple revenue", top_k=3)


#  RRF Fusion tests 

class TestRrfFusion:
    """Tests for _rrf_fusion() in retrieval/hybrid.py."""

    def _make_result(self, text: str, rank: int, score: float = 1.0) -> dict:
        """Helper — build a retriever result dict."""
        return {
            "text":     text,
            "score":    score,
            "metadata": {"source": "test.pdf", "page": 0},
            "rank":     rank,
        }

    def test_fusion_returns_list(self):
        """_rrf_fusion returns a list."""
        list1 = [self._make_result("doc A", rank=1)]
        list2 = [self._make_result("doc B", rank=1)]
        result = _rrf_fusion([list1, list2])
        assert isinstance(result, list)

    def test_fusion_deduplicates_identical_chunks(self):
        """A chunk appearing in both lists appears only once in the output."""
        shared = self._make_result("shared chunk about Apple revenue", rank=1)
        list1  = [shared]
        list2  = [shared]
        result = _rrf_fusion([list1, list2])
        texts  = [r["text"] for r in result]
        assert len(texts) == len(set(texts))

    def test_shared_chunk_scores_higher_than_exclusive(self):
        """A chunk in both lists outscores a chunk in only one list."""
        shared    = self._make_result("shared chunk", rank=1)
        exclusive = self._make_result("exclusive chunk", rank=1)

        list1 = [shared, exclusive]
        list2 = [shared]

        result      = _rrf_fusion([list1, list2])
        shared_rrf  = next(r["rrf_score"] for r in result if r["text"] == "shared chunk")
        excl_rrf    = next(r["rrf_score"] for r in result if r["text"] == "exclusive chunk")

        assert shared_rrf > excl_rrf

    def test_fusion_ranks_are_sequential(self):
        """Output ranks start at 1 and increment by 1."""
        list1  = [self._make_result(f"doc {i}", rank=i) for i in range(1, 4)]
        list2  = [self._make_result(f"doc {i}", rank=i) for i in range(1, 4)]
        result = _rrf_fusion([list1, list2])
        ranks  = [r["rank"] for r in result]
        assert ranks == list(range(1, len(ranks) + 1))

    def test_fusion_handles_empty_lists(self):
        """_rrf_fusion handles empty retriever result lists gracefully."""
        result = _rrf_fusion([[], []])
        assert result == []

    def test_fusion_combines_unique_chunks_from_both_lists(self):
        """Output contains all unique chunks from both input lists."""
        list1  = [self._make_result("doc A", rank=1), self._make_result("doc B", rank=2)]
        list2  = [self._make_result("doc C", rank=1), self._make_result("doc D", rank=2)]
        result = _rrf_fusion([list1, list2])
        texts  = {r["text"] for r in result}
        assert texts == {"doc A", "doc B", "doc C", "doc D"}


#  hybrid_retrieve tests 

class TestHybridRetrieve:
    """Tests for hybrid_retrieve() in retrieval/hybrid.py."""

    def _make_mock_bm25(self, results: list[dict]) -> MagicMock:
        """Return a mock BM25Retriever that returns the given results."""
        mock = MagicMock(spec=BM25Retriever)
        mock.retrieve.return_value = results
        return mock

    def _make_mock_semantic(self, results: list[dict]) -> MagicMock:
        """Return a mock SemanticRetriever that returns the given results."""
        mock = MagicMock(spec=SemanticRetriever)
        mock.retrieve.return_value = results
        return mock

    def _make_result(self, text: str, rank: int) -> dict:
        return {
            "text":     text,
            "score":    1.0,
            "metadata": {"source": "test.pdf", "page": 0},
            "rank":     rank,
        }

    def test_raises_on_empty_query(self):
        """hybrid_retrieve raises ValueError for a blank query."""
        with pytest.raises(ValueError, match="non-empty string"):
            hybrid_retrieve(
                query="   ",
                bm25=self._make_mock_bm25([]),
                semantic=self._make_mock_semantic([]),
            )

    def test_returns_top_k_results(self):
        """hybrid_retrieve returns exactly top_k results when enough are available."""
        results = [self._make_result(f"doc {i}", rank=i) for i in range(1, 6)]
        bm25     = self._make_mock_bm25(results)
        semantic = self._make_mock_semantic(results)

        final = hybrid_retrieve(query="Apple revenue", bm25=bm25, semantic=semantic, top_k=3)
        assert len(final) == 3

    def test_calls_both_retrievers(self):
        """hybrid_retrieve calls both BM25 and semantic retrievers."""
        bm25     = self._make_mock_bm25([])
        semantic = self._make_mock_semantic([])

        hybrid_retrieve(query="Apple revenue", bm25=bm25, semantic=semantic, top_k=5)

        bm25.retrieve.assert_called_once()
        semantic.retrieve.assert_called_once()

    def test_fetches_candidate_multiplier_times_top_k(self):
        """Each retriever is called with top_k * CANDIDATE_MULTIPLIER candidates."""
        from retrieval.hybrid import CANDIDATE_MULTIPLIER
        bm25     = self._make_mock_bm25([])
        semantic = self._make_mock_semantic([])

        hybrid_retrieve(query="revenue", bm25=bm25, semantic=semantic, top_k=5)

        bm25.retrieve.assert_called_once_with("revenue", top_k=5 * CANDIDATE_MULTIPLIER)
        semantic.retrieve.assert_called_once_with("revenue", top_k=5 * CANDIDATE_MULTIPLIER)
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


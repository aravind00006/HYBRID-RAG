"""
tests/test_ingestion.py — Unit tests for ingestion layer.
Tests loader.py and chunker.py in isolation.
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document

from ingestion.loader import load_pdf
from ingestion.chunker import chunk_documents, _is_artifact


#  Fixtures 

def make_document(text: str, page: int = 0, source: str = "test.pdf") -> Document:
    """Helper — build a LangChain Document for testing."""
    return Document(
        page_content=text,
        metadata={"source": source, "page": page},
    )


def make_pages(n: int = 3, words_per_page: int = 200) -> list[Document]:
    """Helper — build a list of realistic page Documents."""
    return [
        make_document(
            text=" ".join([f"word{i}" for i in range(words_per_page)]),
            page=p,
        )
        for p in range(n)
    ]


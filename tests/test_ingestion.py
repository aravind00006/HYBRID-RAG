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


#  loader.py tests ─

class TestLoadPdf:
    """Tests for load_pdf() in ingestion/loader.py."""

    def test_raises_file_not_found(self, tmp_path):
        """load_pdf raises FileNotFoundError for a non-existent path."""
        with pytest.raises(FileNotFoundError, match="PDF not found"):
            load_pdf(tmp_path / "nonexistent.pdf")

    def test_raises_value_error_for_non_pdf(self, tmp_path):
        """load_pdf raises ValueError for files that are not .pdf."""
        txt_file = tmp_path / "document.txt"
        txt_file.write_text("This is not a PDF.")
        with pytest.raises(ValueError, match="Expected a .pdf file"):
            load_pdf(txt_file)

    def test_normalises_source_metadata(self, tmp_path):
        """load_pdf sets metadata source to filename only, not full path."""
        pdf_path = tmp_path / "test.pdf"

        # Mock PyPDFLoader so we don't need a real PDF file.
        mock_doc = make_document("Sample content.", page=0, source=str(pdf_path))
        with patch("ingestion.loader.PyPDFLoader") as mock_loader:
            mock_loader.return_value.load.return_value = [mock_doc]
            # Give the file a real path so exists() passes.
            pdf_path.write_bytes(b"%PDF-1.4")

            docs = load_pdf(pdf_path)

        # Source should be just the filename, not the full system path.
        assert docs[0].metadata["source"] == "test.pdf"
        assert "/" not in docs[0].metadata["source"]
        assert "\\" not in docs[0].metadata["source"]

    def test_returns_list_of_documents(self, tmp_path):
        """load_pdf returns a non-empty list of Documents."""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4")

        mock_docs = [make_document(f"Page {i} content.", page=i) for i in range(3)]
        with patch("ingestion.loader.PyPDFLoader") as mock_loader:
            mock_loader.return_value.load.return_value = mock_docs
            docs = load_pdf(pdf_path)

        assert isinstance(docs, list)
        assert len(docs) == 3


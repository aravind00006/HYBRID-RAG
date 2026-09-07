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


#  _is_artifact tests 

class TestIsArtifact:
    """Tests for the _is_artifact() helper in ingestion/chunker.py."""

    def test_real_content_is_not_artifact(self):
        """A real financial sentence is not an artifact."""
        text = (
            "Apple's total net sales for fiscal year 2024 were $391.035 billion, "
            "compared to $383.285 billion in fiscal year 2023."
        )
        assert _is_artifact(text) is False

    def test_too_short_is_artifact(self):
        """A chunk with fewer than 40 non-whitespace characters is an artifact."""
        assert _is_artifact("Short.") is True
        assert _is_artifact("   ") is True
        assert _is_artifact("") is True

    def test_timestamp_plus_url_is_artifact(self):
        """A timestamp and URL pattern (EDGAR browser print) is an artifact."""
        text = "1/15/2024, 10:32 AM https://www.sec.gov/Archives/edgar/data/320193"
        assert _is_artifact(text) is True

    def test_mostly_url_is_artifact(self):
        """A chunk composed mostly of URLs is an artifact."""
        text = "See https://www.sec.gov/Archives/edgar/data/320193/000032019324000123/aapl-20240930.htm for details."
        # URL is > 30% of the total text length.
        assert _is_artifact(text) is True

    def test_content_with_url_mention_is_not_artifact(self):
        """Content that briefly mentions a URL but is mostly text is not an artifact."""
        text = (
            "For more information about Apple's fiscal year results, refer to the "
            "company's investor relations website at https://investor.apple.com. "
            "Total revenue was $391 billion for the year ended September 28, 2024."
        )
        assert _is_artifact(text) is False


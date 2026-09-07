"""
Keyword search using BM25Okapi ranking.
Single responsibility: exact term matching only.
"""

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from logger import get_logger

logger = get_logger(__name__)


class BM25Retriever:
    """
    In-memory BM25 retriever over a fixed corpus of Documents.
    Build once at startup via BM25Retriever(chunks), then reuse the
    same instance across all requests via app.state.bm25.
    """

    def __init__(self, documents: list[Document]) -> None:
        """
        Build the BM25 index from a list of Documents.

        """
        if not documents:
            raise ValueError("Cannot build BM25 index on an empty document list.")

        self._documents = documents

        logger.info("Building BM25 index over %d documents.", len(documents))

        # Tokenise by whitespace — BM25 operates on word tokens.
        tokenized = [doc.page_content.lower().split() for doc in documents]
        self._bm25 = BM25Okapi(tokenized)

        logger.info("BM25 index built successfully.")

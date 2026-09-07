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

    def retrieve(self, query: str, top_k: int) -> list[dict]:
        """
        Return the top-k documents ranked by BM25 score for the query.

        """
        if not query.strip():
            logger.warning("BM25 received an empty query — returning no results.")
            return []

        tokens = query.lower().split()
        scores = self._bm25.get_scores(tokens)

        # Pair each doc with its score and sort highest first.
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

        results = []
        for rank, (idx, score) in enumerate(ranked[:top_k], start=1):
            if score <= 0.0:
                break  # No more documents share tokens with the query.
            results.append({
                "text":     self._documents[idx].page_content,
                "score":    round(float(score), 4),
                "metadata": self._documents[idx].metadata,
                "rank":     rank,
            })

        logger.info(
            "BM25 retrieved %d results for query: '%s'.",
            len(results),
            query[:60],
        )
        return results
"""
Vector similarity search using ChromaDB.
Single responsibility: semantic search only.

"""

from langchain_chroma import Chroma

from logger import get_logger

logger = get_logger(__name__)


class SemanticRetriever:
    """
    Converts the query to an embedding vector via the same model used during ingestion, 
    then finds the most similar chunk vectors in ChromaDB using cosine similarity.

    """

    def __init__(self, store: Chroma) -> None:
        """
        Initialise the retriever with an existing Chroma vector store.
        """
        if store is None:
            raise ValueError("Chroma store cannot be None.")

        self._store = store
        logger.info("SemanticRetriever initialised with ChromaDB store.")

    def retrieve(self, query: str, top_k: int) -> list[dict]:
        """
        Embeds the query using the same OpenAI model used during ingestion,
        then performs cosine similarity search against the stored vectors.
        """
        if not query.strip():
            logger.warning("SemanticRetriever received an empty query — returning no results.")
            return []

        logger.info(
            "Running semantic search for query: '%s' (top_k=%d).",
            query[:60],
            top_k,
        )

        try:
            raw = self._store.similarity_search_with_relevance_scores(
                query, k=top_k
            )
        except Exception as exc:
            raise RuntimeError(f"Semantic search failed: {exc}") from exc

        results = [
            {
                "text":     doc.page_content,
                "score":    round(score, 4),
                "metadata": doc.metadata,
                "rank":     rank,
            }
            for rank, (doc, score) in enumerate(raw, start=1)
        ]

        logger.info(
            "Semantic search retrieved %d results for query: '%s'.",
            len(results),
            query[:60],
        )
        return results
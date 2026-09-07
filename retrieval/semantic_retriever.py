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

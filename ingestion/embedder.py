"""
Embed chunks and store/load them in ChromaDB.
Single responsibility: vector store management only.
"""

from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from config import get_settings
from logger import get_logger

logger = get_logger(__name__)


def _get_embedding_model() -> OpenAIEmbeddings:
    """
    Build and return the OpenAI embedding model from settings.

    """
    settings = get_settings()
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        openai_api_key=settings.openai_api_key,
    )


def embed_and_store(
    chunks: list[Document],
    collection_name: str | None = None,
    chroma_path: str | None = None,
) -> Chroma:
    """
    Embed chunks with OpenAI and persist them in a ChromaDB collection.

    """
    if not chunks:
        raise ValueError("Cannot embed an empty chunks list.")

    settings = get_settings()
    path = chroma_path or settings.chroma_path

    # Derive collection name from directory name to avoid collisions.
    name = collection_name or Path(path).name or settings.collection_name

    logger.info(
        "Embedding %d chunks into collection '%s' at '%s'.",
        len(chunks),
        name,
        path,
    )

    Path(path).mkdir(parents=True, exist_ok=True)

    try:
        store = Chroma.from_documents(
            documents=chunks,
            embedding=_get_embedding_model(),
            collection_name=name,
            persist_directory=path,
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to embed and store chunks: {exc}") from exc

    logger.info(
        "Embedding complete — %d chunks stored in '%s'.",
        len(chunks),
        name,
    )
    return store


def load_store(
    collection_name: str | None = None,
    chroma_path: str | None = None,
) -> Chroma:
    """
    Load an existing ChromaDB collection from disk without re-embedding.

    """
    settings = get_settings()
    path = chroma_path or settings.chroma_path

    # Derive collection name from directory name — must match embed_and_store().
    name = collection_name or Path(path).name or settings.collection_name

    if not Path(path).exists():
        raise FileNotFoundError(
            f"ChromaDB not found at '{path}'. "
            "Run embed_and_store() first to build the index."
        )

    logger.info("Loading ChromaDB collection '%s' from '%s'.", name, path)

    try:
        store = Chroma(
            collection_name=name,
            embedding_function=_get_embedding_model(),
            persist_directory=path,
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to load ChromaDB collection: {exc}") from exc

    logger.info("ChromaDB collection '%s' loaded successfully.", name)
    return store
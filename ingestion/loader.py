"""
Load a PDF and return one Document per page. single responsibility: file I/O only.
No chunking, no embedding — just reading the PDF off disk.
"""

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from logger import get_logger

logger = get_logger(__name__)


def load_pdf(path: str | Path) -> list[Document]:
    """
    Load a PDF file and return one LangChain Document per page.
    """
    path = Path(path)
    logger.info("Loading PDF: %s", path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, got '{path.suffix}'.")

    try:
        docs = PyPDFLoader(str(path)).load()
    except Exception as exc:
        raise RuntimeError(f"Failed to read PDF '{path}': {exc}") from exc

    # Normalise metadata — store just the filename, not the full system path.
    for doc in docs:
        doc.metadata["source"] = path.name

    logger.info("Loaded %d pages from '%s'.", len(docs), path.name)
    return docs
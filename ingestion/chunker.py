"""
Split Documents into smaller overlapping chunks. text splitting and artifact filtering only.
No PDF loading, no embedding — just chunking.
"""

import re

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import get_settings
from logger import get_logger

logger = get_logger(__name__)

# Split hierarchy — tries paragraph breaks first, then sentence, then word.
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

# Minimum non-whitespace characters for a chunk to be considered real content.
MIN_CONTENT_LENGTH = 40

# Regex to detect browser-print artifacts (URL + timestamp lines from SEC EDGAR).
_ARTIFACT_RE = re.compile(
    r"\d{1,2}/\d{1,2}/\d{2,4},\s*\d{1,2}:\d{2}\s*(AM|PM)",
    re.IGNORECASE,
)


def _is_artifact(text: str) -> bool:
    """
    Return True if a chunk is a browser-print artifact, not real content.

    """
    stripped = text.strip()

    if len(stripped.replace(" ", "").replace("\n", "")) < MIN_CONTENT_LENGTH:
        return True

    urls = re.findall(r"https?://\S+", stripped)

    if _ARTIFACT_RE.search(stripped) and urls:
        return True

    if urls and sum(len(u) for u in urls) / max(len(stripped), 1) >= 0.30:
        return True

    return False


def chunk_documents(pages: list[Document]) -> list[Document]:
    """
    Split a list of page-level Documents into overlapping chunks.
    """
    if not pages:
        raise ValueError("Cannot chunk an empty document list.")

    settings = get_settings()
    logger.info(
        "Chunking %d pages — chunk_size=%d, chunk_overlap=%d.",
        len(pages),
        settings.chunk_size,
        settings.chunk_overlap,
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=SEPARATORS,
        length_function=len,
        is_separator_regex=False,
    )

    chunks = splitter.split_documents(pages)
    total_raw = len(chunks)

    # Filter empty and artifact chunks.
    chunks = [c for c in chunks if c.page_content.strip()]
    chunks = [c for c in chunks if not _is_artifact(c.page_content)]

    logger.info(
        "Chunking complete — %d raw chunks, %d kept after filtering (%d removed).",
        total_raw,
        len(chunks),
        total_raw - len(chunks),
    )

    if not chunks:
        raise ValueError(
            "Chunking produced zero valid chunks. "
            "The PDF may be scanned (image-only), empty, or entirely artifacts. "
            "Try a text-based PDF."
        )

    return chunks
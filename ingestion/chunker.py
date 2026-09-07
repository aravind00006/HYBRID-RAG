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


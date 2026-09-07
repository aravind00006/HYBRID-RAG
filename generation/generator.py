"""
Generate a grounded answer using GPT-4o-mini. Single responsibility: LLM call only.
Receives chunks from retrieval, prompt from prompts.py, returns structured answer.
"""

from dataclasses import dataclass

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from config import get_settings
from generation.prompts import SYSTEM_PROMPT, build_user_message
from logger import get_logger

logger = get_logger(__name__)


@dataclass
class RAGResponse:
    """
    Structured output from the generation step.

    """

    answer: str
    sources: list[dict]
    model: str
    tokens_used: int


def _extract_sources(chunks: list[dict]) -> list[dict]:
    """
    Deduplicate and extract citation metadata from retrieved chunks.

    """
    seen: set[tuple] = set()
    sources: list[dict] = []

    for chunk in chunks:
        meta   = chunk.get("metadata", {})
        source = meta.get("source", "unknown")
        page   = meta.get("page", 0) + 1  # Convert 0-indexed to 1-indexed.
        key    = (source, page)

        if key in seen:
            continue
        seen.add(key)

        sources.append({
            "source":  source,
            "page":    page,
            "preview": chunk.get("text", "")[:150],
        })

    return sources


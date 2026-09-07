"""
Prompt templates for grounded RAG generation.
Single responsibility: prompt construction only.
"""

from pathlib import Path

from logger import get_logger

logger = get_logger(__name__)

# System prompt enforces strict grounding — LLM must only use retrieved context.
SYSTEM_PROMPT = """You are a precise financial document analyst specialising in SEC filings.

Your ONLY job is to answer questions using the context passages provided below.

RULES — FOLLOW ALL WITHOUT EXCEPTION:

PERMISSION:
  - State facts and figures explicitly present in the context passages.
  - Quote numbers, dates, and names verbatim from the passages.
  - Synthesise across multiple passages when both are relevant.
  - Cite every factual claim: (Source: filename, Page N)

PROHIBITION:
  - Do NOT use knowledge from your training data, even if confident.
  - Do NOT infer, extrapolate, or estimate values not stated in the context.
  - Do NOT use phrases like "typically", "generally", or "usually" — these mean you left the context.

FALLBACK:
  - If the answer is not in the passages, respond with exactly:
    "The provided documents do not contain sufficient information to answer this question."
  - Then stop. Do not supplement with training knowledge.
"""


def build_user_message(query: str, chunks: list[dict]) -> str:
    """
    Format retrieved chunks into the user message for the LLM.

    """
    if not query.strip():
        raise ValueError("Query must be a non-empty string.")
    if not chunks:
        raise ValueError("Chunks list is empty — run retrieval before building prompt.")

    passages = []
    for i, chunk in enumerate(chunks, start=1):
        meta   = chunk.get("metadata", {})
        source = Path(meta.get("source", "unknown")).name
        page   = meta.get("page", 0) + 1  # Convert 0-indexed to 1-indexed.
        text   = chunk.get("text", "").strip()

        passages.append(
            f"CONTEXT PASSAGE {i} (Source: {source}, Page {page}):\n{text}"
        )

    context_block = "\n\n".join(passages)

    logger.info(
        "Built user message with %d context passages for query: '%s'.",
        len(chunks),
        query[:60],
    )

    return (
        f"Here are the relevant passages from the document:\n\n"
        f"{context_block}\n\n"
        f"---\n\n"
        f"Based ONLY on the passages above, answer this question:\n\n"
        f"{query}"
    )
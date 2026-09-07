"""
Fuse BM25 and semantic results using Reciprocal Rank Fusion. Single responsibility: rank fusion only.
Receives ranked lists from both retrievers, returns one unified ranking.

"""

from retrieval.bm25_retriever import BM25Retriever
from retrieval.semantic_retriever import SemanticRetriever
from config import get_settings
from logger import get_logger

logger = get_logger(__name__)

# RRF damping constant — higher k reduces the impact of top ranks.
# k=60 is the standard value from the original RRF paper.
RRF_K = 60

# Fetch more candidates than top_k so RRF has room to re-rank effectively.
CANDIDATE_MULTIPLIER = 4


def _rrf_fusion(
    result_lists: list[list[dict]],
    k: int = RRF_K,
) -> list[dict]:
    """
    Merge multiple ranked lists using Reciprocal Rank Fusion (RRF).

    """
    rrf_scores: dict[str, float] = {}
    chunk_data: dict[str, dict] = {}

    for result_list in result_lists:
        for item in result_list:
            text = item["text"]
            rrf_scores[text] = rrf_scores.get(text, 0.0) + 1.0 / (k + item["rank"])
            # Keep metadata from the first retriever that saw this chunk.
            if text not in chunk_data:
                chunk_data[text] = item

    sorted_texts = sorted(rrf_scores, key=lambda t: rrf_scores[t], reverse=True)

    fused = [
        {
            **chunk_data[text],
            "rrf_score": round(rrf_scores[text], 6),
            "rank": rank,
        }
        for rank, text in enumerate(sorted_texts, start=1)
    ]

    logger.info(
        "RRF fusion complete — %d unique chunks from %d retriever lists.",
        len(fused),
        len(result_lists),
    )
    return fused



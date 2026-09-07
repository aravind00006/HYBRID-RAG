"""
evaluation/ragas_eval.py — Evaluate the RAG pipeline using RAGAS metrics.
Single responsibility: quality measurement only.
"""

from dataclasses import dataclass

from datasets import Dataset
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from config import get_settings
from logger import get_logger

logger = get_logger(__name__)

# All four metrics evaluated in every run.
METRICS = [faithfulness, answer_relevancy, context_precision, context_recall]

# Score thresholds — consistent across summary output and status assignment.
PASS_THRESHOLD = 0.80
WARN_THRESHOLD = 0.60

# Progress bar width for the summary display.
BAR_WIDTH = 20


@dataclass
class EvalResult:
    """
    Structured output from a RAGAS evaluation run.

    """

    scores: dict[str, float]
    status: str

    def summary(self) -> str:
        """
        Return a human-readable score summary with ASCII progress bars.
        """
        lines = [
            f"RAGAS Evaluation -- Status: {self.status.upper()}",
            "-" * 55,
        ]
        for metric, score in self.scores.items():
            if score >= PASS_THRESHOLD:
                flag = "PASS"
            elif score >= WARN_THRESHOLD:
                flag = "WARN"
            else:
                flag = "FAIL"

            filled  = "#" * round(score * BAR_WIDTH)
            bar     = "[" + filled.ljust(BAR_WIDTH) + "]"
            lines.append(
                f"  {metric:<25} {score:.4f}   {bar}   {flag}"
            )
        return "\n".join(lines)

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


def _extract_scores(results) -> dict[str, float]:
    """
    Extract per-metric mean scores from a RAGAS EvaluationResult.

    """
    df = results.to_pandas()
    scores: dict[str, float] = {}

    for metric in METRICS:
        if metric.name in df.columns:
            valid = df[metric.name].dropna()
            scores[metric.name] = (
                round(float(valid.mean()), 4) if len(valid) > 0 else 0.0
            )
        else:
            logger.warning("Metric '%s' not found in RAGAS results.", metric.name)
            scores[metric.name] = 0.0

    return scores


def run_evaluation(samples: list[dict]) -> EvalResult:
    """
    Run RAGAS evaluation over a list of RAG samples.

    """
    if not samples:
        raise ValueError("samples list is empty -- nothing to evaluate.")

    required = {"question", "answer", "contexts", "ground_truth"}
    for i, sample in enumerate(samples):
        missing = required - set(sample.keys())
        if missing:
            raise ValueError(
                f"Sample {i} is missing required keys: {missing}. "
                f"Each sample must have: {required}"
            )

    logger.info("Starting RAGAS evaluation over %d samples.", len(samples))

    settings  = get_settings()
    dataset   = Dataset.from_list(samples)

    judge_llm = ChatOpenAI(
        model=settings.llm_model,
        temperature=0.0,
        openai_api_key=settings.openai_api_key,
    )
    judge_emb = OpenAIEmbeddings(
        model=settings.embedding_model,
        openai_api_key=settings.openai_api_key,
    )

    try:
        results = evaluate(
            dataset=dataset,
            metrics=METRICS,
            llm=judge_llm,
            embeddings=judge_emb,
        )
    except Exception as exc:
        raise RuntimeError(f"RAGAS evaluation failed: {exc}") from exc

    scores = _extract_scores(results)

    faith  = scores.get("faithfulness", 0.0)
    if faith >= PASS_THRESHOLD:
        status = "pass"
    elif faith >= WARN_THRESHOLD:
        status = "warn"
    else:
        status = "fail"

    logger.info(
        "RAGAS evaluation complete — status: %s, faithfulness: %.4f.",
        status.upper(),
        faith,
    )

    return EvalResult(scores=scores, status=status)


#  Sample evaluation runner 

SAMPLE_QUESTIONS = [
    {
        "question": "What was Apple's total net sales in fiscal year 2024?",
        "answer":   "Apple's total net sales for fiscal year 2024 were $391.035 billion.",
        "contexts": [
            "Total net sales $391,035 $383,285 $394,328 for fiscal years 2024, 2023, 2022."
        ],
        "ground_truth": "Apple's total net sales in fiscal year 2024 were $391.035 billion.",
    },
    {
        "question": "What was Apple's net income in fiscal year 2024?",
        "answer":   "Apple's net income for fiscal year 2024 was $93.736 billion.",
        "contexts": [
            "Net income $93,736 $96,995 $99,803 for fiscal years 2024, 2023, 2022."
        ],
        "ground_truth": "Apple's net income in fiscal year 2024 was $93.736 billion.",
    },
]


if __name__ == "__main__":
    result = run_evaluation(SAMPLE_QUESTIONS)
    print(result.summary())
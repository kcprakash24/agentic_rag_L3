# import logging

# logger = logging.getLogger(__name__)


# def score_live(
#     trace_id: str,
#     question: str,
#     answer: str,
#     contexts: list[str],
# ) -> dict:
#     """
#     Stub — full RAGAS implementation in Step 7.
#     Returns empty dict for now.
#     """
#     logger.info(f"score_live called for trace {trace_id[:8]} — stub, skipping")
#     return {}

from __future__ import annotations

import logging
from typing import Any

from langfuse import get_client
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy

logger = logging.getLogger(__name__)

# RAGAS expects a Dataset-like object. It commonly works with HuggingFace Datasets.
# If you don't already have it installed, run in your env:
#   pip install datasets
from datasets import Dataset  # type: ignore

from agentic_rag.observability.langfuse_client import _set_langfuse_env


def _to_ragas_dataset(*, question: str, answer: str, contexts: list[str], ground_truth: str | None = None) -> Dataset:
    row: dict[str, Any] = {
        "question": question,
        "answer": answer,
        "contexts": [contexts],
    }
    if ground_truth is not None:
        # Some metrics expect "ground_truth" or "ground_truths" depending on version.
        # For ragas>=0.4.x, ground-truth-based metrics often accept "ground_truth".
        row["ground_truth"] = ground_truth
    return Dataset.from_dict(row)


def _log_score(*, trace_id: str, name: str, value: float, comment: str | None = None) -> None:
    _set_langfuse_env()
    langfuse = get_client()

    langfuse.create_score(
        trace_id=trace_id,
        name=name,
        value=float(value),
        data_type="NUMERIC",
        comment=comment,
    )
    langfuse.flush()


def score_live(
    trace_id: str,
    question: str,
    answer: str,
    contexts: list[str],
) -> dict:
    """
    Runs RAGAS metrics for live traffic (no ground truth):
      - faithfulness (answer vs contexts)
      - answer_relevancy (answer vs question)
    Logs scores to Langfuse (trace-level).
    """
    if not contexts:
        logger.info("score_live: no contexts; skipping")
        return {}

    ds = _to_ragas_dataset(question=question, answer=answer, contexts=contexts)

    # Evaluate with RAGAS
    result = evaluate(ds, metrics=[faithfulness, answer_relevancy])
    scores = result.to_pandas().iloc[0].to_dict()

    f = float(scores.get("faithfulness")) if scores.get("faithfulness") is not None else None
    r = float(scores.get("answer_relevancy")) if scores.get("answer_relevancy") is not None else None

    if f is not None:
        _log_score(trace_id=trace_id, name="ragas_faithfulness", value=f)
    if r is not None:
        _log_score(trace_id=trace_id, name="ragas_answer_relevancy", value=r)

    return {
        "ragas_faithfulness": f,
        "ragas_answer_relevancy": r,
    }
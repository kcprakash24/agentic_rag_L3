from __future__ import annotations

from typing import Any
from langfuse import get_client

from agentic_rag.agent.graph import ask
from agentic_rag.config import get_settings
from agentic_rag.evaluation.scorer import score_live
from agentic_rag.observability.langfuse_client import _set_langfuse_env


def _extract(item: Any) -> tuple[str, str]:
    # Dataset item shape varies slightly by SDK versions.
    # Common: item.input, item.expected_output
    inp = getattr(item, "input", None) or {}
    exp = getattr(item, "expected_output", None) or {}

    q = inp.get("question") or inp.get("text") or ""
    gt = exp.get("answer") or exp.get("text") or ""
    if not q:
        raise ValueError("Dataset item missing input.question")
    if not gt:
        raise ValueError("Dataset item missing expected_output.answer")
    return q, gt


def run_golden_dataset(
    *,
    dataset_name: str | None = None,
    user_id: str = "eval_bot",
    session_id: str = "golden_eval_session",
    limit: int | None = None,
) -> list[dict]:
    _set_langfuse_env()
    langfuse = get_client()

    settings = get_settings()
    dataset_name = dataset_name or settings.golden_dataset_name

    ds = langfuse.get_dataset(name=dataset_name)
    items = list(getattr(ds, "items", []))
    if limit is not None:
        items = items[:limit]

    results: list[dict] = []

    for item in items:
        question, expected = _extract(item)

        # Run your real system (creates a Langfuse trace via your callbacks/spans)
        out = ask(question=question, user_id=user_id, session_id=session_id)

        # Score the produced answer vs its retrieved contexts
        contexts = [c.get("content", "") for c in (out.get("sources") or [])]
        # NOTE: your `sources` are previews, not full contexts. Better:
        # use reranked chunk contents for eval; to do that, return reranked_chunks in ask()
        # (or add them to the final return dict).
        #
        # For now, we score on the context text that was actually used:
        # easiest is to tweak ask() to return final_state["reranked_chunks"] and use that here.

        live_scores = score_live(
            trace_id=out["trace_id"],
            question=question,
            answer=out["answer"],
            contexts=contexts if contexts else [],
        )

        results.append(
            {
                "question": question,
                "expected_answer": expected,
                "answer": out["answer"],
                "trace_id": out["trace_id"],
                **live_scores,
            }
        )

    return results
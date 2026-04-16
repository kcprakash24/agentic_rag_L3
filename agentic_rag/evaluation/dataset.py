from __future__ import annotations

from typing import Any
from langfuse import get_client
from agentic_rag.config import get_settings
from agentic_rag.observability.langfuse_client import _set_langfuse_env  # reuse your env setter


def get_or_create_dataset(*, dataset_name: str, description: str | None = None) -> None:
    _set_langfuse_env()
    langfuse = get_client()

    # create_dataset is idempotent-ish in practice; if your SDK errors on duplicates,
    # wrap in try/except and ignore "already exists".
    try:
        langfuse.create_dataset(
            name=dataset_name,
            description=description or "Golden dataset for RAG evaluation",
        )
    except Exception:
        # dataset likely already exists
        pass


def upsert_dataset_item(
    *,
    dataset_name: str,
    question: str,
    expected_answer: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    _set_langfuse_env()
    langfuse = get_client()

    langfuse.create_dataset_item(
        dataset_name=dataset_name,
        input={"question": question},
        expected_output={"answer": expected_answer},
        metadata=metadata or {},
    )
    langfuse.flush()


def default_dataset_name() -> str:
    return get_settings().golden_dataset_name
import logging
from functools import lru_cache
from sentence_transformers import CrossEncoder
from agentic_rag.config import get_settings

logger = logging.getLogger(__name__)

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@lru_cache
def get_cross_encoder() -> CrossEncoder:
    """
    Load and cache the cross-encoder model.
    Downloaded once (~80MB), cached in memory after first call.
    """
    logger.info(f"Loading cross-encoder: {MODEL_NAME}")
    return CrossEncoder(MODEL_NAME)


def rerank(
    question: str,
    chunks: list[dict],
    top_n: int | None = None,
) -> list[dict]:
    """
    Stage 2 retrieval — cross-encoder reranking.
    Takes bi-encoder candidates and reranks by relevance.

    Args:
        question: Original user question
        chunks: Candidates from retrieve() — list of chunk dicts
        top_n: Final chunks to return (defaults to config rerank_top_n)

    Returns:
        Top-n chunks sorted by cross-encoder score (highest first)
        Each chunk gets a 'rerank_score' field added
    """
    settings = get_settings()
    n = top_n or settings.rerank_top_n

    if not chunks:
        return []

    # Limit top_n to available chunks
    n = min(n, len(chunks))

    cross_encoder = get_cross_encoder()

    # Build (question, chunk_text) pairs for scoring
    pairs = [(question, chunk["content"]) for chunk in chunks]

    # Score all pairs — cross-encoder sees query+doc jointly
    scores = cross_encoder.predict(pairs)

    # Attach scores to chunks
    scored_chunks = []
    for chunk, score in zip(chunks, scores):
        scored_chunk = {
            **chunk,
            "rerank_score": float(score),
        }
        scored_chunks.append(scored_chunk)

    # Sort by rerank score descending
    scored_chunks.sort(key=lambda x: x["rerank_score"], reverse=True)

    top_chunks = scored_chunks[:n]

    logger.info(
        f"Reranked {len(chunks)} → {len(top_chunks)} chunks | "
        f"top score: {top_chunks[0]['rerank_score']:.4f}"
    )

    return top_chunks


def retrieve_and_rerank(
    question: str,
    collection: str | None = None,
    retrieval_k: int | None = None,
    rerank_n: int | None = None,
) -> list[dict]:
    """
    Full two-stage retrieval pipeline.
    Convenience function combining retrieve() + rerank().

    Args:
        question: User question
        collection: Optional collection filter
        retrieval_k: Bi-encoder candidates (default: config retrieval_top_k)
        rerank_n: Final chunks after reranking (default: config rerank_top_n)

    Returns:
        Top-n reranked chunks
    """
    from agentic_rag.retrieval.retriever import retrieve

    # Stage 1: bi-encoder
    candidates = retrieve(question, collection=collection, k=retrieval_k)

    if not candidates:
        return []

    # Stage 2: cross-encoder
    reranked = rerank(question, candidates, top_n=rerank_n)

    return reranked
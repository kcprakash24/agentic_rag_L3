from agentic_rag.embeddings.embedder import embed_query
from agentic_rag.vectorstore.pgvector_store import similarity_search
from agentic_rag.config import get_settings


def retrieve(
    question: str,
    collection: str | None = None,
    k: int | None = None,
) -> list[dict]:
    """
    Stage 1 retrieval — bi-encoder similarity search.
    Returns top-k candidates for reranking.

    Args:
        question: User question
        collection: Optional collection filter
        k: Number of candidates (defaults to config retrieval_top_k)

    Returns:
        List of chunk dicts with similarity scores
    """
    settings = get_settings()
    top_k = k or settings.retrieval_top_k

    query_embedding = embed_query(question)
    results = similarity_search(
        query_embedding=query_embedding,
        k=top_k,
        collection=collection,
    )

    return results
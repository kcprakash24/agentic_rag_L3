from dataclasses import dataclass
from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter
from agentic_rag.ingestion.loader import ParsedDocument


@dataclass
class DocumentChunk:
    """A single chunk ready for embedding."""
    chunk_id: str
    text: str
    metadata: dict


def chunk_document(
    parsed_doc: ParsedDocument,
    max_tokens: int = 512,
    collection: str = "general",
) -> list[DocumentChunk]:
    """
    Chunk a parsed document using Docling's HybridChunker.

    HybridChunker respects:
    - Semantic boundaries (never splits mid-sentence or mid-table)
    - Token limits (max_tokens per chunk)
    - Document structure (tables, headings, lists stay intact)

    Args:
        parsed_doc: Output from load_pdf()
        max_tokens: Maximum tokens per chunk (~512 = ~400 words)
        collection: Which collection this chunk belongs to

    Returns:
        List of DocumentChunk objects
    """
    chunker = HybridChunker(max_tokens=max_tokens)
    raw_chunks = list(chunker.chunk(parsed_doc.docling_document))

    chunks = []
    for i, chunk in enumerate(raw_chunks):
        # Extract text from chunk
        text = chunk.text.strip()

        if not text:
            continue

        chunk_id = f"{parsed_doc.file_name}_chunk_{i:04d}"

        doc_chunk = DocumentChunk(
            chunk_id=chunk_id,
            text=text,
            metadata={
                **parsed_doc.metadata,
                "chunk_index": i,
                "chunk_id": chunk_id,
                "total_chunks": len(raw_chunks),
                "collection": collection,
            }
        )
        chunks.append(doc_chunk)

    print(f"  '{parsed_doc.file_name}' → {len(chunks)} chunks "
          f"(avg {sum(len(c.text) for c in chunks) // max(len(chunks), 1)} chars each)")

    return chunks


def chunk_documents(
    parsed_docs: list[ParsedDocument],
    max_tokens: int = 512,
    collection: str = "general",
) -> list[DocumentChunk]:
    """Chunk multiple documents."""
    all_chunks = []
    for doc in parsed_docs:
        all_chunks.extend(chunk_document(doc, max_tokens, collection))
    return all_chunks
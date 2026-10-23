"""Chunking.

Why chunk at all? Three independent reasons, and it helps to keep them separate:

1. The embedding model has a hard input limit (2048 tokens for gemini-embedding-001).
2. An embedding is a single fixed-length vector. Ask it to represent 40 pages and it
   represents nothing in particular - the signal averages out into mush.
3. The LLM's context window is finite and expensive, so we want to send only the
   passages that matter.

Recursive character splitting tries the biggest natural boundary first (paragraph),
falling back to sentence, then word, then character. That keeps semantically related
text together far more often than blind fixed-size slicing.

Overlap exists so a fact that straddles a chunk boundary survives in at least one chunk.

Phase 4 adds two knobs, both measured by `cli eval` rather than guessed at:

- `CHUNK_STRATEGY=fixed` slices blindly at `chunk_size` - the baseline that shows
  what recursive splitting is actually buying.
- `PARENT_CHUNK_SIZE` turns on parent-document retrieval: small chunks are embedded
  (precise matching) but each carries its larger parent in metadata, and the parent
  is what the model reads (enough context to answer).
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings


def build_splitter(
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    strategy: str | None = None,
) -> RecursiveCharacterTextSplitter:
    settings = get_settings()
    fixed = (strategy or settings.chunk_strategy) == "fixed"
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or settings.chunk_size,
        chunk_overlap=chunk_overlap if chunk_overlap is not None else settings.chunk_overlap,
        # A single empty separator is plain fixed-size slicing.
        separators=[""] if fixed else ["\n\n", "\n", ". ", " ", ""],
        length_function=len,
        add_start_index=True,
    )


def chunk_documents(
    docs: list[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """Split documents and stamp each chunk with a stable, deterministic id.

    The id is `{source}::{page}::{start_index}`. Deterministic ids are what make
    re-ingestion an upsert instead of a duplicate-creating append.
    """
    splitter = build_splitter(chunk_size, chunk_overlap)
    parent_size = get_settings().parent_chunk_size
    if parent_size > splitter._chunk_size:
        chunks = []
        for parent in build_splitter(parent_size, chunk_overlap).split_documents(docs):
            offset = parent.metadata.get("start_index", 0)
            for child in splitter.split_documents([parent]):
                # Child offsets are relative to the parent; ids need absolute ones.
                child.metadata["start_index"] = offset + child.metadata["start_index"]
                child.metadata["parent"] = parent.page_content
                chunks.append(child)
    else:
        chunks = splitter.split_documents(docs)

    for chunk in chunks:
        meta = chunk.metadata
        meta["chunk_id"] = (
            f"{meta.get('source', 'unknown')}"
            f"::{meta.get('page', 1)}"
            f"::{meta.get('start_index', 0)}"
        )
    return chunks


def chunk_ids(chunks: list[Document]) -> list[str]:
    return [c.metadata["chunk_id"] for c in chunks]

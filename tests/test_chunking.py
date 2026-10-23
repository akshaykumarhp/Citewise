"""Tests for the pieces that don't need an API key.

Deliberately no LLM here. Chunking and grading are pure functions and should be
testable without spending a single token - that separation is worth keeping as the
project grows.
"""

from __future__ import annotations

from langchain_core.documents import Document

from app.chunking import chunk_documents, chunk_ids
from app.retrieval import ScoredChunk, format_context, grade


def _doc(text: str, source: str = "test.pdf", page: int = 1) -> Document:
    return Document(page_content=text, metadata={"source": source, "page": page})


def test_chunks_respect_size_limit():
    long_text = " ".join(f"word{i}" for i in range(2000))
    chunks = chunk_documents([_doc(long_text)], chunk_size=200, chunk_overlap=20)
    assert len(chunks) > 1
    assert all(len(c.page_content) <= 260 for c in chunks)  # slack for word boundaries


def test_chunks_carry_source_metadata():
    chunks = chunk_documents([_doc("hello world", source="policy.pdf", page=7)])
    assert chunks[0].metadata["source"] == "policy.pdf"
    assert chunks[0].metadata["page"] == 7


def test_chunk_ids_are_deterministic_and_unique():
    text = " ".join(f"word{i}" for i in range(500))
    first = chunk_ids(chunk_documents([_doc(text)], chunk_size=200, chunk_overlap=20))
    second = chunk_ids(chunk_documents([_doc(text)], chunk_size=200, chunk_overlap=20))
    assert first == second, "ids must be stable so re-ingest upserts instead of duplicating"
    assert len(set(first)) == len(first), "ids must be unique within a document"


def test_overlap_preserves_boundary_content():
    text = " ".join(f"w{i}" for i in range(300))
    no_overlap = chunk_documents([_doc(text)], chunk_size=200, chunk_overlap=0)
    with_overlap = chunk_documents([_doc(text)], chunk_size=200, chunk_overlap=50)
    assert len(with_overlap) >= len(no_overlap)


def test_grade_drops_low_scores():
    chunks = [
        ScoredChunk(_doc("relevant"), 0.80),
        ScoredChunk(_doc("borderline"), 0.36),
        ScoredChunk(_doc("noise"), 0.10),
    ]
    kept = grade(chunks, threshold=0.35)
    assert len(kept) == 2
    assert all(c.score >= 0.35 for c in kept)


def test_grade_can_return_nothing():
    """The 'I don't know' path depends on this being possible."""
    chunks = [ScoredChunk(_doc("noise"), 0.05)]
    assert grade(chunks, threshold=0.35) == []


def test_format_context_numbers_blocks_for_citation():
    chunks = [
        ScoredChunk(_doc("first passage", source="a.pdf", page=2), 0.9),
        ScoredChunk(_doc("second passage", source="b.pdf", page=5), 0.8),
    ]
    context = format_context(chunks)
    assert "[1]" in context and "[2]" in context
    assert "a.pdf" in context and "page: 5" in context
    assert "first passage" in context

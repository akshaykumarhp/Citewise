"""Tests for hybrid retrieval: tokenising, BM25, RRF, reranking, filtering.

No API key needed. These are the pieces where a subtle bug degrades quality
silently — a fusion that quietly favours one ranker, a tokenizer that shatters
identifiers — rather than throwing an error you'd notice.
"""

from __future__ import annotations

from langchain_core.documents import Document

from app.fusion import RRF_K, reciprocal_rank_fusion
from app.lexical import LexicalIndex, tokenize
from app.rerank import NoopReranker, _parse_scores
from app.retrieval import _matches


def _doc(text: str, source: str = "a.pdf", page: int = 1, chunk_id: str | None = None):
    return Document(
        page_content=text,
        metadata={"source": source, "page": page,
                  "chunk_id": chunk_id or f"{source}::{page}::0"},
    )


# --- tokenisation ----------------------------------------------------------


def test_tokenize_keeps_identifiers_whole():
    """The entire point of having BM25 is exact-identifier matching."""
    tokens = tokenize("Compliant with ISO-27001 since v1.2.3")
    assert "iso-27001" in tokens
    assert "v1.2.3" in tokens


def test_tokenize_also_emits_identifier_parts():
    """So a query for '27001' still matches a document containing 'ISO-27001'."""
    tokens = tokenize("ISO-27001")
    assert "iso-27001" in tokens
    assert "27001" in tokens
    assert "iso" in tokens


def test_tokenize_lowercases_and_drops_punctuation():
    assert tokenize("Hello, World!") == ["hello", "world"]


# --- BM25 ------------------------------------------------------------------


CORPUS = [
    "Employees accrue twenty paid vacation days per calendar year.",
    "The platform is certified against ISO-27001 and SOC 2 Type II.",
    "Expense receipts must be submitted within thirty days of purchase.",
]


def _index() -> LexicalIndex:
    return LexicalIndex(
        ids=[f"c{i}" for i in range(len(CORPUS))],
        texts=CORPUS,
        metadatas=[{"source": f"doc{i}.md", "page": 1, "chunk_id": f"c{i}"}
                   for i in range(len(CORPUS))],
    )


def test_bm25_finds_exact_identifier():
    """This is the case dense retrieval reliably fails."""
    hits = _index().search("ISO-27001", k=3)
    assert hits
    assert "ISO-27001" in hits[0][0].page_content


def test_bm25_drops_zero_scoring_documents():
    """BM25 returns non-positive scores for documents sharing no terms. Those are
    not weak matches, they are non-matches, and must not be returned."""
    hits = _index().search("quantum chromodynamics", k=3)
    assert hits == []


def test_bm25_empty_corpus_does_not_explode():
    assert LexicalIndex([], [], []).search("anything", k=5) == []


def test_lexical_index_roundtrips_through_disk(tmp_path):
    path = tmp_path / "lexical.json"
    _index().save(path)
    loaded = LexicalIndex.load(path)
    assert loaded is not None
    assert len(loaded) == len(CORPUS)
    assert loaded.search("ISO-27001", k=1)


def test_corrupt_index_returns_none_rather_than_raising(tmp_path):
    path = tmp_path / "lexical.json"
    path.write_text("{not json")
    assert LexicalIndex.load(path) is None


# --- reciprocal rank fusion ------------------------------------------------


def test_rrf_rewards_agreement_between_rankers():
    """A doc ranked #3 by both rankers should beat one ranked #1 by only one.
    Agreement between two different retrieval strategies is strong evidence."""
    both = _doc("agreed", chunk_id="both")
    only_dense = _doc("dense only", chunk_id="dense-only")

    fused = reciprocal_rank_fusion({
        "dense": [(only_dense, 0.9), (_doc("x", chunk_id="x"), 0.5), (both, 0.4)],
        "lexical": [(_doc("y", chunk_id="y"), 8.0), (_doc("z", chunk_id="z"), 6.0), (both, 4.0)],
    })
    scores = {r.document.metadata["chunk_id"]: r.fused_score for r in fused}
    assert scores["both"] > scores["dense-only"]


def test_rrf_score_matches_the_formula():
    doc = _doc("only", chunk_id="only")
    fused = reciprocal_rank_fusion({"dense": [(doc, 0.9)]})
    assert fused[0].fused_score == 1 / (RRF_K + 1)


def test_rrf_is_immune_to_score_scale():
    """BM25 scores are unbounded and cosine scores are 0-1. Fusion must not care."""
    a, b = _doc("a", chunk_id="a"), _doc("b", chunk_id="b")
    small = reciprocal_rank_fusion({"dense": [(a, 0.01)], "lexical": [(b, 0.02)]})
    huge = reciprocal_rank_fusion({"dense": [(a, 0.01)], "lexical": [(b, 9999.0)]})
    assert [r.document.metadata["chunk_id"] for r in small] == \
           [r.document.metadata["chunk_id"] for r in huge]


def test_rrf_records_provenance():
    doc = _doc("shared", chunk_id="shared")
    fused = reciprocal_rank_fusion({"dense": [(doc, 0.9)], "lexical": [(doc, 5.0)]})
    assert fused[0].found_by == ["dense", "lexical"]
    assert fused[0].ranks == {"dense": 1, "lexical": 1}


def test_rrf_weights_shift_the_balance():
    a, b = _doc("a", chunk_id="a"), _doc("b", chunk_id="b")
    lists = {"dense": [(a, 0.9), (b, 0.1)], "lexical": [(b, 9.0), (a, 1.0)]}
    dense_heavy = reciprocal_rank_fusion(lists, weights={"dense": 3.0, "lexical": 1.0})
    assert dense_heavy[0].document.metadata["chunk_id"] == "a"
    lex_heavy = reciprocal_rank_fusion(lists, weights={"dense": 1.0, "lexical": 3.0})
    assert lex_heavy[0].document.metadata["chunk_id"] == "b"


def test_rrf_on_empty_input():
    assert reciprocal_rank_fusion({}) == []


# --- reranking -------------------------------------------------------------


def test_parse_scores_handles_clean_json():
    assert _parse_scores('[{"id": 1, "score": 8}, {"id": 2, "score": 0}]', 2) == {1: 8.0, 2: 0.0}


def test_parse_scores_survives_markdown_fences_and_prose():
    raw = 'Sure, here you go:\n```json\n[{"id": 1, "score": 9}]\n```'
    assert _parse_scores(raw, 1) == {1: 9.0}


def test_parse_scores_ignores_out_of_range_ids():
    assert _parse_scores('[{"id": 99, "score": 10}, {"id": 1, "score": 5}]', 2) == {1: 5.0}


def test_parse_scores_returns_empty_on_garbage():
    """Which is what makes the reranker fall back to fusion order."""
    assert _parse_scores("I could not score these passages.", 3) == {}
    assert _parse_scores("", 3) == {}


def test_parse_scores_clamps_range():
    assert _parse_scores('[{"id": 1, "score": 50}, {"id": 2, "score": -3}]', 2) == {1: 10.0, 2: 0.0}


def test_noop_reranker_preserves_order_and_truncates():
    docs = [_doc(f"d{i}", chunk_id=f"d{i}") for i in range(5)]
    out = NoopReranker().rerank("q", docs, top_n=3)
    assert [d.metadata["chunk_id"] for d, _ in out] == ["d0", "d1", "d2"]


# --- metadata filtering ----------------------------------------------------


def test_filter_equality():
    assert _matches({"source": "a.pdf"}, {"source": "a.pdf"})
    assert not _matches({"source": "b.pdf"}, {"source": "a.pdf"})


def test_filter_membership():
    assert _matches({"source": "a.pdf"}, {"source": ["a.pdf", "b.pdf"]})
    assert _matches({"source": "a.pdf"}, {"source": {"$in": ["a.pdf"]}})
    assert not _matches({"source": "c.pdf"}, {"source": {"$in": ["a.pdf"]}})


def test_filter_requires_all_keys_to_match():
    assert not _matches({"source": "a.pdf", "page": 2}, {"source": "a.pdf", "page": 5})


def test_missing_metadata_key_does_not_match():
    assert not _matches({"page": 1}, {"source": "a.pdf"})

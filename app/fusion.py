"""Reciprocal Rank Fusion.

The problem: dense similarity scores (cosine, 0-1) and BM25 scores (unbounded,
corpus-dependent) live on incomparable scales. You cannot average them, and
normalising them per-query is unstable — a query where everything scores badly
gets its noise stretched to fill the same 0-1 range as a query with a perfect hit.

RRF sidesteps this entirely by throwing away the scores and keeping only the
*ranks*:

    score(d) = Σ over rankers  1 / (k + rank(d))

with rank 1-based and k a damping constant (60 is the value from the original
paper and remains the conventional default). A document ranked #1 by one ranker
and absent from the other scores 1/61. A document ranked #3 by both scores
2/63 — higher. That is the behaviour you want: agreement between two different
retrieval strategies is strong evidence, and neither ranker can dominate through
score inflation.

The constant k controls how sharply top ranks are rewarded. Small k makes rank 1
overwhelmingly important; large k flattens everything toward equal weight.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from langchain_core.documents import Document

RRF_K = 60


@dataclass
class FusedResult:
    document: Document
    fused_score: float
    # Per-ranker provenance, e.g. {"dense": 1, "lexical": 4}. Kept because
    # "which ranker found this?" is the first question when debugging retrieval.
    ranks: dict[str, int] = field(default_factory=dict)
    raw_scores: dict[str, float] = field(default_factory=dict)

    @property
    def found_by(self) -> list[str]:
        return sorted(self.ranks)


def _key(doc: Document) -> str:
    """Identity for fusion. Chunk ids are deterministic, so prefer them; fall back
    to the text itself when a document arrives without metadata."""
    return doc.metadata.get("chunk_id") or doc.page_content


def reciprocal_rank_fusion(
    ranked_lists: dict[str, list[tuple[Document, float]]],
    k: int = RRF_K,
    weights: dict[str, float] | None = None,
) -> list[FusedResult]:
    """Fuse several ranked lists into one.

    Args:
        ranked_lists: {ranker_name: [(document, raw_score), ...]}, best first.
        k: RRF damping constant.
        weights: optional per-ranker multiplier. Defaults to 1.0 for every ranker.

    Returns:
        Fused results, highest score first.
    """
    weights = weights or {}
    fused: dict[str, FusedResult] = {}

    for ranker, results in ranked_lists.items():
        weight = weights.get(ranker, 1.0)
        for rank, (doc, raw) in enumerate(results, start=1):
            key = _key(doc)
            entry = fused.get(key)
            if entry is None:
                entry = FusedResult(document=doc, fused_score=0.0)
                fused[key] = entry
            entry.fused_score += weight / (k + rank)
            entry.ranks[ranker] = rank
            entry.raw_scores[ranker] = float(raw)

    return sorted(fused.values(), key=lambda r: r.fused_score, reverse=True)

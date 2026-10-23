"""Reranking.

Retrieval and reranking answer different questions, and conflating them is why a
lot of RAG systems plateau.

**Retrieval is a recall problem.** A bi-encoder embeds the query and every document
*independently*, ahead of time, into fixed vectors. That is what makes searching a
million chunks fast — but the model never sees a query and a document together, so
it can only ever measure a blurry, general-purpose notion of similarity. Fetch 20
candidates and the right passage is very likely somewhere among them.

**Reranking is a precision problem.** A reranker looks at (query, document) *pairs*
directly and scores their actual relevance to each other. Far more accurate, far
too slow to run over the whole corpus — but perfectly affordable over 20 candidates.

So: retrieve wide and cheap, rerank narrow and accurate. Recall from the first
stage, precision from the second.

Two backends:

- `llm` (default) — scores every candidate in a **single** Gemini call. No extra
  dependencies, works with the API key you already have. One call per query.
- `cross-encoder` — a local sentence-transformers cross-encoder. Genuinely better
  and free per query, but pulls in torch (~2GB). Opt in with:
  `pip install sentence-transformers` and `RERANK_BACKEND=cross-encoder`.
- `none` — passthrough, keeping fusion order. Useful as a baseline to measure
  whether reranking is actually helping on your corpus.

Every backend fails safe: if reranking errors out, the original order is returned.
A broken reranker degrades the system to Phase 2 behaviour, it does not break it.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Protocol

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from app.config import get_settings
from app.prompts import text_of

logger = logging.getLogger(__name__)


class Reranker(Protocol):
    def rerank(
        self, query: str, documents: list[Document], top_n: int
    ) -> list[tuple[Document, float]]: ...


# --- passthrough -----------------------------------------------------------


class NoopReranker:
    """Keeps the incoming order. The honest baseline to measure against."""

    name = "none"

    def rerank(self, query: str, documents: list[Document], top_n: int):
        return [(d, 0.0) for d in documents[:top_n]]


# --- LLM reranker ----------------------------------------------------------

RERANK_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You score how well each passage answers a specific question.\n\n"
            "For every passage, output an integer relevance score from 0 to 10:\n"
            "  10 = directly and completely answers the question\n"
            "   7 = contains most of the answer\n"
            "   4 = related topic, but does not answer the question\n"
            "   1 = same general domain only\n"
            "   0 = irrelevant\n\n"
            "Judge only whether the passage answers THIS question. A well-written, "
            "interesting passage about something else scores 0.\n\n"
            "Output ONLY a JSON array, one object per passage, in the form:\n"
            '[{{"id": 1, "score": 8}}, {{"id": 2, "score": 0}}]\n'
            "No prose, no markdown fences, no explanation.",
        ),
        ("human", "QUESTION: {question}\n\nPASSAGES:\n{passages}\n\nJSON scores:"),
    ]
)

# Passages are truncated before scoring: the reranker needs enough to judge
# relevance, not the whole chunk, and 20 full chunks is a lot of tokens per query.
RERANK_EXCERPT_CHARS = 900


class LLMReranker:
    """Scores every candidate in one call. No extra dependencies."""

    name = "llm"

    def __init__(self, llm=None):
        self._llm = llm

    def _get_llm(self):
        if self._llm is None:
            from app.graph import get_llm

            self._llm = get_llm()
        return self._llm

    def rerank(self, query: str, documents: list[Document], top_n: int):
        if not documents:
            return []

        passages = "\n\n".join(
            f"[{i}] {d.page_content.strip()[:RERANK_EXCERPT_CHARS]}"
            for i, d in enumerate(documents, start=1)
        )

        try:
            response = (RERANK_PROMPT | self._get_llm()).invoke(
                {"question": query, "passages": passages}
            )
            scores = _parse_scores(text_of(response), len(documents))
        except Exception as exc:  # noqa: BLE001 - never let reranking break a query
            logger.warning("LLM rerank failed (%s); keeping fusion order", exc)
            return [(d, 0.0) for d in documents[:top_n]]

        if not scores:
            logger.warning("LLM rerank returned no usable scores; keeping fusion order")
            return [(d, 0.0) for d in documents[:top_n]]

        # Normalise 0-10 to 0-1 so downstream thresholds have one scale to reason about.
        ranked = sorted(
            ((documents[i], scores.get(i + 1, 0) / 10.0) for i in range(len(documents))),
            key=lambda pair: pair[1],
            reverse=True,
        )
        return ranked[:top_n]


def _parse_scores(raw, expected: int) -> dict[int, float]:
    """Pull {id: score} out of a model response, tolerating stray prose or fences."""
    text = text_of(raw).strip()

    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}

    scores: dict[int, float] = {}
    for item in parsed:
        if not isinstance(item, dict):
            continue
        try:
            idx = int(item["id"])
            score = float(item["score"])
        except (KeyError, TypeError, ValueError):
            continue
        if 1 <= idx <= expected:
            scores[idx] = max(0.0, min(10.0, score))
    return scores


# --- cross-encoder ---------------------------------------------------------


class CrossEncoderReranker:
    """Local cross-encoder. Better and free per query, but needs torch."""

    name = "cross-encoder"

    def __init__(self, model_name: str | None = None):
        settings = get_settings()
        self.model_name = model_name or settings.cross_encoder_model
        self._model = None

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:  # pragma: no cover - depends on optional dep
                raise RuntimeError(
                    "RERANK_BACKEND=cross-encoder needs sentence-transformers.\n"
                    "Install it with: pip install sentence-transformers\n"
                    "(this pulls in torch, ~2GB). Or set RERANK_BACKEND=llm."
                ) from exc
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, query: str, documents: list[Document], top_n: int):
        if not documents:
            return []
        try:
            model = self._get_model()
            pairs = [(query, d.page_content) for d in documents]
            raw = model.predict(pairs)
        except Exception as exc:  # noqa: BLE001
            logger.warning("cross-encoder rerank failed (%s); keeping order", exc)
            return [(d, 0.0) for d in documents[:top_n]]

        # Cross-encoder logits are unbounded; squash to 0-1 so the relevance
        # threshold means the same thing whichever backend is in use.
        import math

        scored = [
            (doc, 1.0 / (1.0 + math.exp(-float(score))))
            for doc, score in zip(documents, raw)
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_n]


# --- factory ---------------------------------------------------------------

_BACKENDS = {
    "none": NoopReranker,
    "llm": LLMReranker,
    "cross-encoder": CrossEncoderReranker,
}

_cached: dict[str, Reranker] = {}


def get_reranker(backend: str | None = None) -> Reranker:
    settings = get_settings()
    name = (backend or settings.rerank_backend).lower()
    if name not in _BACKENDS:
        logger.warning("unknown rerank backend %r; falling back to 'none'", name)
        name = "none"
    if name not in _cached:
        _cached[name] = _BACKENDS[name]()
    return _cached[name]


def reset_cache() -> None:
    _cached.clear()

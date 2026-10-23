"""Semantic answer cache.

Exact-string caching misses "how many vacation days do I get" vs "how many vacation
days do I get?". Instead the standalone question is embedded (an embedding dense
retrieval needed anyway, and which the query-embedding cache keeps) and a cached
answer is reused when a new question lands within `answer_cache_similarity`
cosine of an old one, under identical retrieval settings.

Only history-free questions are cached: with history, the same words can mean a
different question. The cache is dropped after every ingest, since new documents
can change any answer.
"""

from __future__ import annotations

import json
import logging

from app import embeddings
from app.config import get_settings

logger = logging.getLogger(__name__)

# ponytail: in-process linear scan, fine to a few thousand entries; a vector index
# (or Redis) when it is shared across replicas or grows past that.
_ENTRIES: list[tuple[list[float], str, dict]] = []
_MAX_ENTRIES = 2000


def params_key(**params) -> str:
    return json.dumps(params, sort_keys=True, default=str)


def _embed(question: str) -> list[float] | None:
    try:
        return embeddings.get_embeddings().embed_query(question.strip().lower())
    except Exception as exc:  # noqa: BLE001 - the cache must never fail a query
        logger.warning("answer cache embed failed: %s", exc)
        return None


def lookup(question: str, key: str) -> dict | None:
    threshold = get_settings().answer_cache_similarity
    if threshold <= 0 or not _ENTRIES:
        return None
    vector = _embed(question)
    if vector is None:
        return None
    for cached_vector, cached_key, result in _ENTRIES:
        # Vectors are unit-normalised, so the dot product is the cosine.
        if cached_key == key and sum(a * b for a, b in zip(vector, cached_vector)) >= threshold:
            logger.info("answer cache hit for %r", question)
            return result
    return None


def store(question: str, key: str, result: dict) -> None:
    if get_settings().answer_cache_similarity <= 0:
        return
    vector = _embed(question)
    if vector is None:
        return
    if len(_ENTRIES) >= _MAX_ENTRIES:
        _ENTRIES.pop(0)
    _ENTRIES.append((vector, key, result))


def clear() -> None:
    _ENTRIES.clear()

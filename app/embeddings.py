"""Embeddings.

Two things here are easy to get wrong and both measurably hurt retrieval quality.

**1. Asymmetric task types.**
A question and the passage that answers it do not look alike. "How many vacation days
do I get?" shares almost no vocabulary with "Employees accrue 1.67 days of paid leave
per completed month of service." Gemini's embedding model can be told which side of
that asymmetry it is embedding - `RETRIEVAL_DOCUMENT` when indexing, `RETRIEVAL_QUERY`
when searching - and it projects them into a space where they land near each other.

LangChain's vector-store integrations call `embed_documents()` and `embed_query()`
without passing a task type, so the wrapper below binds the right one to each method.

**2. Normalisation.**
`gemini-embedding-001` returns unit-normalised vectors only at its native 3072
dimensions. At any truncated dimension (768, 1536, ...) the vectors are NOT normalised,
and cosine similarity on unnormalised vectors is quietly wrong. We normalise ourselves.
"""

from __future__ import annotations

import logging
import math
import random
import time
from typing import Any, Callable, TypeVar

from langchain_core.embeddings import Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.config import get_settings

logger = logging.getLogger(__name__)

T = TypeVar("T")

NATIVE_DIMENSIONS = 3072

# Query embedding cache. Document embeddings are already never repeated (the ingest
# manifest skips unchanged files); query embeddings repeat constantly - the answer
# cache, HyDE-off dense search and eval reruns all embed the same strings.
# ponytail: per-process and cleared wholesale when full; use Redis if replicas > 1.
_QUERY_CACHE: dict[tuple[str, int, str], list[float]] = {}
_QUERY_CACHE_MAX = 4096


def with_backoff(fn: Callable[[], T], *, max_retries: int, label: str = "call") -> T:
    """Retry with exponential backoff and jitter.

    Free-tier embedding quotas are per-minute, so a bulk ingest will hit 429s.
    That is expected, not an error - back off and continue.
    """
    delay = 1.0
    last: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - provider raises many shapes
            last = exc
            text = str(exc).lower()
            retryable = any(
                token in text
                for token in ("429", "resource_exhausted", "rate", "503", "unavailable",
                              "500", "internal", "deadline", "timeout")
            )
            if not retryable or attempt == max_retries:
                raise
            sleep_for = delay + random.uniform(0, 0.5 * delay)
            logger.warning(
                "%s failed (attempt %d/%d): %s - retrying in %.1fs",
                label, attempt, max_retries, exc, sleep_for,
            )
            time.sleep(sleep_for)
            delay = min(delay * 2, 60.0)
    assert last is not None
    raise last


def _normalise(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0.0:
        return vector
    return [v / norm for v in vector]


class GeminiRetrievalEmbeddings(Embeddings):
    """Task-typed, normalised, retrying, batched Gemini embeddings."""

    def __init__(
        self,
        model: str | None = None,
        dimensions: int | None = None,
        api_key: str | None = None,
        batch_size: int | None = None,
        max_retries: int | None = None,
    ) -> None:
        settings = get_settings()
        self.model = model or settings.embedding_model
        self.dimensions = dimensions or settings.embedding_dimensions
        self.batch_size = batch_size or settings.embed_batch_size
        self.max_retries = max_retries or settings.max_retries
        self._client = GoogleGenerativeAIEmbeddings(
            model=self.model,
            api_key=api_key or settings.require_api_key(),
        )

    # -- helpers ------------------------------------------------------------
    @property
    def _dim_arg(self) -> int | None:
        # Passing the native dimension explicitly is a no-op; omit it.
        return None if self.dimensions == NATIVE_DIMENSIONS else self.dimensions

    def _post(self, vectors: list[list[float]]) -> list[list[float]]:
        if self.dimensions == NATIVE_DIMENSIONS:
            return vectors
        return [_normalise(v) for v in vectors]

    # -- Embeddings interface ----------------------------------------------
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        total = len(texts)
        for start in range(0, total, self.batch_size):
            batch = texts[start : start + self.batch_size]

            def call(batch: list[str] = batch) -> list[list[float]]:
                return self._client.embed_documents(
                    batch,
                    task_type="RETRIEVAL_DOCUMENT",
                    output_dimensionality=self._dim_arg,
                    batch_size=len(batch),
                )

            vectors = with_backoff(
                call,
                max_retries=self.max_retries,
                label=f"embed_documents[{start}:{start + len(batch)}]",
            )
            out.extend(self._post(vectors))
            logger.info("embedded %d/%d chunks", min(start + len(batch), total), total)
        return out

    def embed_query(self, text: str) -> list[float]:
        key = (self.model, self.dimensions, text)
        if key in _QUERY_CACHE:
            return _QUERY_CACHE[key]

        def call() -> list[float]:
            return self._client.embed_query(
                text,
                task_type="RETRIEVAL_QUERY",
                output_dimensionality=self._dim_arg,
            )

        vector = with_backoff(call, max_retries=self.max_retries, label="embed_query")
        if len(_QUERY_CACHE) >= _QUERY_CACHE_MAX:
            _QUERY_CACHE.clear()
        _QUERY_CACHE[key] = self._post([vector])[0]
        return _QUERY_CACHE[key]


def get_embeddings(**kwargs: Any) -> GeminiRetrievalEmbeddings:
    return GeminiRetrievalEmbeddings(**kwargs)

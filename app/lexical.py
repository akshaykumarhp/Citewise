"""BM25 lexical index.

Dense retrieval fails hardest exactly where precision matters most: exact
identifiers, product codes, error codes, version numbers, surnames, acronyms.
An embedding of "ISO-27001" lands near "information security standard" — which is
semantically right and operationally useless when the user typed the code because
they want the passage containing that code.

BM25 is the opposite failure mode: it cannot match "vacation" to "annual leave",
but it will find "ISO-27001" every single time. Running both and fusing the
rankings covers each one's blind spot. That is the whole argument for hybrid search.

**Index lifecycle.** BM25 needs the full corpus to compute term statistics, so it
cannot be built incrementally the way the vector store is. Instead it is rebuilt
from the Chroma collection after any ingest that wrote chunks. That is cheap —
no embedding calls, just tokenising and counting — and it guarantees the lexical
and dense indexes never disagree about what is in the corpus.

The index persists as JSON rather than a pickle: inspectable, diffable, and immune
to breaking when rank_bm25 changes its internals.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from langchain_core.documents import Document

from app.config import get_settings

logger = logging.getLogger(__name__)

# Keeps compound identifiers whole: "iso-27001", "v1.2.3", "gemini-embedding-001".
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-_./][a-z0-9]+)*")
_SPLIT_RE = re.compile(r"[-_./]")


def tokenize(text: str) -> list[str]:
    """Lowercase, keep compound identifiers whole, and also emit their parts.

    Emitting both forms means a query for `27001` matches a document containing
    `ISO-27001`, and a query for the full code still scores higher because it
    matches the compound token too.
    """
    tokens: list[str] = []
    for match in _TOKEN_RE.findall(text.lower()):
        tokens.append(match)
        if _SPLIT_RE.search(match):
            tokens.extend(p for p in _SPLIT_RE.split(match) if p)
    return tokens


class LexicalIndex:
    """A BM25 index over the same chunks the vector store holds."""

    def __init__(self, ids: list[str], texts: list[str], metadatas: list[dict]):
        from rank_bm25 import BM25Okapi

        self.ids = ids
        self.texts = texts
        self.metadatas = metadatas
        self._corpus = [tokenize(t) for t in texts]
        # BM25Okapi on an empty corpus raises; guard it.
        self._bm25 = BM25Okapi(self._corpus) if self._corpus else None

    def __len__(self) -> int:
        return len(self.ids)

    def search(self, query: str, k: int = 20) -> list[tuple[Document, float]]:
        """Top-k by BM25 score, highest first. Zero-scoring documents are dropped."""
        if not self._bm25:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        results = []
        for i in ranked:
            # BM25 can return non-positive scores for documents sharing no terms.
            # Those are not matches, they are the least-bad non-matches.
            if scores[i] <= 0:
                continue
            results.append(
                (
                    Document(page_content=self.texts[i], metadata=dict(self.metadatas[i])),
                    float(scores[i]),
                )
            )
        return results

    # -- persistence --------------------------------------------------------

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {"ids": self.ids, "texts": self.texts, "metadatas": self.metadatas}
            )
        )

    @classmethod
    def load(cls, path: Path) -> "LexicalIndex | None":
        if not path.exists():
            return None
        try:
            raw = json.loads(path.read_text())
        except json.JSONDecodeError:
            logger.warning("lexical index at %s is corrupt; ignoring", path)
            return None
        return cls(raw["ids"], raw["texts"], raw["metadatas"])


def build_from_vectorstore() -> LexicalIndex:
    """Rebuild the lexical index from whatever is currently in the vector store.

    Reads documents and metadata only — no embedding function needed, so this
    never requires an API key.
    """
    from app.vectorstore import all_chunks

    ids, texts, metadatas = all_chunks()
    return LexicalIndex(ids=ids, texts=texts, metadatas=metadatas)


def rebuild_index() -> int:
    """Rebuild and persist. Returns the number of documents indexed."""
    settings = get_settings()
    index = build_from_vectorstore()
    index.save(settings.lexical_index_path)
    logger.info("rebuilt lexical index: %d documents", len(index))
    return len(index)


_cached: LexicalIndex | None = None
_cached_mtime: float | None = None


def get_lexical_index() -> LexicalIndex | None:
    """Load the persisted index, reloading if it changed on disk."""
    global _cached, _cached_mtime
    settings = get_settings()
    path = settings.lexical_index_path
    if not path.exists():
        return None
    mtime = path.stat().st_mtime
    if _cached is None or _cached_mtime != mtime:
        _cached = LexicalIndex.load(path)
        _cached_mtime = mtime
    return _cached


def reset_cache() -> None:
    """Drop the in-process cache. Tests and `--rebuild` use this."""
    global _cached, _cached_mtime
    _cached = None
    _cached_mtime = None

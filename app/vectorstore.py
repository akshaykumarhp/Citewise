"""Vector store access.

Everything else in the codebase talks to the vector store through the small surface
in this module, so the backend is a one-setting swap:

- `chroma` (default) persists to a directory on disk. Zero infrastructure.
- `pgvector` stores vectors in PostgreSQL (`VECTOR_BACKEND=pgvector`, `DATABASE_URL`).
  The realistic production answer: backups, replication and access control come
  from a database you already operate.

A note on the distance metric: Chroma defaults to squared L2. We explicitly ask for
cosine, because our embeddings are unit-normalised and cosine is what the embedding
model was trained against. PGVector defaults to cosine already.
"""

from __future__ import annotations

from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from app.config import get_settings
from app.embeddings import get_embeddings


def _pg() -> bool:
    return get_settings().vector_backend.lower() == "pgvector"


def get_vectorstore(embeddings: Embeddings | None = None) -> VectorStore:
    return _store(embeddings or get_embeddings())


def _store(embeddings: Embeddings | None = None) -> VectorStore:
    """The raw store. No embedding function is needed to read metadata or delete,
    so stats and maintenance never require an API key."""
    settings = get_settings()
    if _pg():
        from langchain_postgres import PGVector

        if not settings.database_url:
            raise RuntimeError("VECTOR_BACKEND=pgvector needs DATABASE_URL")
        return PGVector(
            embeddings=embeddings,
            connection=settings.database_url,
            collection_name=settings.collection_name,
            embedding_length=settings.embedding_dimensions,
            use_jsonb=True,
        )

    from langchain_chroma import Chroma

    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=embeddings,
        persist_directory=str(settings.chroma_dir),
        collection_metadata={"hnsw:space": "cosine"},
    )


def all_chunks(where: dict | None = None) -> tuple[list[str], list[str], list[dict]]:
    """(ids, texts, metadatas) for every stored chunk, optionally filtered by
    metadata equality. Used for stats, deletes and rebuilding the BM25 index."""
    settings = get_settings()
    if _pg():
        from sqlalchemy import create_engine, text

        _store()  # creates the tables on first use
        engine = create_engine(settings.database_url)
        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    "SELECT e.id, e.document, e.cmetadata FROM langchain_pg_embedding e "
                    "JOIN langchain_pg_collection c ON e.collection_id = c.uuid "
                    "WHERE c.name = :name ORDER BY e.id"
                ),
                {"name": settings.collection_name},
            ).all()
        engine.dispose()
        rows = [
            r for r in rows
            if not where or all((r[2] or {}).get(k) == v for k, v in where.items())
        ]
        return [r[0] for r in rows], [r[1] for r in rows], [dict(r[2] or {}) for r in rows]

    if not settings.chroma_dir.exists():
        return [], [], []
    raw = _store().get(where=where, include=["documents", "metadatas"])
    return (
        raw.get("ids") or [],
        raw.get("documents") or [],
        [dict(m or {}) for m in (raw.get("metadatas") or [])],
    )


def collection_stats() -> dict:
    """Counts and per-source breakdown, without needing an API key."""
    _, _, metadatas = all_chunks()
    sources: dict[str, int] = {}
    for meta in metadatas:
        name = meta.get("source", "unknown")
        sources[name] = sources.get(name, 0) + 1
    return {"chunks": len(metadatas), "sources": dict(sorted(sources.items()))}


def delete_by_source(source: str) -> int:
    """Remove every chunk originating from one file. Used when a file changes."""
    ids, _, _ = all_chunks(where={"source": source})
    if ids:
        _store().delete(ids=ids)
    return len(ids)


def reset_collection() -> None:
    """Drop the whole collection. `ingest --rebuild` uses this."""
    if not _pg() and not get_settings().chroma_dir.exists():
        return
    _store().delete_collection()

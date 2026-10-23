"""The ingestion pipeline: discover -> load -> chunk -> embed -> upsert.

The one non-obvious design choice here is the **manifest**. It records the content
hash of every file that has been ingested. On re-run, unchanged files are skipped
entirely.

That matters more than it sounds like it does: embedding is the slow, rate-limited,
metered step in the whole system. Without a manifest, adding one new PDF to a folder
of fifty means re-embedding all fifty. With one, it means embedding one.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from app import cache as answer_cache
from app import lexical
from app.chunking import chunk_documents, chunk_ids
from app.config import get_settings
from app.loaders import discover_files, file_hash, load_file
from app.vectorstore import delete_by_source, get_vectorstore, reset_collection

logger = logging.getLogger(__name__)


@dataclass
class IngestReport:
    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)
    chunks_written: int = 0
    lexical_documents: int = 0

    @property
    def touched(self) -> int:
        return len(self.added) + len(self.updated)


def _load_manifest(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        logger.warning("manifest at %s is corrupt; starting fresh", path)
        return {}


def _save_manifest(path: Path, manifest: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True))


def ingest(
    data_dir: Path | None = None,
    rebuild: bool = False,
    progress=None,
) -> IngestReport:
    """Ingest every supported file under `data_dir`.

    Args:
        data_dir: folder to scan. Defaults to settings.data_dir.
        rebuild: drop the collection and manifest first, re-embedding everything.
        progress: optional callable(str) for user-facing status lines.
    """
    settings = get_settings()
    data_dir = data_dir or settings.data_dir
    say = progress or (lambda _msg: None)

    if rebuild:
        say("Rebuilding: dropping existing collection, manifest and lexical index")
        reset_collection()
        settings.manifest_path.unlink(missing_ok=True)
        settings.lexical_index_path.unlink(missing_ok=True)
        lexical.reset_cache()

    manifest = _load_manifest(settings.manifest_path)
    files = discover_files(data_dir)
    report = IngestReport()

    if not files:
        say(f"No supported documents found under {data_dir}")
        return report

    store = get_vectorstore()

    for path in files:
        key = str(path.relative_to(data_dir)) if path.is_relative_to(data_dir) else str(path)
        try:
            digest = file_hash(path)
            if manifest.get(key) == digest:
                report.skipped.append(key)
                continue

            is_update = key in manifest
            if is_update:
                removed = delete_by_source(path.name)
                say(f"{path.name}: changed - removed {removed} stale chunks")

            say(f"{path.name}: loading")
            docs = load_file(path)
            if not docs:
                report.failed.append((key, "no extractable text"))
                continue

            chunks = chunk_documents(docs)
            say(f"{path.name}: {len(docs)} page(s) -> {len(chunks)} chunk(s), embedding")

            # Deterministic ids make this an upsert rather than an append.
            store.add_documents(documents=chunks, ids=chunk_ids(chunks))

            manifest[key] = digest
            report.chunks_written += len(chunks)
            (report.updated if is_update else report.added).append(key)
            # Save after each file so an interrupted run does not lose its work.
            _save_manifest(settings.manifest_path, manifest)

        except Exception as exc:  # noqa: BLE001 - one bad file must not kill the run
            logger.exception("failed to ingest %s", path)
            report.failed.append((key, str(exc)))

    _save_manifest(settings.manifest_path, manifest)

    # BM25 needs whole-corpus term statistics, so it can't be updated incrementally
    # the way the vector store is. Rebuilding from Chroma is cheap (no embedding
    # calls) and guarantees the lexical and dense indexes never disagree about
    # what is in the corpus.
    if report.touched or rebuild:
        answer_cache.clear()
        lexical.reset_cache()
        count = lexical.rebuild_index()
        report.lexical_documents = count
        say(f"Rebuilt lexical (BM25) index: {count} chunks")

    return report

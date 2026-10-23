"""FastAPI service.

    uvicorn app.api:app --reload --port 8000
    open http://localhost:8000/docs

Endpoints:
    GET  /healthz       liveness + configuration sanity, never raises
    GET  /stats         what is in the index
    POST /ingest        run the ingestion pipeline
    POST /ask           blocking, returns the full answer plus sources
    POST /ask/stream    SSE: sources first, then answer tokens

Note on /ingest: it runs synchronously and can take minutes on a large corpus.
A real multi-user deployment wants a background worker and a job id; for a
single-corpus service an explicit, admin-only, idempotent call is enough.

Phase 5 hardening, all in this file:
- JSON logs with a request id on every line (`X-Request-ID` echoed back)
- per-client rate limiting on the endpoints that spend tokens
- `/ingest` behind `ADMIN_TOKEN`, and only for folders inside the project
- CORS narrowed by `ALLOWED_ORIGINS`
- token and cost accounting per request (logged, and returned as `usage`)
"""

from __future__ import annotations

import contextvars
import json
import logging
import secrets
import time
import uuid
from collections import defaultdict, deque
from pathlib import Path
from typing import Iterator

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from app.config import PROJECT_ROOT, get_settings
from app.graph import ask as run_ask
from app.graph import ask_stream as run_ask_stream
from app.schemas import (
    AskRequest,
    AskResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    StatsResponse,
)
from app.vectorstore import collection_stats

logger = logging.getLogger(__name__)

# --- structured logging ------------------------------------------------------

request_id: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": round(record.created, 3),
            "level": record.levelname,
            "logger": record.name,
            "request_id": request_id.get(),
            "msg": record.getMessage(),
        }
        entry.update(getattr(record, "fields", {}))
        if record.exc_info:
            entry["exc"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str)


_handler = logging.StreamHandler()
_handler.setFormatter(JsonFormatter())
logging.getLogger("app").handlers[:] = [_handler]
logging.getLogger("app").setLevel(logging.INFO)
logging.getLogger("app").propagate = False

app = FastAPI(
    title="Citewise",
    description="Answers with receipts: grounded, cited question answering over your "
    "own documents.",
    version="0.4.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in get_settings().allowed_origins.split(",")],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_context(request: Request, call_next):
    rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    token = request_id.set(rid)
    started = time.perf_counter()
    try:
        response = await call_next(request)
    finally:
        request_id.reset(token)
    response.headers["X-Request-ID"] = rid
    logger.info(
        "request",
        extra={"fields": {
            "request_id": rid, "method": request.method, "path": request.url.path,
            "status": response.status_code,
            "ms": round((time.perf_counter() - started) * 1000, 1),
        }},
    )
    return response


# --- guards ------------------------------------------------------------------

# ponytail: in-process sliding window, so the limit is per replica; move to Redis
# (or the reverse proxy) when running more than one.
_hits: dict[str, deque[float]] = defaultdict(deque)


def client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def rate_limit(request: Request) -> None:
    limit = get_settings().rate_limit_per_minute
    if limit <= 0:
        return
    now = time.monotonic()
    window = _hits[client_ip(request)]
    while window and now - window[0] > 60:
        window.popleft()
    if len(window) >= limit:
        raise HTTPException(
            status_code=429,
            detail=f"rate limit: {limit} requests per minute",
            headers={"Retry-After": str(int(60 - (now - window[0])) + 1)},
        )
    window.append(now)


def require_admin(request: Request) -> None:
    token = get_settings().admin_token
    if not token:
        return
    supplied = request.headers.get("authorization", "").removeprefix("Bearer ").strip()
    if not secrets.compare_digest(supplied, token):
        raise HTTPException(status_code=401, detail="admin token required")


@app.get("/healthz", response_model=HealthResponse, tags=["ops"])
def healthz() -> HealthResponse:
    """Liveness plus a configuration sanity check.

    Deliberately never raises: a health endpoint that 500s tells you the service is
    up but gives an orchestrator nothing to act on. Degraded-with-a-reason is more
    useful than a stack trace.
    """
    settings = get_settings()
    try:
        chunks = collection_stats()["chunks"]
    except Exception as exc:  # noqa: BLE001
        return HealthResponse(
            status="degraded",
            api_key_configured=bool(settings.google_api_key),
            chat_model=settings.chat_model,
            embedding_model=settings.embedding_model,
            indexed_chunks=0,
            detail=f"vector store unreadable: {exc}",
        )

    from app.lexical import get_lexical_index

    index = get_lexical_index()
    lexical_count = len(index) if index else 0

    problems = []
    if not settings.google_api_key:
        problems.append("GOOGLE_API_KEY not configured")
    if chunks == 0:
        problems.append("index is empty - run /ingest")
    elif settings.retrieval_mode == "hybrid" and lexical_count == 0:
        # Hybrid silently degrades to dense-only without this - worth surfacing.
        problems.append("hybrid mode but no lexical index - re-run /ingest")

    return HealthResponse(
        status="ok" if not problems else "degraded",
        api_key_configured=bool(settings.google_api_key),
        chat_model=settings.chat_model,
        embedding_model=settings.embedding_model,
        indexed_chunks=chunks,
        lexical_documents=lexical_count,
        retrieval_mode=settings.retrieval_mode,
        rerank_backend=settings.rerank_backend,
        detail="; ".join(problems) or None,
    )


@app.get("/stats", response_model=StatsResponse, tags=["ops"])
def stats() -> StatsResponse:
    info = collection_stats()
    return StatsResponse(
        chunks=info["chunks"],
        sources=[{"source": s, "chunks": c} for s, c in info["sources"].items()],
    )


@app.post(
    "/ingest", response_model=IngestResponse, tags=["ingestion"],
    dependencies=[Depends(require_admin), Depends(rate_limit)],
)
def ingest(request: IngestRequest) -> IngestResponse:
    """Index documents. Synchronous - can take minutes on a large corpus."""
    from app.ingest import ingest as run_ingest

    settings = get_settings()
    target = (Path(request.path) if request.path else settings.data_dir).resolve()
    allowed = (PROJECT_ROOT.resolve(), settings.data_dir.resolve())
    if not any(target.is_relative_to(root) for root in allowed):
        raise HTTPException(status_code=400, detail="path must be inside the project")
    if not target.exists():
        raise HTTPException(status_code=400, detail=f"path does not exist: {target}")

    try:
        report = run_ingest(data_dir=target, rebuild=request.rebuild)
    except Exception as exc:  # noqa: BLE001
        logger.exception("ingestion failed")
        raise HTTPException(status_code=500, detail=f"ingestion failed: {exc}") from exc

    return IngestResponse(
        added=report.added,
        updated=report.updated,
        skipped=report.skipped,
        failed=[{"file": name, "reason": reason} for name, reason in report.failed],
        chunks_written=report.chunks_written,
        lexical_documents=report.lexical_documents,
    )


def _log_answer(result: dict) -> None:
    logger.info("answered", extra={"fields": {
        "grounded": result.get("grounded"), "cached": result.get("cached", False),
        "sources": len(result.get("sources") or []), **(result.get("usage") or {}),
    }})


@app.post(
    "/ask", response_model=AskResponse, tags=["query"], dependencies=[Depends(rate_limit)]
)
def ask(request: AskRequest) -> AskResponse:
    """Ask a question. Pass `history` to make follow-ups resolve correctly."""
    try:
        result = run_ask(
            request.question,
            history=request.history_dicts(),
            k=request.k,
            threshold=request.threshold,
            mode=request.mode,
            candidate_k=request.candidate_k,
            rerank_backend=request.rerank_backend,
            where=request.where,
        )
    except RuntimeError as exc:  # missing API key
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("query failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    _log_answer(result)
    return AskResponse(**result)


def _sse(events: Iterator[dict]) -> Iterator[str]:
    """Server-Sent Events framing: `data: <json>\\n\\n` per event."""
    for event in events:
        if event.get("event") == "done":
            _log_answer(event)
        yield f"data: {json.dumps(event)}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/ask/stream", tags=["query"], dependencies=[Depends(rate_limit)])
def ask_stream(request: AskRequest) -> StreamingResponse:
    """Stream an answer as SSE.

    Event order is always: one `meta` (with sources), then zero or more `token`,
    then one `done`. An `error` event may replace the tail at any point.
    """
    events = run_ask_stream(
        request.question,
        history=request.history_dicts(),
        k=request.k,
        threshold=request.threshold,
        mode=request.mode,
        candidate_k=request.candidate_k,
        rerank_backend=request.rerank_backend,
        where=request.where,
    )
    return StreamingResponse(
        _sse(events),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

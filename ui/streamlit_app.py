"""Streamlit chat UI.

    streamlit run ui/streamlit_app.py

Talks to the FastAPI service over HTTP rather than importing the pipeline directly.
That is the point of having built the API: the UI is a client like any other, and
nothing here can drift from what an external caller would get.

The UI consumes `/ask/stream`, so sources appear before the answer starts writing.

With `EMBED_API=1` (single-process hosts like Streamlit Community Cloud) the UI
indexes the corpus and starts the same FastAPI app on localhost in a background
thread, then talks to it over HTTP exactly as before.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx
import streamlit as st

API_URL = os.environ.get("RAG_API_URL", "http://localhost:8000")


@st.cache_resource(show_spinner="Indexing the documents (first visit only)...")
def start_embedded_api() -> str:
    """Run ingest + the API as child processes, once per server.

    Separate processes, not threads: the pipeline hangs when run on a thread
    inside Streamlit's process, and a subprocess is exactly what `make serve` runs.
    """
    root = Path(__file__).resolve().parent.parent
    # Idempotent (unchanged files are skipped); a failure shows up in /healthz.
    subprocess.run([sys.executable, "-m", "app.cli", "ingest"], cwd=root, check=False)
    subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.api:app", "--host", "127.0.0.1",
         "--port", "8000"],
        cwd=root,
    )
    for _ in range(100):
        try:
            httpx.get("http://127.0.0.1:8000/healthz", timeout=1.0)
            break
        except httpx.HTTPError:
            time.sleep(0.2)
    return "http://127.0.0.1:8000"


if os.environ.get("EMBED_API") == "1":
    API_URL = start_embedded_api()
REQUEST_TIMEOUT = httpx.Timeout(connect=5.0, read=180.0, write=10.0, pool=5.0)

st.set_page_config(page_title="Citewise", page_icon="📎", layout="centered")


def forwarded_headers() -> dict:
    """Pass the visitor's IP through, so the API rate-limits per visitor rather than
    treating every Streamlit user as one client."""
    ip = st.context.headers.get("X-Forwarded-For") or st.context.ip_address
    return {"X-Forwarded-For": ip} if ip else {}


# --- API client ------------------------------------------------------------


def api_get(path: str) -> dict | None:
    try:
        response = httpx.get(f"{API_URL}{path}", timeout=10.0)
        response.raise_for_status()
        return response.json()
    except Exception:
        return None


def stream_answer(
    question: str, history: list[dict], k: int, threshold: float,
    mode: str, rerank: str, source_filter: str | None = None,
):
    """Yield events from /ask/stream."""
    payload = {
        "question": question,
        "history": history,
        "k": k,
        "threshold": threshold,
        "mode": mode,
        "rerank_backend": rerank,
    }
    if source_filter:
        payload["where"] = {"source": source_filter}
    with httpx.stream(
        "POST", f"{API_URL}/ask/stream", json=payload, timeout=REQUEST_TIMEOUT,
        headers=forwarded_headers(),
    ) as response:
        if response.status_code == 429:
            raise RuntimeError("Too many questions in a minute - give it a moment.")
        response.raise_for_status()
        for line in response.iter_lines():
            if not line or not line.startswith("data: "):
                continue
            data = line[len("data: ") :]
            if data == "[DONE]":
                return
            try:
                yield json.loads(data)
            except json.JSONDecodeError:
                continue


# --- rendering helpers -----------------------------------------------------


def render_sources(sources: list[dict], search_query: str | None, rewritten: bool) -> None:
    if rewritten and search_query:
        st.caption(f"🔎 Searched for: *{search_query}*")
    if not sources:
        return
    label = f"📎 {len(sources)} source{'s' if len(sources) != 1 else ''}"
    with st.expander(label, expanded=False):
        for s in sources:
            found = "+".join(s.get("found_by") or []) or "?"
            st.markdown(
                f"**[{s['n']}] {s['source']}** · page {s['page']} · "
                f"score `{s['score']:.3f}` · found by `{found}`"
            )
            st.caption(s["excerpt"] + ("…" if len(s["excerpt"]) >= 400 else ""))
            st.divider()


# --- sidebar ---------------------------------------------------------------

with st.sidebar:
    st.title("📎 Citewise")
    st.caption("Answers with receipts.")

    health = api_get("/healthz")
    if health is None:
        st.error(
            f"Can't reach the API at `{API_URL}`.\n\n"
            "Start it with:\n```\nuvicorn app.api:app --reload --port 8000\n```"
        )
    elif health["status"] == "ok":
        st.success(f"Connected · {health['indexed_chunks']} chunks indexed")
    else:
        st.warning(f"Degraded: {health.get('detail')}")

    stats = api_get("/stats")
    if stats and stats["chunks"]:
        with st.expander(f"📄 {len(stats['sources'])} documents indexed"):
            for row in stats["sources"]:
                st.caption(f"{row['source']} — {row['chunks']} chunks")

    st.divider()
    st.subheader("Retrieval")
    mode = st.selectbox(
        "Mode", ["hybrid", "dense", "lexical"], index=0,
        help="hybrid = semantic + BM25 keyword, fused by rank. dense = semantic only "
             "(misses exact codes and identifiers). lexical = keyword only (misses "
             "paraphrases). Switch between them on the same question to see the "
             "difference on your own corpus.",
    )
    rerank = st.selectbox(
        "Reranker", ["llm", "none", "cross-encoder"], index=0,
        help="Reranking scores each (question, passage) pair directly, which is far "
             "more accurate than vector similarity but too slow for the whole corpus. "
             "Set 'none' to see what the retrieval stage alone produces.",
    )
    k = st.slider(
        "Chunks kept (k)", 1, 20, 5,
        help="How many passages survive reranking and reach the model.",
    )
    threshold = st.slider(
        "Relevance floor", 0.0, 1.0, 0.30, 0.05,
        help="Passages scoring below this are discarded. Higher = more refusals but "
             "cleaner citations. Lower = more answers but more risk of answering "
             "from noise.",
    )

    doc_names = [row["source"] for row in (stats or {}).get("sources", [])]
    source_filter = None
    if doc_names:
        picked = st.selectbox("Restrict to document", ["(all documents)"] + doc_names)
        source_filter = None if picked == "(all documents)" else picked

    st.divider()
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    if health:
        st.caption(f"Model: `{health['chat_model']}`")
        if health.get("lexical_documents"):
            st.caption(f"BM25 index: {health['lexical_documents']} chunks")
        elif mode == "hybrid":
            st.caption("⚠️ No BM25 index — hybrid is running dense-only. Re-run ingest.")


# --- chat ------------------------------------------------------------------

st.title("Ask the documents")

if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    st.caption(
        "Answers are grounded in your indexed documents and cited with [n] markers. "
        "If nothing relevant is found, the assistant says so instead of guessing. "
        "Follow-up questions work — the previous turns are used to resolve references."
    )
    st.caption(
        "The demo corpus is a fictional company, Halcyon Robotics: employee handbook, "
        "security policy, travel & expenses, IT runbook and a drone product FAQ. Try "
        "*\"What does ERR-4021 mean?\"*, *\"Can I carry over unused leave?\"* — "
        "or something it can't know, like *\"Who is the CEO?\"*"
    )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_sources(
                message.get("sources", []),
                message.get("search_query"),
                message.get("rewritten", False),
            )

if question := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # History excludes the message we just appended - the API takes prior turns
    # plus the current question separately.
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[:-1]
    ]

    with st.chat_message("assistant"):
        sources_slot = st.container()
        answer_slot = st.empty()
        answer = ""
        meta: dict = {}

        try:
            with st.spinner("Searching your documents..."):
                events = stream_answer(
                    question, history, k, threshold, mode, rerank, source_filter
                )
                first = next(events, None)

            if first is None:
                raise RuntimeError("no response from API")
            if first.get("event") == "error":
                raise RuntimeError(first.get("message", "unknown error"))

            meta = first
            with sources_slot:
                render_sources(
                    meta.get("sources", []),
                    meta.get("search_query"),
                    meta.get("rewritten", False),
                )

            for event in events:
                kind = event.get("event")
                if kind == "token":
                    answer += event["text"]
                    answer_slot.markdown(answer + "▌")
                elif kind == "done":
                    answer = event.get("answer") or answer
                    meta["usage"] = event.get("usage")
                elif kind == "error":
                    raise RuntimeError(event.get("message", "unknown error"))

            answer_slot.markdown(answer)
            usage = meta.get("usage") or {}
            if meta.get("cached"):
                st.caption("⚡ served from the semantic answer cache")
            elif usage.get("total_tokens"):
                st.caption(f"{usage['total_tokens']:,} tokens")

        except Exception as exc:  # noqa: BLE001
            answer = f"⚠️ Request failed: {exc}"
            answer_slot.error(answer)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": meta.get("sources", []),
            "search_query": meta.get("search_query"),
            "rewritten": meta.get("rewritten", False),
        }
    )

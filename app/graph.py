"""The RAG pipeline as an explicit LangGraph state machine.

Phase 3 shape:

    START -> rewrite -> retrieve -> grade -> (generate | no_context) -> END
                            ^
                            └── internally: dense + BM25 candidates
                                -> RRF fusion -> rerank  (see retrieval.py)

The `retrieve` node stays one node on purpose. Candidate generation, fusion and
reranking are all answering the same question - "which passages should the model
see?" - and none of them branches. Splitting them into graph nodes would add
ceremony without adding an inspectable decision. The genuine branch in this system
is still the one after `grade`.

Why a graph rather than a linear chain? Because the interesting behaviour of a RAG
system is conditional. "Did we retrieve anything worth answering from?" is a branch,
and the whole point of the honesty guarantee is that when the answer is no, the LLM
is never called at all. A graph makes that branch a first-class, inspectable thing
instead of an `if` buried inside a chain.

**Two graphs, shared nodes.** `build_graph()` is the full pipeline used by the
blocking `ask()`. `build_retrieval_graph()` stops after `grade`, and is what the
streaming path uses: it runs rewrite -> retrieve -> grade to completion, emits the
sources immediately, and only then streams generation token by token. The user sees
citations before the first word of the answer, which is the right ordering - it lets
them start judging trustworthiness while the answer is still being written.

Both graphs are assembled from the same node functions, so there is exactly one
implementation of each step.

**Phase 6 - agentic retrieval.** With `AGENTIC_RETRIES > 0`, a `grade` that keeps
nothing no longer ends the run: `requery` asks the LLM for a differently-worded
search and loops back to `retrieve`. The loop is bounded, and whatever it finds must
still clear the same grader, so it can recover recall without weakening honesty:

    grade --(nothing kept, attempts left)--> requery -> retrieve -> grade ...
"""

from __future__ import annotations

import logging
from typing import Any, Iterator, Literal, TypedDict

from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph

from app import cache as answer_cache
from app.config import get_settings
from app.prompts import (
    ANSWER_PROMPT,
    HYDE_PROMPT,
    NO_CONTEXT_MESSAGE,
    REQUERY_PROMPT,
    text_of,
)
from app.retrieval import ScoredChunk, format_context, grade, retrieve
from app.rewrite import rewrite_question

logger = logging.getLogger(__name__)


class RAGState(TypedDict, total=False):
    question: str            # what the user typed
    history: list[dict]      # [{"role": "user"|"assistant", "content": str}, ...]
    search_query: str        # standalone question actually used for retrieval
    k: int
    threshold: float
    mode: str                # "hybrid" | "dense" | "lexical"
    candidate_k: int
    rerank_backend: str      # "llm" | "cross-encoder" | "none"
    where: dict              # metadata filter, e.g. {"source": "policy.pdf"}
    retrieved: list[ScoredChunk]
    kept: list[ScoredChunk]
    answer: str
    grounded: bool
    attempts: int            # agentic re-queries made so far
    tried: list[str]         # search queries already attempted


def get_llm(**overrides: Any) -> ChatGoogleGenerativeAI:
    settings = get_settings()
    params: dict[str, Any] = {
        "model": settings.chat_model,
        "temperature": settings.temperature,
        "api_key": settings.require_api_key(),
    }
    params.update(overrides)
    return ChatGoogleGenerativeAI(**params)


# --- nodes -----------------------------------------------------------------


def rewrite_node(state: RAGState) -> RAGState:
    """Resolve a follow-up against the conversation into a standalone query.

    No history means no LLM call - a first question is already standalone.
    """
    question = state["question"]
    history = state.get("history") or []
    if not history:
        return {"search_query": question}
    return {"search_query": rewrite_question(question, history, get_llm())}


def _text(response) -> str:
    return text_of(response).strip()


def hyde_query(query: str) -> str | None:
    """A hypothetical answer to embed in place of the question (HYDE=true)."""
    if not get_settings().hyde:
        return None
    try:
        return _text((HYDE_PROMPT | get_llm()).invoke({"question": query})) or None
    except Exception as exc:  # noqa: BLE001 - HyDE is an optimisation, never a failure
        logger.warning("HyDE failed (%s); embedding the question itself", exc)
        return None


def retrieve_node(state: RAGState) -> RAGState:
    """Candidates -> fusion -> rerank. See retrieval.py for why each stage exists."""
    query = state.get("search_query") or state["question"]
    return {
        "tried": [*state.get("tried", []), query],
        "retrieved": retrieve(
            query,
            k=state.get("k"),
            mode=state.get("mode"),
            candidate_k=state.get("candidate_k"),
            rerank_backend=state.get("rerank_backend"),
            where=state.get("where"),
            dense_query=hyde_query(query),
        ),
    }


def requery_node(state: RAGState) -> RAGState:
    """Nothing survived grading: propose a differently-worded search and retry."""
    tried = state.get("tried", [])
    try:
        query = _text(
            (REQUERY_PROMPT | get_llm()).invoke(
                {"question": state["question"], "tried": " | ".join(tried)}
            )
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("requery failed (%s)", exc)
        query = ""
    # A failed or repeated proposal still consumes the attempt, so the loop ends.
    return {
        "search_query": query if query and query not in tried else tried[-1],
        "attempts": state.get("attempts", 0) + 1,
    }


def grade_node(state: RAGState) -> RAGState:
    return {
        "kept": grade(
            state.get("retrieved", []),
            threshold=state.get("threshold"),
            rerank_backend=state.get("rerank_backend"),
        )
    }


def generate_node(state: RAGState) -> RAGState:
    chain = ANSWER_PROMPT | get_llm()
    response = chain.invoke(
        {
            "context": format_context(state["kept"]),
            "question": state.get("search_query") or state["question"],
        }
    )
    return {"answer": text_of(response), "grounded": True}


def no_context_node(state: RAGState) -> RAGState:
    return {"answer": NO_CONTEXT_MESSAGE, "grounded": False}


def _can_requery(state: RAGState) -> bool:
    return state.get("attempts", 0) < get_settings().agentic_retries


def route_after_grade(state: RAGState) -> Literal["generate", "requery", "no_context"]:
    if state.get("kept"):
        return "generate"
    return "requery" if _can_requery(state) else "no_context"


def route_after_grade_retrieval_only(state: RAGState) -> Literal["requery", "__end__"]:
    return "requery" if not state.get("kept") and _can_requery(state) else END


# --- assembly --------------------------------------------------------------


def _add_retrieval_nodes(builder: StateGraph) -> StateGraph:
    """rewrite -> retrieve -> grade (+ the requery loop). Shared by both graphs."""
    builder.add_node("rewrite", rewrite_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("grade", grade_node)
    builder.add_node("requery", requery_node)
    builder.add_edge(START, "rewrite")
    builder.add_edge("rewrite", "retrieve")
    builder.add_edge("retrieve", "grade")
    builder.add_edge("requery", "retrieve")
    return builder


def build_graph():
    """Full pipeline, including generation."""
    builder = _add_retrieval_nodes(StateGraph(RAGState))
    builder.add_node("generate", generate_node)
    builder.add_node("no_context", no_context_node)
    builder.add_conditional_edges(
        "grade",
        route_after_grade,
        {"generate": "generate", "requery": "requery", "no_context": "no_context"},
    )
    builder.add_edge("generate", END)
    builder.add_edge("no_context", END)
    return builder.compile()


def build_retrieval_graph():
    """Everything up to and including grading. Used by the streaming path."""
    builder = _add_retrieval_nodes(StateGraph(RAGState))
    builder.add_conditional_edges(
        "grade", route_after_grade_retrieval_only, {"requery": "requery", END: END}
    )
    return builder.compile()


_graph = None
_retrieval_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def get_retrieval_graph():
    global _retrieval_graph
    if _retrieval_graph is None:
        _retrieval_graph = build_retrieval_graph()
    return _retrieval_graph


# --- result shaping --------------------------------------------------------


def serialise_sources(kept: list[ScoredChunk]) -> list[dict]:
    return [
        {
            "n": i,
            "source": c.source,
            "page": c.page,
            "score": round(c.score, 4),
            "found_by": c.found_by,
            "excerpt": c.document.page_content.strip()[:400],
        }
        for i, c in enumerate(kept, start=1)
    ]


def _initial_state(
    question: str,
    history: list[dict] | None = None,
    k: int | None = None,
    threshold: float | None = None,
    mode: str | None = None,
    candidate_k: int | None = None,
    rerank_backend: str | None = None,
    where: dict | None = None,
) -> RAGState:
    state: RAGState = {"question": question}
    for key, value in (
        ("history", history),
        ("k", k),
        ("threshold", threshold),
        ("mode", mode),
        ("candidate_k", candidate_k),
        ("rerank_backend", rerank_backend),
        ("where", where),
    ):
        if value is not None and value != []:
            state[key] = value
    return state


# --- public entry points ---------------------------------------------------


def usage_summary(handler: UsageMetadataCallbackHandler) -> dict[str, Any]:
    """Token totals across every LLM call in the request, plus a cost estimate."""
    totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for usage in handler.usage_metadata.values():
        for key in totals:
            totals[key] += usage.get(key, 0) or 0
    settings = get_settings()
    totals["cost_usd"] = round(
        (totals["input_tokens"] * settings.input_cost_per_mtok
         + totals["output_tokens"] * settings.output_cost_per_mtok) / 1e6,
        6,
    )
    return totals


def _cache_key(history, k, threshold, mode, candidate_k, rerank_backend, where) -> str | None:
    """None means "don't cache": with history the same words can mean another question."""
    if history:
        return None
    return answer_cache.params_key(
        k=k, threshold=threshold, mode=mode, candidate_k=candidate_k,
        rerank_backend=rerank_backend, where=where,
    )


def ask(
    question: str,
    history: list[dict] | None = None,
    k: int | None = None,
    threshold: float | None = None,
    mode: str | None = None,
    candidate_k: int | None = None,
    rerank_backend: str | None = None,
    where: dict | None = None,
) -> dict[str, Any]:
    """Run one question through the full graph and return a serialisable result."""
    key = _cache_key(history, k, threshold, mode, candidate_k, rerank_backend, where)
    if key and (hit := answer_cache.lookup(question, key)):
        return {**hit, "question": question, "cached": True,
                "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
                          "cost_usd": 0.0}}

    usage = UsageMetadataCallbackHandler()
    result = get_graph().invoke(
        _initial_state(
            question, history, k, threshold, mode, candidate_k, rerank_backend, where
        ),
        config={"callbacks": [usage]},
    )
    kept = result.get("kept", [])
    out = {
        "question": question,
        "search_query": result.get("search_query", question),
        "rewritten": result.get("search_query", question) != question,
        "answer": result.get("answer", ""),
        "grounded": result.get("grounded", False),
        "sources": serialise_sources(kept),
        "retrieved_count": len(result.get("retrieved", [])),
        "kept_count": len(kept),
        "mode": result.get("mode", get_settings().retrieval_mode),
        "cached": False,
        "usage": usage_summary(usage),
    }
    if key:
        answer_cache.store(question, key, out)
    return out


def ask_stream(
    question: str,
    history: list[dict] | None = None,
    k: int | None = None,
    threshold: float | None = None,
    mode: str | None = None,
    candidate_k: int | None = None,
    rerank_backend: str | None = None,
    where: dict | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield events for one question: `meta`, then `token`*, then `done`.

    Event shapes:
        {"event": "meta",  "search_query": str, "rewritten": bool,
         "grounded": bool, "sources": [...], "cached": bool}
        {"event": "token", "text": str}
        {"event": "done",  "answer": str, "grounded": bool, "usage": {...}}
        {"event": "error", "message": str}

    Sources are emitted before the first token deliberately - the reader can start
    assessing where an answer is coming from while it is still being written.
    """
    try:
        key = _cache_key(history, k, threshold, mode, candidate_k, rerank_backend, where)
        if key and (hit := answer_cache.lookup(question, key)):
            yield {"event": "meta", **{f: hit[f] for f in (
                "search_query", "rewritten", "grounded", "sources",
                "retrieved_count", "kept_count", "mode")}, "cached": True}
            yield {"event": "token", "text": hit["answer"]}
            yield {"event": "done", "answer": hit["answer"], "grounded": hit["grounded"],
                   "usage": usage_summary(UsageMetadataCallbackHandler())}
            return

        usage = UsageMetadataCallbackHandler()
        config = {"callbacks": [usage]}
        state = get_retrieval_graph().invoke(
            _initial_state(
                question, history, k, threshold, mode, candidate_k, rerank_backend, where
            ),
            config=config,
        )
        kept = state.get("kept", [])
        search_query = state.get("search_query", question)

        meta = {
            "search_query": search_query,
            "rewritten": search_query != question,
            "grounded": bool(kept),
            "sources": serialise_sources(kept),
            "retrieved_count": len(state.get("retrieved", [])),
            "kept_count": len(kept),
            "mode": state.get("mode", get_settings().retrieval_mode),
        }
        yield {"event": "meta", **meta, "cached": False}

        if not kept:
            # Same refusal as the graph's no_context node, streamed in one piece.
            answer = NO_CONTEXT_MESSAGE
            yield {"event": "token", "text": answer}
        else:
            chain = ANSWER_PROMPT | get_llm()
            parts: list[str] = []
            for chunk in chain.stream(
                {"context": format_context(kept), "question": search_query}, config=config
            ):
                text = text_of(chunk)
                if text:
                    parts.append(text)
                    yield {"event": "token", "text": text}
            answer = "".join(parts)

        summary = usage_summary(usage)
        yield {"event": "done", "answer": answer, "grounded": bool(kept), "usage": summary}
        if key:
            answer_cache.store(question, key, {**meta, "question": question,
                                               "answer": answer, "cached": False,
                                               "usage": summary})

    except Exception as exc:  # noqa: BLE001 - the stream must close cleanly
        yield {"event": "error", "message": str(exc)}

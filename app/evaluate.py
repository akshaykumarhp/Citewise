"""Evaluation harness (Phase 4): prove it works, and measure every change.

    python -m app.cli eval                      # full: retrieval + generation + judge
    python -m app.cli eval --retrieval-only     # cheap: no generation, no judge
    python -m app.cli eval --matrix             # A/B the retrieval knobs side by side
    python -m app.cli eval --gate               # exit 1 if below the success criteria

The golden set (`eval/golden.jsonl`) is one JSON object per line:

    {"id": "leave-1", "question": "...", "sources": ["handbook.md"], "answer": "..."}

`sources` lists the documents that contain the answer; an empty list marks a
deliberately unanswerable question, where the only correct behaviour is to abstain.

**Retrieval metrics** are computed on the ranked list *before* grading - grading is
a separate decision, measured separately by abstention:

- hit-rate@k  is a relevant chunk anywhere in the top k?
- MRR         1 / rank of the first relevant chunk
- nDCG@k      rank-discounted gain; relevance is binary (chunk from an expected source)
              and the ideal ordering is the same list sorted, since the total number
              of relevant chunks in the corpus is unknown

**Generation metrics** come from an LLM judge with a fixed rubric, scoring the
answer against the exact context the model saw (faithfulness), the question
(relevance) and the reference answer (correctness). Context precision/recall are
computed directly from source matches - no judge needed for those.

RAGAS computes the same family of metrics; it is not a dependency because the
judge here is ~20 lines, uses the key already configured, and its rubric is visible.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.prompts import ChatPromptTemplate

from app.config import PROJECT_ROOT, get_settings

GOLDEN_PATH = PROJECT_ROOT / "eval" / "golden.jsonl"
REPORTS_DIR = PROJECT_ROOT / "eval" / "reports"

# Success criteria from plan.md: S3 retrieval, S2 abstention, S1 groundedness.
GATES = {"hit_rate": 0.85, "abstention": 0.90, "faithfulness": 0.80}

JUDGE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You grade a question-answering system that must answer ONLY from the "
            "context it was given, citing it, and must decline when the context does "
            "not contain the answer.\n\n"
            "Score each field:\n"
            "- faithfulness (0, 0.5, 1): 1 if every factual claim in ANSWER is "
            "supported by CONTEXT; 0.5 if a minor claim is unsupported; 0 if any "
            "substantive claim is unsupported. A refusal is faithful.\n"
            "- relevance (0, 0.5, 1): does ANSWER address QUESTION?\n"
            "- correctness (0, 0.5, 1): does ANSWER agree with REFERENCE? If REFERENCE "
            "is 'UNANSWERABLE', 1 only if ANSWER declines.\n"
            "- refused (true/false): does ANSWER decline to answer, saying the "
            "documents don't contain the information?\n\n"
            'Output ONLY JSON: {{"faithfulness": x, "relevance": x, "correctness": x, '
            '"refused": true|false}}',
        ),
        (
            "human",
            "QUESTION: {question}\n\nREFERENCE: {reference}\n\nCONTEXT:\n{context}\n\n"
            "ANSWER:\n{answer}\n\nJSON:",
        ),
    ]
)


@dataclass
class EvalConfig:
    mode: str | None = None
    rerank_backend: str | None = None
    candidate_k: int | None = None
    k: int | None = None
    retrieval_only: bool = False

    @property
    def label(self) -> str:
        s = get_settings()
        return (
            f"{self.mode or s.retrieval_mode}/{self.rerank_backend or s.rerank_backend}"
            f"/cand{self.candidate_k or s.candidate_k}/k{self.k or s.retrieval_k}"
        )


def load_golden(path: Path = GOLDEN_PATH) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# --- metrics -----------------------------------------------------------------


def retrieval_metrics(ranked_sources: list[str], expected: set[str]) -> dict[str, float]:
    rels = [1 if s in expected else 0 for s in ranked_sources]
    first = next((i for i, r in enumerate(rels, start=1) if r), None)
    dcg = sum(r / math.log2(i + 1) for i, r in enumerate(rels, start=1))
    idcg = sum(r / math.log2(i + 1) for i, r in enumerate(sorted(rels, reverse=True), 1))
    return {
        "hit": float(first is not None),
        "rr": 1.0 / first if first else 0.0,
        "ndcg": dcg / idcg if idcg else 0.0,
    }


def _parse_judge(raw: Any) -> dict[str, Any]:
    from app.graph import _text

    text = _text(raw)
    start, end = text.find("{"), text.rfind("}")
    try:
        data = json.loads(text[start : end + 1])
    except (json.JSONDecodeError, ValueError):
        return {}
    return {
        "faithfulness": float(data.get("faithfulness", 0)),
        "relevance": float(data.get("relevance", 0)),
        "correctness": float(data.get("correctness", 0)),
        "refused": bool(data.get("refused", False)),
    }


def judge(question: str, reference: str, context: str, answer: str) -> dict[str, Any]:
    from app.graph import get_llm

    response = (JUDGE_PROMPT | get_llm()).invoke(
        {"question": question, "reference": reference, "context": context or "(none)",
         "answer": answer}
    )
    return _parse_judge(response)


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


# --- runner ------------------------------------------------------------------


def evaluate_one(item: dict, config: EvalConfig, usage) -> dict[str, Any]:
    from app.graph import _initial_state, get_graph, get_retrieval_graph
    from app.retrieval import format_context

    graph = get_retrieval_graph() if config.retrieval_only else get_graph()
    state = graph.invoke(
        _initial_state(
            item["question"], k=config.k, mode=config.mode,
            candidate_k=config.candidate_k, rerank_backend=config.rerank_backend,
        ),
        config={"callbacks": [usage]},
    )
    retrieved, kept = state.get("retrieved", []), state.get("kept", [])
    expected = set(item.get("sources") or [])
    row: dict[str, Any] = {
        "id": item.get("id"),
        "question": item["question"],
        "answerable": bool(expected),
        "retrieved": [c.source for c in retrieved],
        "kept": [c.source for c in kept],
    }

    if expected:
        row.update(retrieval_metrics(row["retrieved"], expected))
        row["context_recall"] = float(any(s in expected for s in row["kept"]))
        row["context_precision"] = (
            sum(s in expected for s in row["kept"]) / len(kept) if kept else 0.0
        )

    if config.retrieval_only:
        row["abstained"] = not kept
        return row

    row["answer"] = state.get("answer", "")
    verdict = judge(
        item["question"], item.get("answer") or "UNANSWERABLE",
        format_context(kept), row["answer"],
    ) if kept else {"faithfulness": 1.0, "relevance": 0.0,
                    "correctness": 0.0 if expected else 1.0, "refused": True}
    row.update(verdict)
    # The graph's refusal is structural; the judge catches "I don't know" answers
    # the model wrote itself from weakly relevant context.
    row["abstained"] = not kept or bool(verdict.get("refused"))
    return row


def run(config: EvalConfig, golden: list[dict] | None = None, progress=None) -> dict:
    settings = get_settings()
    # The cache would serve repeated questions across configs without retrieving.
    saved_cache = settings.answer_cache_similarity
    settings.answer_cache_similarity = 0.0
    say = progress or (lambda _m: None)
    golden = golden if golden is not None else load_golden()
    usage = UsageMetadataCallbackHandler()
    started = time.perf_counter()
    rows = []
    try:
        for i, item in enumerate(golden, start=1):
            say(f"[{i}/{len(golden)}] {item['question'][:70]}")
            rows.append(evaluate_one(item, config, usage))
    finally:
        settings.answer_cache_similarity = saved_cache

    answerable = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]
    summary = {
        "hit_rate": _mean([r["hit"] for r in answerable]),
        "mrr": _mean([r["rr"] for r in answerable]),
        "ndcg": _mean([r["ndcg"] for r in answerable]),
        "context_precision": _mean([r["context_precision"] for r in answerable]),
        "context_recall": _mean([r["context_recall"] for r in answerable]),
        "abstention": _mean([float(r["abstained"]) for r in unanswerable]),
        "false_refusals": _mean([float(r["abstained"]) for r in answerable]),
    }
    if not config.retrieval_only:
        summary |= {
            "faithfulness": _mean([r["faithfulness"] for r in rows if "faithfulness" in r]),
            "relevance": _mean([r["relevance"] for r in answerable if "relevance" in r]),
            "correctness": _mean([r["correctness"] for r in rows if "correctness" in r]),
        }

    from app.graph import usage_summary

    return {
        "label": config.label,
        "retrieval_only": config.retrieval_only,
        "questions": len(rows),
        "seconds": round(time.perf_counter() - started, 1),
        "summary": summary,
        "usage": usage_summary(usage),
        "settings": {
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "chunk_strategy": settings.chunk_strategy,
            "parent_chunk_size": settings.parent_chunk_size,
            "hyde": settings.hyde,
            "agentic_retries": settings.agentic_retries,
            "rrf_k": settings.rrf_k,
            "dense_weight": settings.dense_weight,
            "lexical_weight": settings.lexical_weight,
        },
        "rows": rows,
    }


MATRIX = [
    EvalConfig(mode=mode, rerank_backend=rerank, retrieval_only=True)
    for mode in ("hybrid", "dense", "lexical")
    for rerank in ("llm", "none")
]


def gate_failures(report: dict) -> list[str]:
    """Which success criteria the report misses. Unmeasured metrics don't fail."""
    failures = []
    for metric, floor in GATES.items():
        value = report["summary"].get(metric)
        if value is not None and value < floor:
            failures.append(f"{metric} {value:.2f} < {floor:.2f}")
    return failures


def save(report: dict, directory: Path = REPORTS_DIR) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    name = time.strftime("%Y%m%d-%H%M%S") + "_" + report["label"].replace("/", "_") + ".json"
    path = directory / name
    path.write_text(json.dumps(report, indent=2))
    return path

# Citewise — Project Plan

*Formerly "RAG Knowledge Assistant". Citewise: every answer comes with receipts.*

**Owner:** Akshay Kumar
**Started:** 2026-08-23
**Status:** Phases 1–5 implemented, Phase 6 partially (see below). Deployed as a Hugging Face Space.

---

## 1. Objective

Build a production-shaped **Retrieval-Augmented Generation (RAG) knowledge assistant** that
answers natural-language questions over a private document corpus (PDF / DOCX / Markdown /
TXT) with **grounded, cited answers** and an honest "I don't know" when the corpus doesn't
contain the answer.

This is a learning project, so the secondary objective matters as much as the first: by the
end, be able to explain and defend *every* design decision in a RAG system — chunking
strategy, embedding task types, retrieval scoring, prompt grounding, hallucination control,
and evaluation.

### Success criteria

| # | Criterion | How it's measured |
|---|-----------|-------------------|
| S1 | Answers are grounded | Every claim traceable to a cited chunk; manual spot-check on 20 questions |
| S2 | Refuses out-of-corpus questions | ≥ 90% correct abstention on a deliberate "unanswerable" question set |
| S3 | Retrieval actually retrieves | Hit-rate@5 ≥ 0.85 on a hand-labelled question→document set |
| S4 | Reproducible | `make ingest && make ask` works from a clean clone + API key |
| S5 | Deployable | `docker compose up` serves the API and UI; public demo on Hugging Face |

### Explicit non-goals (v1)

- Multi-tenancy / per-user auth
- Fine-tuning any model
- Real-time document sync (ingestion is an explicit, idempotent command)
- Multi-modal retrieval (images, tables-as-images) — Phase 6 stretch

---

## 2. Stack decisions (and why)

| Layer | Choice | Rationale |
|-------|--------|-----------|
| LLM | **Gemini** (`gemini-3.5-flash`) via Gemini Developer API | Free tier for learning; mirrors the Vertex AI/Gemini stack Akshay uses professionally, so the skills transfer directly. Swappable via `.env`. |
| Embeddings | **`gemini-embedding-001`**, 768 dims, task-typed | GA model, flexible output dims (128–3072). 768 keeps the index small and is the documented sweet spot. Uses `RETRIEVAL_DOCUMENT` / `RETRIEVAL_QUERY` task types — asymmetric embedding measurably beats naive symmetric embedding for QA. |
| Orchestration | **LangChain 1.x + LangGraph 1.x** | Industry-standard; LangGraph's explicit state machine makes the RAG pipeline inspectable rather than a magic chain. |
| Lexical index | **BM25** (`rank_bm25`), rebuilt from Chroma after each ingest, persisted as JSON | Covers dense retrieval's blind spot on exact identifiers. JSON rather than a pickle so it stays inspectable and survives library changes. |
| Reranker | **LLM (Gemini) by default**, optional local cross-encoder | A real cross-encoder is better and free per query, but pulls in torch (~2GB). Defaulting to an LLM reranker keeps the dependency footprint honest for a learning project; the cross-encoder is one env var away. |
| Vector store | **Chroma** (local, persisted to disk) → **pgvector** in Phase 5 | Chroma = zero infrastructure to start. pgvector later because Akshay already runs PostgreSQL and it's the realistic production answer. |
| Document parsing | **pypdf / python-docx**, hand-written loaders | `langchain-community` is being sunset. Owning the loaders means owning the metadata (page numbers → citations), which is the whole point. |
| API | **FastAPI** | Already in Akshay's stack. |
| UI | **Streamlit** | Fastest path to a demoable chat surface. |
| Config | **pydantic-settings** | Typed config, `.env`-driven, no magic constants scattered in code. |
| CLI | **Typer** | Ingest/ask/inspect without needing the server running. |

### Design rules held throughout

1. **No un-cited claims.** The prompt forbids answering from model priors; sources are
   returned as structured data, not just prose.
2. **Ingestion is idempotent.** A content-hash manifest means re-running `ingest` only
   re-embeds files that actually changed. Embedding calls cost money and time.
3. **Every layer is independently testable.** Chunking, retrieval, and generation can each
   be exercised without the others.
4. **Config over constants.** Model names, chunk sizes, and `k` live in `.env`.

---

## 3. Architecture

```
                        ┌─────────────────────────────────────┐
   data/raw/*.pdf ─────►│  INGESTION (offline, idempotent)    │
   *.docx *.md *.txt    │                                     │
                        │  load → chunk → embed → upsert      │
                        │   │      │       │        │         │
                        │   │      │       │        └─ Chroma │
                        │   │      │       └─ gemini-embedding│
                        │   │      │          (RETRIEVAL_DOC) │
                        │   │      └─ RecursiveCharacterSplit │
                        │   └─ pypdf/python-docx + page meta  │
                        │                                     │
                        │  manifest.json ← content hashes     │
                        └─────────────────────────────────────┘
                                          │
                                    data/chroma/
                                          │
                        ┌─────────────────▼───────────────────┐
   "can I carry   ────►│  QUERY (LangGraph state machine)    │
    them over?"        │                                     │
   + chat history      │  rewrite ─► retrieve ─► grade ─► gen │
                        │     │          │          │      │  │
                        │     │          │          │      └LLM
                        │     │          │          └ drop weak│
                        │     │          └ embed (RETRIEVAL_QRY)
                        │     └ resolve follow-up -> standalone
                        └─────────────────┬───────────────────┘
                                          │
                            { answer, sources[], grounded }
                                          │
                        ┌─────────────────▼───────────────────┐
                        │  SURFACES                           │
                        │   Typer CLI ──────┐                 │
                        │   FastAPI ◄── Streamlit UI (SSE)    │
                        └─────────────────────────────────────┘
```

### The LangGraph state machine (Phase 3)

```
START ──► rewrite ──► retrieve ──► grade ──┬──► generate ──► END
             │                             │
             │                             └──► no_context ──► END
             └─ skipped entirely when there is no history
```

`grade` is the honesty gate: if no retrieved chunk clears the relevance bar, the
graph routes to `no_context` and the assistant says it doesn't know **instead of calling the
LLM at all**. This is cheaper and structurally prevents a whole class of hallucination.

`rewrite` is the multi-turn fix. Retrieval is stateless — "can I carry them over?" embeds
to noise on its own. The rewriter resolves it against the conversation into a standalone
query *before* anything is embedded.

**Two compiled graphs, one set of nodes.** The full graph above serves blocking `ask()`.
A second graph stops after `grade` and serves the streaming path: it emits sources first,
then streams generation token by token, so the reader can judge the citations while the
answer is still being written.

**Inside `retrieve` (Phase 3)** — four stages, one node:

```
query ─► dense candidates (cosine, candidate_k) ─┐
      └► BM25 candidates  (keyword, candidate_k) ─┴─► RRF fusion ─► rerank ─► top k
```

Dense and lexical retrieval fail in opposite directions — dense can't find `ISO-27001`,
lexical can't match "vacation" to "annual leave" — so both run and their *ranks* are
fused (their scores are on incomparable scales). Reranking then scores each
(query, passage) pair directly, which is far more accurate than vector similarity and
far too slow for the whole corpus, but cheap over 20 candidates. Retrieval buys recall;
reranking buys precision.

---

## 4. Phased roadmap

Each phase ends with something that runs end to end. No phase leaves the repo broken.

### ✅ Phase 1 — Core RAG loop (CLI)
*Goal: ask a question about your own PDF and get a cited answer.*

- [x] Repo scaffold, typed config, `.env.example`
- [x] Loaders for PDF / DOCX / MD / TXT carrying `source` + `page` metadata
- [x] Recursive character chunking with configurable size/overlap
- [x] Gemini embeddings with correct task types + batching + retry
- [x] Chroma persistence with a content-hash manifest (idempotent re-ingest)
- [x] Retriever with score threshold
- [x] Grounded prompt + citation contract
- [x] LangGraph: retrieve → grade → generate / no_context
- [x] Typer CLI: `ingest`, `ask`, `stats`, `check`

**Learning targets:** why chunk size matters; why query and document embeddings use
different task types; what a vector store actually stores; why the prompt, not the model,
is what enforces grounding.

### ✅ Phase 2 — Serving layer
*Goal: it's a service, not a script.*

- [x] FastAPI app: `POST /ask`, `POST /ask/stream`, `POST /ingest`, `GET /stats`, `GET /healthz`
- [x] Pydantic request/response models — `sources` as structured objects
- [x] Streaming responses (SSE) so the UI shows tokens as they generate
- [x] Streamlit chat UI with an expandable "Sources" panel per answer
- [x] Conversation memory + **query rewriting**: rewrite "what about the second one?" into a
      standalone question before retrieval (this is the single biggest quality win in
      multi-turn RAG)

**Learning targets:** why retrieval is stateless and what that breaks in multi-turn
chat; why sources should stream before tokens; why a health endpoint should report
degraded rather than raise.

### ✅ Phase 3 — Retrieval quality
*Goal: stop losing the right chunk.*

- [x] **Hybrid search**: BM25 (keyword) + dense (semantic), fused with Reciprocal Rank Fusion.
      Dense retrieval alone fails badly on exact identifiers, product codes, and rare terms.
- [x] **Reranking** of the top ~20 down to the top ~5, with three backends: `llm`
      (one Gemini call, default), `cross-encoder` (local, optional torch), `none` (baseline)
- [x] Metadata filtering (by source), applied to both retrievers so they see the same slice
- [x] `cli debug` rebuilt as a per-stage inspector showing where each passage was
      promoted or dropped
- [ ] **Parent-document retrieval**: embed small chunks for precision, return the larger
      surrounding chunk for context — *deferred to Phase 4*
- [ ] Chunking experiment: fixed-size vs. recursive vs. semantic — *deferred to Phase 4*

**Deferred deliberately.** Both remaining items are tuning decisions, and tuning
without measurement is guessing. They move to Phase 4 so they can be evaluated
against the golden question set rather than eyeballed.

**Learning targets:** why dense and lexical retrieval fail in opposite directions;
why scores from different rankers can't be averaged but ranks can be fused; why
bi-encoders and cross-encoders exist and where each belongs; why retrieval is a
recall problem and reranking is a precision problem.

### ✅ Phase 4 — Evaluation
*Goal: prove it works; measure every change.*

- [x] Golden question set (40 Q/A pairs, incl. 10 deliberately unanswerable) over a bundled
      demo corpus (`corpus/`, fictional Halcyon Robotics)
- [x] Retrieval metrics: hit-rate@k, MRR, nDCG
- [x] Generation metrics: faithfulness, answer relevance, correctness, context precision/recall
      — the same family RAGAS computes, via an in-repo judge instead of the dependency
- [x] LLM-as-judge with a rubric for groundedness
- [x] `make eval` producing a comparable JSON report; regression gate in CI
- [x] Measure the Phase 3 knobs: `make eval-matrix` (mode × reranker); `--candidates`,
      RRF weights via `.env`
- [x] **Parent-document retrieval**: `PARENT_CHUNK_SIZE` — embed small chunks for
      precision, return the larger surrounding chunk for context
- [x] **Chunking experiment**: `CHUNK_STRATEGY=fixed|recursive`. Semantic chunking
      skipped — it needs `langchain-experimental` and an embedding call per sentence at
      ingest; add it if fixed vs recursive shows chunking is the bottleneck

### ✅ Phase 5 — Production hardening
*Goal: the thing you'd actually put in front of people.*

- [x] Swap Chroma → **pgvector** behind the existing `VectorStore` interface (`VECTOR_BACKEND`)
- [x] Dockerfile + `docker-compose.yml` (api + ui + postgres)
- [x] Structured logging, request IDs, per-request token/cost accounting
- [x] Rate limiting, input size caps, prompt-injection guardrails on ingested content,
      admin token on `/ingest`
- [x] Caching: query-embedding cache + semantic answer cache
- [x] GitHub Actions: lint, test, smoke, eval gate

### Phase 6 — Stretch
- [x] Agentic retrieval (LangGraph loop: grade → requery → retrieve), `AGENTIC_RETRIES`
- [ ] Table and figure extraction from PDFs — not done; needs a layout-aware parser
      (pdfplumber / a vision model). DOCX tables are already extracted
- [x] HyDE (`HYDE=true`)
- [ ] Per-user corpora + auth — not done; a v1 non-goal. `where` filtering is the hook

---

## 5. Known risks and how they're handled

| Risk | Handling |
|------|----------|
| **Hallucination** — model answers from priors | Grounding prompt + `grade_documents` gate + citation requirement + faithfulness eval in Phase 4 |
| **Bad chunking silently destroys quality** | Chunk size/overlap in config; Phase 3 measures alternatives against the golden set instead of guessing |
| **Dense retrieval misses exact terms** | Hybrid BM25 + dense in Phase 3 |
| **Free-tier rate limits during ingestion** | Batched embedding calls with exponential backoff; manifest prevents re-embedding unchanged files |
| **API costs creep** | Token accounting in Phase 5; flash-tier model by default |
| **Model IDs change** (Gemini moves fast) | All model names in `.env`; `cli.py check` lists what your key can actually reach |
| **Prompt injection via ingested docs** | Phase 5 guardrail: retrieved content is delimited and explicitly labelled untrusted data in the prompt |

---

## 6. Working agreement

- `PROJECT_NOTES.md` is the living state of the repo and is updated **every time code changes**.
- This `plan.md` changes only when the *plan* changes; checkboxes get ticked as phases land.
- Decisions with a "why" go in the Decision Log in `PROJECT_NOTES.md`, not in commit messages
  where they'll never be found again.

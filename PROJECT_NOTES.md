# PROJECT NOTES — Citewise (formerly RAG Knowledge Assistant), current state of the repo

> This file is the single source of truth for what exists right now.
> **It is updated every time code is written or modified.**
> For objectives and the roadmap, see [`plan.md`](./plan.md).

**Last updated:** 2026-10-23
**Current phase:** Phases 4–5 complete, Phase 6 partial — eval harness, hardening, deploy
**Measured (demo corpus, hybrid + LLM rerank, flash-lite):** hit-rate@5 1.00 · MRR 0.98 · nDCG 0.96 · abstention 0.90 · false refusals 0.10 — gates pass
**Verified:** 41/41 unit tests · 107/107 offline smoke checks (on Chroma *and* pgvector) · ruff clean · Docker image boots

---

## Quickstart

```bash
make setup                  # venv + deps + .env   (Python 3.11+)
# add GOOGLE_API_KEY to .env — free at https://aistudio.google.com/apikey
make check                  # confirm key + that the model names exist
cp ~/your-docs/*.pdf data/raw/
make ingest
```

Then pick a surface:

```bash
make ask Q="what is our retention policy?"   # one-shot CLI
make chat                                    # CLI with conversation memory
make serve                                   # API on :8000, docs at /docs
make ui                                      # Streamlit on :8501 (needs `make serve`)
```

No API key yet? Verify the whole system offline first — real loaders, real chunker,
real Chroma, real LangGraph, real FastAPI, fake model:

```bash
make smoke        # 107 checks, no network, no tokens
make test         # 40 unit tests
```

---

## File map

| Path | What it does | Why it exists |
|------|--------------|---------------|
| `app/config.py` | Typed `.env`-driven settings, cached singleton | Every tunable in one place; no magic constants |
| `app/loaders.py` | PDF / DOCX / MD / TXT → `Document` + content hashing | Owning the loader means owning page metadata → real citations |
| `app/chunking.py` | Recursive character splitting + deterministic chunk ids | Ids make re-ingest an upsert, not a duplicate-append |
| `app/embeddings.py` | Gemini embeddings: task types, normalisation, batching, backoff | The two easiest things to get wrong in RAG live here |
| `app/vectorstore.py` | Chroma access, stats, delete-by-source, reset | The only file that knows about Chroma — swap point for pgvector |
| `app/ingest.py` | discover → load → chunk → embed → upsert, with a manifest | Manifest = never re-embed an unchanged file |
| `app/retrieval.py` | The 4-stage pipeline: candidates → fuse → rerank → grade | Separating ranking from grading is what makes "I don't know" possible |
| `app/lexical.py` | **Phase 3** — BM25 index, tokeniser, rebuild-from-Chroma | Dense retrieval cannot find `ISO-27001`; BM25 always can |
| `app/fusion.py` | **Phase 3** — Reciprocal Rank Fusion | Cosine and BM25 scores are incomparable quantities; ranks are not |
| `app/rerank.py` | **Phase 3** — LLM / cross-encoder / none backends | Bi-encoders never see query and doc together; a reranker does |
| `app/rewrite.py` | **Phase 2** — history-aware query rewriting + guardrails | Retrieval is stateless; follow-ups embed to noise without this |
| `app/prompts.py` | Grounding contract, citation rules, refusal message | The prompt is where grounding is actually enforced |
| `app/graph.py` | LangGraph state machines + `ask()` / `ask_stream()` | Makes the conditional branch a first-class, inspectable thing |
| `app/schemas.py` | **Phase 2** — Pydantic request/response models | Sources stay structured objects, never a prose blob |
| `app/api.py` | **Phase 2** — FastAPI service incl. SSE streaming | The pipeline becomes a service, not a script |
| `app/cli.py` | `check` `ingest` `stats` `ask` `debug` `chat` `serve` | Exercise every layer without a browser |
| `ui/streamlit_app.py` | **Phase 2** — chat UI, streams from the API over HTTP | The UI is a client like any other, so it can't drift from the API |
| `scripts/smoke_test.py` | Full offline E2E with fake model + fake embeddings | Separates "wiring bug" from "credentials bug" |
| `tests/test_chunking.py` | Unit tests for chunking + grading | Pure functions, no tokens spent |
| `tests/test_rewrite.py` | **Phase 2** — rewriting guardrails and fallbacks | A bad rewrite is worse than no rewrite; every degenerate case must fall back |
| `tests/test_retrieval_phase3.py` | **Phase 3** — tokenising, BM25, RRF, rerank parsing, filters | These bugs degrade quality silently rather than raising |
| `app/evaluate.py` | **Phase 4** — golden-set runner, retrieval metrics, LLM judge, gates | Every knob is a guess until it is measured |
| `eval/golden.jsonl` | **Phase 4** — 40 questions (10 unanswerable) over `corpus/` | Abstention can only be measured with questions that *should* be refused |
| `corpus/` | **Phase 4** — fictional Halcyon Robotics docs (handbook, security, travel, IT, product FAQ) | A known corpus makes the eval reproducible and the demo self-contained |
| `app/cache.py` | **Phase 5** — semantic answer cache | Repeated questions shouldn't cost a rerank + generation |
| `Dockerfile`, `start.sh` | **Phase 5** — one image: all-in-one demo, or `api`/`ui` roles in compose | The HF Space and compose run the same artifact |
| `docker-compose.yml` | **Phase 5** — postgres+pgvector, API, UI | The production-shaped stack, one command |
| `scripts/deploy_hf.py` | **Phase 5** — push to a Hugging Face Docker Space | Key goes in as a Space secret, never in the repo |
| `.github/workflows/ci.yml` | **Phase 5** — lint, tests, smoke, eval gate (when `GOOGLE_API_KEY` secret is set) | Every push is checked offline; the eval gate needs a key |

---

## How a question flows through the system

```
question + chat history
   │
   ├─ REWRITE  ─ history empty? skip the LLM entirely, the question is already standalone
   │            ─ otherwise: resolve pronouns/references into a standalone query
   │
   ├─ CANDIDATES (candidate_k from each, default 20)
   │    ├─ dense:   embed query (RETRIEVAL_QUERY), normalise, Chroma cosine search
   │    └─ lexical: BM25 over the same chunks
   ├─ FUSE: reciprocal rank fusion — combines by RANK, since the two score
   │    scales are not comparable. Agreement between rankers scores highest.
   ├─ RERANK: score each (query, passage) pair directly, keep top k (default 5)
   ├─ GRADE: drop anything below the floor (RERANK_THRESHOLD or RELEVANCE_THRESHOLD)
   │
   ├── nothing survived? ──► no_context ──► honest refusal, LLM never called
   │
   └── something survived? ─► numbered [n] blocks ─► grounding prompt ─► Gemini
                           ─► answer with [n] citations + structured sources[]
```

Streaming takes the same path but emits `meta` (with sources) **before** the first
token, then `token` events, then `done`.

---

## HTTP API

`uvicorn app.api:app --reload` (or `make serve`) — interactive docs at `/docs`.

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/healthz` | Liveness + config sanity. Returns `degraded` with a reason rather than raising. |
| `GET` | `/stats` | Chunk counts per source document |
| `POST` | `/ingest` | Run ingestion. **Synchronous** — can take minutes on a large corpus. |
| `POST` | `/ask` | Blocking. Returns answer + structured sources. |
| `POST` | `/ask/stream` | SSE. Event order: one `meta`, then `token`*, then `done`. |

Request body for both ask endpoints:

```json
{
  "question": "can I carry them over?",
  "history": [
    {"role": "user", "content": "how many vacation days do I get?"},
    {"role": "assistant", "content": "Twenty per calendar year [1]."}
  ],
  "k": 5,
  "threshold": 0.3,
  "mode": "hybrid",
  "candidate_k": 20,
  "rerank_backend": "llm",
  "where": {"source": "handbook.pdf"}
}
```

`mode`, `rerank_backend` and `candidate_k` are per-request overrides — the point of
exposing them is A/B-ing the same question against different retrieval settings
without restarting anything.

`history` is what makes follow-ups work. Omit it and each question is treated as
standalone (which is correct for one-shot use, and skips a rewrite LLM call).

Response adds two fields worth watching while tuning:

- `search_query` — the standalone query actually embedded
- `rewritten` — whether rewriting changed the question
- `mode` — retrieval mode actually used
- `sources[].found_by` — which retrievers surfaced each chunk (`dense`, `lexical`, or both)

---

## CLI reference

| Command | Purpose |
|---------|---------|
| `python -m app.cli check` | Lists the models your key can actually reach; flags a mismatch with `.env` |
| `python -m app.cli ingest` | Index `data/raw/`. Skips unchanged files. |
| `python -m app.cli ingest --rebuild` | Drop the index and re-embed everything |
| `python -m app.cli stats` | Chunk counts per source document |
| `python -m app.cli ask "..."` | One grounded answer + a source table. `--mode`, `--rerank`, `--candidates`, `--source` override retrieval per call. |
| `python -m app.cli debug "..."` | **Per-stage retrieval inspector.** Shows both candidate lists, the fused ranking, and what reranking promoted or dropped. Run this first when an answer is wrong. |
| `python -m app.cli chat` | Interactive with memory. `clear` resets history, `--no-memory` disables it. |
| `python -m app.cli serve` | Run the FastAPI service |

---

## Decision log

Decisions with reasoning, so future-you doesn't relitigate them.

| # | Decision | Reasoning |
|---|----------|-----------|
| D1 | Own the document loaders instead of using `langchain-community` | `langchain-community` is being sunset upstream. More importantly, hand-written loaders carry page numbers into metadata, which is what turns "cited from policy.pdf" into "policy.pdf, p. 12". |
| D2 | Asymmetric embedding task types (`RETRIEVAL_DOCUMENT` / `RETRIEVAL_QUERY`) | A question and its answer share little vocabulary. Task-typed embeddings project both sides into a space where they land near each other. LangChain's vector stores don't pass task types through, hence the `GeminiRetrievalEmbeddings` wrapper. |
| D3 | Normalise embeddings manually at 768 dims | `gemini-embedding-001` only returns unit-normalised vectors at its native 3072 dims. At any truncated dimension the vectors are unnormalised, and cosine similarity on unnormalised vectors is quietly wrong — a silent quality bug. |
| D4 | Force Chroma to cosine (`hnsw:space: cosine`) | Chroma defaults to squared L2. Cosine is what the embedding model was trained against. |
| D5 | Content-hash manifest for ingestion | Embedding is the slow, rate-limited, metered step. Adding one PDF to a folder of fifty should cost one embedding run, not fifty. |
| D6 | Deterministic chunk ids `{source}::{page}::{start_index}` | Makes re-ingestion an upsert. Without stable ids, re-running ingest silently duplicates the corpus and skews retrieval. |
| D7 | Separate `retrieve` from `grade` | Top-k always returns k results — vector search has no concept of "nothing matches", it returns the k least-bad vectors it has. Grading converts "least bad" into "actually relevant". |
| D8 | Route to `no_context` **before** calling the LLM | Structurally prevents a whole class of hallucination, and it's free — no tokens spent on a question we can't answer. |
| D9 | LangGraph over a linear chain | The interesting behaviour is conditional. A graph makes the branch inspectable and gives Phases 3/6 obvious places to hang new nodes. |
| D10 | Model names in `.env`, plus a `check` command | Gemini model IDs change fast — the 2.x line was already superseded by 3.x during this project's setup. Hardcoding a model name is guaranteed future breakage. |
| D11 | Chroma now, pgvector in Phase 5 | Zero infrastructure to start. `vectorstore.py` is the only file that knows about Chroma, so the swap touches one file. |
| D12 | Offline smoke test with fake model + fake embeddings | When something breaks, the first question is "is my code wrong or my key wrong?" This answers it in five seconds without spending a token. |
| **D13** | **Rewrite the query before retrieval, not after** | Retrieval is stateless. "Can I carry them over?" contains no nouns to match against — it embeds to noise, and no amount of generation quality recovers from retrieving the wrong passages. Rewriting must happen *before* anything is embedded. |
| **D14** | **Skip the rewrite call entirely when history is empty** | A first question is already standalone. Rewriting it adds latency and a token cost and can only make it worse. |
| **D15** | **Every rewrite failure falls back to the original question** | Empty output, an over-long output (the model answered instead of rewriting), or a provider exception all fall back. A broken rewriter must degrade to Phase 1 behaviour, never break the query. |
| **D16** | **Two compiled graphs sharing one set of node functions** | The streaming path needs retrieval to finish before generation starts so sources can be emitted first. Rather than duplicating the pipeline, `build_retrieval_graph()` stops after `grade` and reuses the same nodes. One implementation of each step. |
| **D17** | **Sources stream before tokens** | The reader can start assessing where an answer comes from while it's still being written. Sources arriving after the answer is the wrong ordering for a system whose main claim is trustworthiness. |
| **D18** | **The Streamlit UI talks HTTP to the API, not Python to the pipeline** | Importing the pipeline directly would be simpler and would let the UI drift from what an external caller actually gets. As a client, the UI proves the API works. |
| **D19** | **`/healthz` returns `degraded` with a reason instead of raising** | A health endpoint that 500s tells an orchestrator the service is up but gives it nothing to act on. `degraded` + "index is empty — run /ingest" is actionable. |
| **D20** | **Assistant turns truncated to 500 chars in rewrite history** | The rewriter needs enough of the previous answer to resolve a reference, not the whole thing. Feeding it full answers makes it drift toward summarising the conversation instead of rewriting one question. |
| **D21** | **Hybrid (dense + BM25) rather than dense alone** | They fail in opposite directions. Dense can't retrieve `ISO-27001` — the embedding lands near "information security standard", which is semantically right and operationally useless when the user typed the code because they want the passage containing it. BM25 can't match "vacation" to "annual leave". Each covers the other's blind spot. |
| **D22** | **Fuse by rank (RRF), never by score** | Cosine scores (0–1) and BM25 scores (unbounded, corpus-dependent) are not comparable quantities. Per-query normalisation is unstable — a query where everything scores badly gets its noise stretched to fill the same range as a query with a perfect hit. RRF discards magnitudes and keeps only ranks, so neither ranker can dominate through score inflation, and agreement between them is rewarded. |
| **D23** | **Rebuild the BM25 index from Chroma after every ingest that wrote chunks** | BM25 needs whole-corpus term statistics, so unlike the vector store it can't be updated incrementally. Rebuilding is cheap (no embedding calls) and guarantees the lexical and dense indexes never disagree about what's in the corpus. |
| **D24** | **Persist the lexical index as JSON, not a pickle** | Inspectable, diffable, and immune to breaking when `rank_bm25` changes its internals. The BM25 object is reconstructed on load, which is milliseconds. |
| **D25** | **Tokenizer emits compound identifiers AND their parts** | `ISO-27001` yields `iso-27001`, `iso`, `27001`. Keeping only the compound means a query for `27001` misses; keeping only the parts means the exact code has no more weight than its fragments. Emitting both means the full code still scores highest. |
| **D26** | **Drop BM25 results with non-positive scores** | BM25 returns a score for every document, including ones sharing no terms with the query. Those aren't weak matches, they're non-matches, and feeding them into fusion pollutes the ranking. |
| **D27** | **LLM reranker as the default, cross-encoder optional** | A cross-encoder is genuinely better and free per query, but `sentence-transformers` pulls in torch (~2GB) — a big ask for a learning project. The LLM reranker scores every candidate in **one** call using the API key already present. `RERANK_BACKEND=cross-encoder` switches, and `none` exists as the baseline to prove reranking is actually helping. |
| **D28** | **Reranked and cosine scores get separate thresholds** | They're both 0–1 but they mean different things — one is pairwise relevance, the other is vector proximity. Sharing a threshold would silently change how strict the honesty gate is whenever the reranker is switched. |
| **D29** | **Retrieval metrics on the pre-grade ranking; grading measured as abstention** | Ranking and the honesty gate are separate decisions (D7). Scoring retrieval after grading would let a strict threshold masquerade as bad retrieval, and vice versa. |
| **D30** | **In-repo LLM judge instead of RAGAS** | Same metric family, ~20 lines, uses the configured key, rubric visible in the code. RAGAS would add a dependency tree and hide the prompt being trusted. |
| **D31** | **Ship a fictional demo corpus with the golden set** | An eval is only reproducible against a fixed corpus, and a public demo must not expose real documents. Synthetic content also guarantees the 10 unanswerable questions are genuinely unanswerable. |
| **D32** | **HyDE, parent-document and agentic re-query ship off by default** | Each costs tokens or latency, and none has earned its place on the golden set yet. They are knobs for `make eval`, not defaults by hope. |
| **D33** | **The requery loop is bounded and re-graded** | A loop that retries until something passes would launder weak context into an answer. Every retry must clear the same grader; the attempt counter ends it. |
| **D34** | **Cache only history-free questions, clear on ingest** | With history the same words can mean a different question. Any ingest can change any answer. |
| **D35** | **One Docker image for the Space and for compose** | The demo and the production-shaped stack run the same artifact; only the command and env differ. |
| **D38** | **`EMBED_API=1`: the UI launches ingest + the API as child processes** | Free single-process hosts (Streamlit Community Cloud) can't run two services. Child processes, not threads: the pipeline hung when run on a thread inside Streamlit's process, and a subprocess is exactly what `make serve` runs. The UI still talks HTTP (D18). |
| **D37** | **Default chat model is `gemini-3.1-flash-lite`** | On the free tier `gemini-3.5-flash` allows 20 requests/day — one eval run, or a few minutes of a public demo. Each question costs 2–3 calls (rerank + generate, + rewrite). Flash-lite's quota is what makes the demo viable; switch back via `CHAT_MODEL` on a paid key. |
| **D36** | **The UI forwards the visitor IP to the API** | Behind Streamlit every visitor is 127.0.0.1, so a per-IP limit would be one global limit. |

---

## Configuration reference

| Setting | Default | Effect of changing it |
|---------|---------|----------------------|
| `CHAT_MODEL` | `gemini-3.1-flash-lite` | Generation quality/cost. Verify with `cli check`. |
| `EMBEDDING_MODEL` | `gemini-embedding-001` | **Changing invalidates the index** — re-run `ingest --rebuild`. |
| `EMBEDDING_DIMENSIONS` | `768` | Index size vs. retrieval precision. **Also invalidates the index.** |
| `CHUNK_SIZE` | `1200` chars | Smaller = more precise retrieval, less context per hit. Larger = more context, mushier embeddings. Must stay under ~8000 chars (2048-token model limit). |
| `CHUNK_OVERLAP` | `200` chars | Insurance against facts that straddle a chunk boundary. |
| `RETRIEVAL_MODE` | `hybrid` | `hybrid` = dense + BM25 fused. `dense` = semantic only. `lexical` = keyword only. |
| `CANDIDATE_K` | `20` | Candidates from **each** retriever before fusion. Higher = better recall, more reranking cost. |
| `RETRIEVAL_K` | `5` | Chunks kept after reranking — what the model actually sees. |
| `RRF_K` | `60` | Fusion damping. Lower makes top ranks dominate more. |
| `DENSE_WEIGHT` / `LEXICAL_WEIGHT` | `1.0` / `1.0` | Tilt fusion toward semantic or keyword matching. |
| `RERANK_BACKEND` | `llm` | `llm` (one Gemini call), `cross-encoder` (needs torch), `none` (baseline). |
| `RERANK_THRESHOLD` | `0.3` | **The honesty dial when reranking is on.** 0.3 == 3/10 from the LLM reranker. |
| `RELEVANCE_THRESHOLD` | `0.35` | The honesty dial when `RERANK_BACKEND=none`, on the cosine scale. |
| `RAG_API_URL` (env) | `http://localhost:8000` | Where the Streamlit UI looks for the API. |
| `DATA_DIR` | `data/raw` | Folder `ingest` reads. `corpus` for the demo/eval corpus. |
| `CHUNK_STRATEGY` | `recursive` | `fixed` is the blind-slicing baseline. **Re-ingest after changing.** |
| `PARENT_CHUNK_SIZE` | `0` (off) | >0: embed `CHUNK_SIZE` children, show the model the parent. **Re-ingest.** |
| `HYDE` | `false` | Embed a hypothetical answer for dense search. +1 LLM call per question. |
| `AGENTIC_RETRIES` | `0` | Re-query attempts before refusing. +1 LLM call per attempt, only on would-be refusals. |
| `VECTOR_BACKEND` / `DATABASE_URL` | `chroma` / – | `pgvector` + `postgresql+psycopg://...`. pgvector fixes the vector width at table creation: changing `EMBEDDING_DIMENSIONS` means dropping the `langchain_pg_*` tables. |
| `RATE_LIMIT_PER_MINUTE` | `30` | Per client IP on token-spending endpoints. 0 = off. |
| `ADMIN_TOKEN` | – | When set, `/ingest` needs `Authorization: Bearer <token>`. |
| `ALLOWED_ORIGINS` | `*` | CORS. Narrow it anywhere public. |
| `ANSWER_CACHE_SIMILARITY` | `0.97` | Cosine above which a cached answer is reused. 0 = off. |
| `INPUT_COST_PER_MTOK` / `OUTPUT_COST_PER_MTOK` | `0` | USD per million tokens for the `usage.cost_usd` estimate. |

Also tunable in code: `app/rewrite.py::MAX_HISTORY_TURNS` (default 6) — how many
prior turns the rewriter sees.

---

## Known limitations (after Phase 5)

Deliberate and scheduled, not oversights.

- **The golden set is small and synthetic.** 40 questions over a 5-document demo
  corpus catches regressions; it does not prove quality on *your* documents. Write a
  golden set for your corpus (`--golden path.jsonl`) before trusting the numbers there.
- **The judge is the same model family as the generator.** Self-grading is biased toward
  leniency. Point the judge at a different/larger model if the scores drive decisions.
- **Caches and rate limits are per process.** Fine for one replica; Redis when scaling out.
- **Removed files stay indexed.** `ingest` upserts and replaces changed files but never
  deletes chunks for files that disappeared from `DATA_DIR`; use `--rebuild`.
- **The LLM reranker costs one extra call per query.** Roughly doubles latency on a
  short answer. `RERANK_BACKEND=cross-encoder` removes the per-query cost at the price
  of a 2GB dependency.
- **BM25 index lives in memory.** Fine to a few hundred thousand chunks; beyond that it
  wants a real search backend (Elasticsearch, or Postgres full-text alongside pgvector).
- **`/ingest` blocks.** Synchronous, no job id, no progress endpoint. Admin-only and
  idempotent, which is enough for a single-corpus service; a job queue for multi-tenant.
- **Conversation history lives in the client.** The API is stateless and takes
  history per request. That's a defensible design, but it means no server-side
  session, no persistence, and a long conversation grows the request payload.
- **No per-user auth.** `/ask` is public by design (rate-limited); per-user corpora are
  a non-goal for v1.
- **Thresholds are measured only on the demo corpus.** Re-tune `RERANK_THRESHOLD`
  with `make eval` on your own golden set.
- **Scanned/image-only PDFs yield nothing.** No OCR → Phase 6.
- **Cost is an estimate.** Prices are configured, not fetched; the default is 0 (tokens only).
- **Prompt-injection defence is heuristic.** Delimiters are neutralised and
  instruction-shaped text is flagged in-band, but a paraphrased injection gets through
  to a model told to treat it as data. That rule is the real defence; the regex is a tripwire.
- **No PDF table extraction** (pypdf flattens tables to text) and **no OCR** → Phase 6.

---

## Troubleshooting

| Symptom | Likely cause | What to do |
|---------|--------------|------------|
| `GOOGLE_API_KEY is not set` | No `.env`, or key not filled in | `cp .env.example .env`, add the key |
| `check` says a model isn't in your list | Model ID changed or isn't available to your key | Set `CHAT_MODEL` / `EMBEDDING_MODEL` to one `check` printed |
| "I couldn't find anything relevant" for something that IS in your docs | Threshold too strict, or bad retrieval | `cli debug "question"` to see actual scores, then lower `RELEVANCE_THRESHOLD` |
| Answers cite the wrong passages | Chunk size mismatched to your documents | Try `CHUNK_SIZE=600` (dense reference docs) or `2000` (flowing prose), then `ingest --rebuild` |
| Ingest fails with 429 / RESOURCE_EXHAUSTED | Free-tier per-minute quota | It already retries with backoff; if it exhausts retries, re-run — the manifest resumes where it stopped |
| Assistant can't find an exact code / ID / product number | Lexical index missing, so hybrid is silently dense-only | `cli stats` shows the BM25 count; if zero, run `ingest`. Confirm with `cli debug "<the code>"` — BM25 should rank it #1 |
| Hybrid and dense give identical results | No BM25 index, or the query has no rare terms | Check `cli stats`; on a purely conceptual question the two legitimately agree |
| Answers got slower after Phase 3 | The LLM reranker adds one call per query | `--rerank none` to confirm that's the cause, then either accept it or switch to `cross-encoder` |
| Reranking seems to make things worse | Possible — that's why `none` exists | A/B it: `cli ask "..." --rerank none` vs `--rerank llm` on the same question, and settle it properly in Phase 4 |
| Follow-up question retrieves nonsense | Rewriting produced a bad standalone query | Check `search_query` in the response (the CLI prints "searched for:" when it rewrites); if wrong, `clear` the history or use `--no-memory` |
| Streamlit shows "Can't reach the API" | API not running, or on a different port | Start `make serve`; if not on :8000, set `RAG_API_URL` |
| UI answers differ from CLI answers | Different `k` / threshold — the UI sliders override `.env` | Match the sliders to your `.env` values |
| `stats` shows duplicate-looking counts | Ingested before deterministic ids existed | `make rebuild` |
| PDF ingests but produces no chunks | Scanned / image-only PDF | Needs OCR (Phase 6); for now, use a text-based PDF |

---

## Changelog

### 2026-10-23 — Phases 4–6: evaluation, hardening, deploy; renamed to Citewise
- **Phase 4.** `app/evaluate.py` + `cli eval`: hit-rate@k, MRR, nDCG on the pre-grade
  ranking; abstention and false-refusal rates; context precision/recall from source
  matches; an LLM judge for faithfulness, relevance and correctness. `--retrieval-only`,
  `--matrix`, `--gate`. JSON reports in `eval/reports/`. 40-question golden set over a
  new bundled demo corpus. `CHUNK_STRATEGY` and `PARENT_CHUNK_SIZE` (parent-document
  retrieval, with sibling de-duplication) for the chunking experiments.
- **Phase 5.** `VECTOR_BACKEND=pgvector`; `vectorstore.py` gained `all_chunks()` so
  stats, deletes and the BM25 rebuild are backend-agnostic. JSON logs with request ids;
  per-IP rate limiting; `ADMIN_TOKEN` and a project-root path check on `/ingest`;
  history/turn size caps; `ALLOWED_ORIGINS`; token + cost accounting (`usage` on
  responses and the stream's `done` event); query-embedding cache; semantic answer
  cache; delimiter neutralising + injection flagging in `format_context`. Dockerfile,
  compose (pgvector), Hugging Face deploy script, GitHub Actions.
- **Phase 6.** `AGENTIC_RETRIES` adds a bounded `requery` loop to both graphs; `HYDE`
  embeds a hypothetical answer for the dense retriever only.
- Gemini 3 returns content as a list of parts: `prompts.text_of()` now normalises it
  everywhere. Before, the rewriter silently fell back and generation would have failed.
- UI rebranded; forwards the visitor IP so rate limits are per visitor, shows tokens
  and cache hits.
- **Verified:** 107 smoke checks on Chroma and on a real pgvector container; cache hits
  skip every LLM call; requery recovers an off-vocabulary question and is bounded;
  HyDE only fires when enabled; `/ingest` 401s without the token and 400s outside the
  project; the 3rd request in a 2/min window gets 429.

### 2026-08-29 — Phase 3 (hybrid retrieval + reranking)
- `app/lexical.py`: BM25 index with an identifier-preserving tokeniser, JSON
  persistence, mtime-aware caching, and rebuild-from-Chroma. Non-positive-scoring
  documents are dropped rather than returned as weak matches.
- `app/fusion.py`: reciprocal rank fusion with per-ranker weights, keeping rank and
  raw-score provenance for every fused result.
- `app/rerank.py`: three backends (`llm` scoring all candidates in one call,
  `cross-encoder` via optional sentence-transformers, `none`). Tolerant JSON parsing;
  every failure path falls back to fusion order.
- `app/retrieval.py`: rewritten as candidates → fuse → rerank → grade, with metadata
  filtering applied to **both** retrievers so they see the same corpus slice.
  `ScoredChunk` now carries `found_by` and per-stage scores.
- `app/ingest.py`: rebuilds the BM25 index after any ingest that wrote chunks.
- `app/graph.py`, `schemas.py`, `api.py`, `cli.py`, `ui/`: `mode`, `candidate_k`,
  `rerank_backend` and `where` threaded through every surface as per-request overrides.
- `cli debug` rebuilt as a per-stage inspector: both candidate lists side by side, the
  fused ranking with provenance, and arrows showing what reranking promoted or dropped.
- `/healthz` now warns when `RETRIEVAL_MODE=hybrid` but no lexical index exists —
  otherwise hybrid silently degrades to dense-only.
- `tests/test_retrieval_phase3.py`: 24 tests. `scripts/smoke_test.py`: extended from
  52 to 72 checks.
- **Verified:** BM25 retrieves an exact identifier that fusion then ranks; `dense` and
  `lexical` modes are distinguishable by provenance; `rerank=none` makes zero LLM calls
  while `rerank=llm` makes exactly one and demonstrably reorders; metadata filters apply
  to both retrievers; invalid `mode` → 422.

### 2026-08-23 — Phase 2 (serving layer, multi-turn, streaming)
- `app/rewrite.py`: history-aware query rewriting. Skips the LLM when history is
  empty; falls back to the original question on empty output, answer-shaped output,
  or a provider exception. Assistant turns truncated to 500 chars, history capped at
  6 turns.
- `app/graph.py`: added a `rewrite` node ahead of `retrieve`; `RAGState` now carries
  `history` and `search_query`. Added `build_retrieval_graph()` (stops after `grade`)
  and `ask_stream()` yielding `meta` → `token`* → `done`.
- `app/schemas.py`: Pydantic models for every endpoint; `Source` keeps `n`, `source`,
  `page`, `score`, `excerpt` as fields.
- `app/api.py`: FastAPI with `/healthz`, `/stats`, `/ingest`, `/ask`, `/ask/stream`
  (SSE). Missing API key surfaces as 503, not a stack trace.
- `ui/streamlit_app.py`: chat UI consuming `/ask/stream` over HTTP, per-answer
  sources expander, sidebar with index stats and live `k` / threshold sliders.
- `app/cli.py`: `chat` now keeps conversation history (`clear` to reset,
  `--no-memory` to disable); new `serve` command.
- `tests/test_rewrite.py`: 9 tests covering rewrite guardrails and history formatting.
- `scripts/smoke_test.py`: extended from 22 to 52 checks — rewriting behaviour,
  streaming event order, and every API endpoint via `TestClient`.
- **Verified:** rewrite fires only with history and its output is what gets embedded;
  streamed tokens reassemble to the full answer; SSE emits sources before tokens and
  terminates with `[DONE]`; empty question → 422; missing key → 503.

### 2026-08-23 — Phase 1 (core RAG loop)
- Initial scaffold: `config`, `loaders`, `chunking`, `embeddings`, `vectorstore`,
  `ingest`, `retrieval`, `prompts`, `graph`, `cli`
- Pinned dependency versions verified against PyPI (LangChain 1.3.16, LangGraph 1.2.11,
  langchain-google-genai 4.3.5, chromadb 1.5.9)
- Model defaults set to `gemini-3.5-flash` / `gemini-embedding-001` after confirming
  the current Gemini model lineup
- Added `scripts/smoke_test.py` — 22 offline checks; `tests/test_chunking.py` — 7 tests
- **Verified:** loaders extract per-page PDF metadata and DOCX tables; re-ingest skips
  unchanged files; changed files re-embed; refusal path returns zero sources

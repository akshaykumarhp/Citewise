# 📎 Citewise — Grounded RAG Knowledge Assistant

*Answers with receipts.*

Ask questions about your own documents and get answers that are **grounded in them,
cited to the page, and honest when the answer isn't there.**

Built with LangChain 1.x + LangGraph, Gemini, Chroma / pgvector, FastAPI and Streamlit.

```
$ python -m app.cli ask "how many vacation days do I accrue?"

╭─ Answer ─────────────────────────────────────────────────────────────╮
│ Full-time employees accrue 20 paid vacation days per calendar year,  │
│ accruing monthly [1]. Up to 5 unused days may be carried into the    │
│ following year [1].                                                  │
╰──────────────────────────────────────────────────────────────────────╯
      Sources
  # Document            Page   Score
  1 employee_handbook.pdf   14   0.812
```

## Why this exists

An LLM asked about your internal documents will answer confidently from its
pretraining. RAG fixes that by retrieving the actual passages first — but only if
you get chunking, embedding task types, retrieval scoring, and the grounding
prompt right. This project is built to make each of those visible and tunable
rather than hidden behind a framework abstraction.

## Quickstart

Requires **Python 3.11+**.

```bash
make setup                                  # venv + deps + .env
# add GOOGLE_API_KEY to .env (free: https://aistudio.google.com/apikey)
make check                                  # confirm key + model names
cp ~/your-docs/*.pdf data/raw/
make ingest
```

Then pick a surface:

```bash
make ask Q="what is our data retention policy?"   # one-shot
make chat                                          # CLI, remembers the conversation
make serve                                         # API on :8000 (docs at /docs)
make ui                                            # Streamlit chat on :8501
```

No documents handy? `make demo` indexes the bundled demo corpus (`corpus/`, a fictional
robotics company) — the same one the evaluation set and the live demo use.

Kick the tyres with no API key at all:

```bash
make smoke        # 107 offline end-to-end checks, no network, no tokens
make test         # 40 unit tests
```

## Run it as a service

```bash
make up           # docker compose: postgres+pgvector, API on :8000, UI on :8501
docker compose exec api python -m app.cli ingest
```

Deploy for free on [Streamlit Community Cloud](https://share.streamlit.io): new app →
this repo, branch `main`, main file `ui/streamlit_app.py`, Python 3.12, and these secrets:

```toml
GOOGLE_API_KEY = "..."
DATA_DIR = "corpus"
EMBED_API = "1"     # the UI starts the API itself - one process on that host
```

Or deploy the all-in-one image to a Hugging Face Space (Docker Spaces need HF PRO) (the API key goes in as a Space
secret, never into the repo):

```bash
HF_TOKEN=hf_... make deploy SPACE=<you>/citewise
```

## Measure it

```bash
make eval-cheap   # retrieval metrics + abstention, no generation (cheap)
make eval         # + generation and an LLM judge; fails below the success criteria
make eval-matrix  # hybrid/dense/lexical × llm/none reranking, side by side
```

Reports land in `eval/reports/` as JSON, so two runs can be diffed.

## What it does

- **Ingests** PDF, DOCX, Markdown and TXT, carrying page numbers into metadata
- **Chunks** recursively on natural boundaries, with configurable size and overlap
- **Embeds** with Gemini using asymmetric task types, unit-normalised, batched, retrying
- **Indexes** into Chroma with deterministic ids, so re-ingest upserts instead of duplicating
- **Skips unchanged files** via a content-hash manifest — no wasted embedding calls
- **Retrieves** top-k by cosine, then **grades** against a relevance floor
- **Refuses** — if nothing clears the floor, it says so and never calls the LLM
- **Cites** every claim with `[n]` markers backed by structured source records
- **Handles follow-ups** — "can I carry them over?" is rewritten into a standalone
  query against the conversation *before* retrieval, because retrieval is stateless
- **Searches two ways at once** — semantic (dense) and keyword (BM25), fused by rank.
  Dense alone can't find `ISO-27001`; keyword alone can't match "vacation" to
  "annual leave"
- **Reranks** the top 20 candidates down to the 5 that actually answer the question,
  scoring each (question, passage) pair directly
- **Serves** over FastAPI with SSE streaming, and ships a Streamlit chat UI that
  shows sources before the answer starts writing
- **Measures itself** — a 40-question golden set (10 deliberately unanswerable) scored
  for hit-rate, MRR, nDCG, abstention, and judge-rated faithfulness; a CI gate
- **Hardened for the public** — rate limits, request ids and JSON logs, token and cost
  accounting, an admin-only `/ingest`, prompt-injection guardrails on retrieved text,
  and a semantic answer cache
- **Optional extras, off until the eval says they help** — parent-document retrieval,
  HyDE, and an agentic re-query loop before refusing

## Documentation

- [`plan.md`](./plan.md) — objectives, architecture, and the Phase 1–6 roadmap
- [`PROJECT_NOTES.md`](./PROJECT_NOTES.md) — current state, file map, decision log,
  configuration reference, troubleshooting

## Debugging retrieval

When an answer is wrong, it is usually retrieval, not generation. `make debug` shows
every stage — both candidate lists side by side, the fused ranking, and what
reranking promoted or dropped:

```bash
make debug Q="what is our data retention policy?"
```

You can also A/B retrieval settings on the same question without restarting anything:

```bash
python -m app.cli ask "ISO-27001 status?" --mode dense    # semantic only
python -m app.cli ask "ISO-27001 status?" --mode lexical  # keyword only
python -m app.cli ask "ISO-27001 status?" --rerank none   # no reranking
```

## Status

Phases 1–5 are complete: the core RAG loop, the serving layer, hybrid retrieval with
reranking, the evaluation harness, and production hardening. From Phase 6, agentic
re-querying and HyDE are in; PDF table extraction and per-user corpora are not.
See `plan.md`.

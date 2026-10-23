"""Typed, .env-driven configuration.

Everything tunable lives here so that no magic constants end up scattered through
the pipeline. Change a value in .env, not in the code.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- credentials -------------------------------------------------------
    google_api_key: str = Field(default="", description="Gemini Developer API key")

    # --- models ------------------------------------------------------------
    # Gemini model IDs move fast. Run `python -m app.cli check` to list the
    # models your key can actually reach today.
    chat_model: str = "gemini-3.1-flash-lite"
    embedding_model: str = "gemini-embedding-001"
    # gemini-embedding-001 supports 128-3072; 768 is the documented sweet spot
    # and keeps the on-disk index small.
    embedding_dimensions: int = 768
    temperature: float = 0.0

    # --- chunking ----------------------------------------------------------
    # gemini-embedding-001 caps input at 2048 tokens. 1200 characters is
    # ~300 tokens, comfortably inside the limit with room for long words.
    chunk_size: int = 1200
    chunk_overlap: int = 200

    # --- retrieval ---------------------------------------------------------
    retrieval_k: int = 5
    # Chroma cosine "relevance score" is 1 - distance, so higher is better.
    # Below this, a chunk is treated as noise and dropped by the grader.
    relevance_threshold: float = 0.35

    # "hybrid" (dense + BM25, fused) | "dense" | "lexical"
    retrieval_mode: str = "hybrid"
    # Candidates fetched from EACH retriever before fusion and reranking.
    # Retrieval is a recall problem: fetch wide here, narrow later.
    candidate_k: int = 20
    rrf_k: int = 60
    dense_weight: float = 1.0
    lexical_weight: float = 1.0

    # --- reranking ---------------------------------------------------------
    # "llm" (one Gemini call, no extra deps) | "cross-encoder" (needs torch) | "none"
    rerank_backend: str = "llm"
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # Reranked scores are 0-1 but mean something different from cosine relevance,
    # so they get their own floor. 0.3 == a score of 3/10 from the LLM reranker.
    rerank_threshold: float = 0.3

    # --- chunking strategy (Phase 4 experiments) ----------------------------
    # "recursive" (natural boundaries) | "fixed" (blind fixed-size slices)
    chunk_strategy: str = "recursive"
    # Parent-document retrieval: >0 embeds small chunks but hands the model the
    # surrounding parent chunk of this many characters. 0 = off.
    parent_chunk_size: int = 0

    # --- Phase 6 retrieval extras (off until the eval says they help) -------
    # HyDE: embed a hypothetical answer instead of the question for dense search.
    hyde: bool = False
    # Agentic retrieval: when grading keeps nothing, let the LLM propose a new
    # search query and retry up to this many times before refusing.
    agentic_retries: int = 0

    # --- vector store backend (Phase 5) -------------------------------------
    # "chroma" (local directory) | "pgvector" (needs DATABASE_URL)
    vector_backend: str = "chroma"
    database_url: str = ""

    # --- serving / hardening (Phase 5) --------------------------------------
    # Comma-separated. "*" is fine on localhost, narrow it anywhere public.
    allowed_origins: str = "*"
    # Requests per minute per client IP on /ask*, /ingest. 0 = unlimited.
    rate_limit_per_minute: int = 30
    # When set, /ingest requires `Authorization: Bearer <token>`.
    admin_token: str = ""
    # Semantic answer cache: reuse an answer when a new standalone question embeds
    # this close to a cached one. 0 disables.
    answer_cache_similarity: float = 0.97
    # USD per million tokens, for the cost estimate in responses. 0 = tokens only.
    input_cost_per_mtok: float = 0.0
    output_cost_per_mtok: float = 0.0

    # --- paths -------------------------------------------------------------
    data_dir: Path = PROJECT_ROOT / "data" / "raw"
    chroma_dir: Path = PROJECT_ROOT / "data" / "chroma"
    manifest_path: Path = PROJECT_ROOT / "data" / "manifest.json"
    lexical_index_path: Path = PROJECT_ROOT / "data" / "lexical_index.json"
    collection_name: str = "knowledge_base"

    # --- ingestion ---------------------------------------------------------
    embed_batch_size: int = 50
    max_retries: int = 5

    def require_api_key(self) -> str:
        if not self.google_api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY is not set. Copy .env.example to .env and add your key "
                "from https://aistudio.google.com/apikey"
            )
        return self.google_api_key


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

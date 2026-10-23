"""API request/response models.

Sources are structured objects, never a prose blob. The whole value of citations is
that a client can render them, link them, count them and check them - which is only
possible if `source`, `page` and `score` survive as fields.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., max_length=8000)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)
    history: list[Turn] = Field(
        default_factory=list,
        max_length=40,
        description="Prior turns, oldest first. Enables follow-up questions via "
        "query rewriting.",
    )
    k: int | None = Field(default=None, ge=1, le=50, description="Chunks kept after reranking")
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    mode: Literal["hybrid", "dense", "lexical"] | None = Field(
        default=None,
        description="hybrid = dense + BM25 fused with RRF (default). dense = "
        "semantic only. lexical = keyword only. Useful for A/B-ing retrieval.",
    )
    candidate_k: int | None = Field(
        default=None, ge=1, le=200,
        description="Candidates fetched from EACH retriever before fusion and reranking",
    )
    rerank_backend: Literal["llm", "cross-encoder", "none"] | None = Field(
        default=None, description="Set 'none' to measure the pipeline without reranking"
    )
    where: dict | None = Field(
        default=None,
        description='Metadata filter, e.g. {"source": "policy.pdf"}. Applied to both '
        "retrievers so they see the same corpus slice.",
    )

    def history_dicts(self) -> list[dict]:
        return [t.model_dump() for t in self.history]


class Source(BaseModel):
    n: int = Field(..., description="Citation number matching [n] markers in the answer")
    source: str
    page: int
    score: float = Field(
        ..., description="Reranker score (0-1) when reranking is on, else cosine relevance"
    )
    found_by: list[str] = Field(
        default_factory=list,
        description="Which retrievers surfaced this chunk: 'dense', 'lexical', or both",
    )
    excerpt: str


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = Field(
        default=0.0, description="Estimate from INPUT/OUTPUT_COST_PER_MTOK; 0 if unset"
    )


class AskResponse(BaseModel):
    question: str
    search_query: str = Field(
        ..., description="The standalone query actually used for retrieval"
    )
    rewritten: bool = Field(
        ..., description="True if query rewriting changed the question"
    )
    answer: str
    grounded: bool = Field(
        ...,
        description="False when nothing cleared the relevance floor - the answer is "
        "a refusal and the LLM was not called",
    )
    sources: list[Source]
    retrieved_count: int
    kept_count: int
    mode: str = Field(..., description="Retrieval mode actually used")
    cached: bool = Field(
        default=False, description="Served from the semantic answer cache"
    )
    usage: Usage | None = None


class IngestRequest(BaseModel):
    path: str | None = Field(default=None, description="Folder to ingest; defaults to data/raw")
    rebuild: bool = False


class IngestResponse(BaseModel):
    added: list[str]
    updated: list[str]
    skipped: list[str]
    failed: list[dict]
    chunks_written: int
    lexical_documents: int = Field(
        default=0, description="Chunks in the rebuilt BM25 index"
    )


class SourceStat(BaseModel):
    source: str
    chunks: int


class StatsResponse(BaseModel):
    chunks: int
    sources: list[SourceStat]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    api_key_configured: bool
    chat_model: str
    embedding_model: str
    indexed_chunks: int
    lexical_documents: int = 0
    retrieval_mode: str = "hybrid"
    rerank_backend: str = "llm"
    detail: str | None = None

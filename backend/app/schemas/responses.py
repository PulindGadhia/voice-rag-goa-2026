from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class CitationResponse(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalInfo(BaseModel):
    top_k: int
    candidate_top_k: int
    num_results: int
    total_latency_ms: float
    hybrid_latency_ms: float
    rerank_latency_ms: float
    embedding_latency_ms: float = 0.0
    vector_search_latency_ms: float = 0.0
    bm25_latency_ms: float = 0.0
    rrf_latency_ms: float = 0.0
    model_name: str | None = None
    device: str | None = None
    retrieval_route: str | None = None
    translation_used: bool = False
    detected_language: str | None = None


class LatencyInfo(BaseModel):
    retrieval_ms: float = 0.0
    generation_ms: float = 0.0
    extractive_generation_ms: float = 0.0
    local_generation_ms: float = 0.0
    groq_generation_ms: float = 0.0
    grounding_ms: float = 0.0
    total_ms: float = 0.0


class GenerationInfo(BaseModel):
    provider: str | None = None
    model: str | None = None
    route: str | None = None  # "cache", "local", "groq"
    confidence_decision: str | None = None  # "LOCAL_ELIGIBLE", "LOCAL_UNCERTAIN", "GROQ_REQUIRED"
    fallback_used: bool = False


class QueryResponse(BaseModel):
    request_id: str
    query: str
    answer: str | None
    status: str
    grounded: bool | None = None
    citations: list[CitationResponse] = Field(default_factory=list)
    retrieval: RetrievalInfo | None = None
    generation: GenerationInfo | None = None
    latency: LatencyInfo
    guardrail_status: str = "passed"
    guardrail_reason: str | None = None
    source_language: str | None = None
    answer_language: str | None = None
    translation_used: bool = False

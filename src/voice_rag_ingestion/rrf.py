"""Reciprocal Rank Fusion for independent retrieval rankings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .qdrant_store import RetrievedChunk


@dataclass(frozen=True)
class HybridResult:
    chunk_id: str
    document_id: str
    text: str
    rrf_score: float
    vector_score: float | None
    bm25_score: float | None
    metadata: dict

    @property
    def score(self) -> float:
        """Common score property for future reranker compatibility."""

        return self.rrf_score


class RRFFuser:
    def __init__(self, rrf_k: int = 60) -> None:
        if rrf_k < 0:
            raise ValueError("rrf_k must be >= 0")
        self.rrf_k = rrf_k

    def fuse(
        self,
        vector_results: Sequence[RetrievedChunk],
        bm25_results: Sequence[RetrievedChunk],
        *,
        top_k: int = 10,
    ) -> list[HybridResult]:
        if top_k <= 0:
            raise ValueError("top_k must be > 0")
        merged: dict[str, dict] = {}
        for source_name, results in (("vector", vector_results), ("bm25", bm25_results)):
            seen_in_ranking: set[str] = set()
            for rank, result in enumerate(results, start=1):
                if result.chunk_id in seen_in_ranking:
                    continue
                seen_in_ranking.add(result.chunk_id)
                item = merged.setdefault(
                    result.chunk_id,
                    {
                        "chunk_id": result.chunk_id,
                        "document_id": result.document_id,
                        "text": result.text,
                        "metadata": dict(result.metadata),
                        "vector_score": None,
                        "bm25_score": None,
                        "rrf_score": 0.0,
                    },
                )
                item["rrf_score"] += 1.0 / (self.rrf_k + rank)
                if source_name == "vector":
                    item["vector_score"] = result.score
                else:
                    item["bm25_score"] = result.score
                if not item["text"]:
                    item["text"] = result.text
                if not item["document_id"]:
                    item["document_id"] = result.document_id
                item["metadata"].update(result.metadata)
        fused = [HybridResult(**item) for item in merged.values()]
        fused.sort(key=lambda result: (-result.rrf_score, result.chunk_id))
        return fused[:top_k]

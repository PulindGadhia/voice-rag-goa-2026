"""Small label-grounded retrieval evaluation for MSMARCO-XI samples."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from .rrf import HybridResult


@dataclass(frozen=True)
class EvaluationCase:
    query: str
    relevant_chunk_ids: frozenset[str]


@dataclass(frozen=True)
class RetrievalMetrics:
    recall_at_k: float
    mrr_at_k: float
    cases: int


def evaluate_rankings(
    rankings: Iterable[Sequence[HybridResult | object]],
    cases: Sequence[EvaluationCase],
    *,
    top_k: int,
) -> RetrievalMetrics | None:
    if top_k <= 0:
        raise ValueError("top_k must be > 0")
    cases_list = list(cases)
    if not cases_list:
        return None
    recalls: list[float] = []
    reciprocal_ranks: list[float] = []
    for ranking, case in zip(rankings, cases_list):
        if not case.relevant_chunk_ids:
            continue
        top_results = list(ranking)[:top_k]
        ids = [getattr(result, "chunk_id") for result in top_results]
        hits = set(ids) & set(case.relevant_chunk_ids)
        recalls.append(1.0 if hits else 0.0)
        reciprocal = 0.0
        for index, chunk_id in enumerate(ids, start=1):
            if chunk_id in case.relevant_chunk_ids:
                reciprocal = 1.0 / index
                break
        reciprocal_ranks.append(reciprocal)
    if not recalls:
        return None
    return RetrievalMetrics(
        recall_at_k=sum(recalls) / len(recalls),
        mrr_at_k=sum(reciprocal_ranks) / len(reciprocal_ranks),
        cases=len(recalls),
    )

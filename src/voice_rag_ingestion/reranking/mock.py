"""Small deterministic reranker for offline development and tests."""

from __future__ import annotations

from typing import Sequence

from ..rrf import HybridResult
from ..tokenization import UnicodeWordTokenizer
from .base import RerankResult, deduplicate_candidates


class LexicalOverlapReranker:
    """Deterministic local stand-in; it is not a semantic reranker."""

    def __init__(self) -> None:
        self.tokenizer = UnicodeWordTokenizer()

    def rerank(
        self,
        query: str,
        candidates: Sequence[HybridResult],
        top_k: int,
    ) -> list[RerankResult]:
        if top_k <= 0:
            raise ValueError("top_k must be > 0")
        if not query or not query.strip() or not candidates:
            return []
        query_tokens = set(self.tokenizer.tokenize(query))
        scored = []
        for candidate in deduplicate_candidates(candidates):
            passage_tokens = set(self.tokenizer.tokenize(candidate.text or ""))
            score = float(len(query_tokens & passage_tokens))
            scored.append((candidate, score))
        scored.sort(key=lambda item: (-item[1], -item[0].rrf_score, item[0].chunk_id))
        return [
            RerankResult.from_candidate(candidate, rerank_score=score, rerank_rank=rank)
            for rank, (candidate, score) in enumerate(scored[:top_k], start=1)
        ]


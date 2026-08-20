"""Traceable context construction for the generation stage."""

from __future__ import annotations

from dataclasses import dataclass

from .retrieval.contract import RetrievalCandidate


@dataclass(frozen=True)
class ContextSource:
    source_id: str
    chunk_id: str
    document_id: str
    text: str
    score: float


@dataclass(frozen=True)
class BuiltContext:
    query: str
    sources: list[ContextSource]
    text: str


def build_context(query: str, candidates: list[RetrievalCandidate]) -> BuiltContext:
    sources = [
        ContextSource(
            source_id=f"source-{index + 1}",
            chunk_id=item.chunk_id,
            document_id=item.document_id,
            text=item.text,
            score=item.score,
        )
        for index, item in enumerate(candidates)
    ]
    text = "\n\n".join(
        f"[{source.source_id}] chunk_id={source.chunk_id} document_id={source.document_id}\n{source.text}"
        for source in sources
    )
    return BuiltContext(query=query, sources=sources, text=text)


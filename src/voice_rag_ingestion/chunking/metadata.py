"""Chunking strategy that explicitly carries MSMARCO-XI provenance metadata."""

from __future__ import annotations

from copy import deepcopy

from ..documents import NormalizedDocument
from .base import Chunk, ChunkingConfig
from .sentence import SentenceAwareChunker


class MetadataAwareChunker(SentenceAwareChunker):
    name = "metadata"

    def __init__(self, config: ChunkingConfig | None = None) -> None:
        super().__init__(config or ChunkingConfig(strategy="metadata"))

    def chunk(self, document: NormalizedDocument) -> list[Chunk]:
        chunks = super().chunk(document)
        required_source = {
            "language": document.language,
            "query_id": document.query_id,
            "document_id": document.document_id,
            "dataset_name": document.dataset_name,
            "split": document.source.get("split"),
            "passage_index": document.source.get("passage_index"),
            "text_source": document.source.get("text_source"),
            "source_lang": document.source.get("source_lang"),
            "target_lang": document.source.get("target_lang"),
        }
        enriched: list[Chunk] = []
        for chunk in chunks:
            source = deepcopy(chunk.source)
            for key, value in required_source.items():
                source.setdefault(key, value)
            enriched.append(
                Chunk(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    parent_chunk_id=chunk.parent_chunk_id,
                    text=chunk.text,
                    language=chunk.language,
                    chunk_index=chunk.chunk_index,
                    chunk_strategy=chunk.chunk_strategy,
                    query_id=chunk.query_id,
                    source=source,
                    metadata=deepcopy(chunk.metadata),
                )
            )
        return enriched

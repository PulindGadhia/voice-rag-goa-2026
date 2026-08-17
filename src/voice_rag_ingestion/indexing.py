"""Reusable normalized-document to vector-index pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .chunking import ChunkingConfig, chunk_document
from .chunking.base import Chunk
from .documents import NormalizedDocument
from .embeddings.base import EmbeddingProvider, timed_batch_embed
from .qdrant_store import QdrantVectorStore


@dataclass
class IndexStats:
    documents: int = 0
    chunks: int = 0
    vectors: int = 0
    embedding_batches: int = 0
    embedding_seconds: float = 0.0
    upserted: int = 0


def index_documents(
    documents: Sequence[NormalizedDocument],
    *,
    embedder: EmbeddingProvider,
    vector_store: QdrantVectorStore,
    chunking_config: ChunkingConfig,
    embedding_batch_size: int,
    recreate_collection: bool | None = None,
) -> tuple[IndexStats, list[Chunk]]:
    chunks = [
        chunk
        for document in documents
        for chunk in chunk_document(document, config=chunking_config)
    ]
    vector_store.ensure_collection(embedder.dimension, recreate=recreate_collection)
    vectors, embedding_stats = timed_batch_embed(
        embedder,
        [chunk.text for chunk in chunks],
        batch_size=embedding_batch_size,
        input_type="passage",
    )
    upserted = vector_store.upsert(chunks, vectors)
    return (
        IndexStats(
            documents=len(documents),
            chunks=len(chunks),
            vectors=len(vectors),
            embedding_batches=embedding_stats.batches,
            embedding_seconds=embedding_stats.elapsed_seconds,
            upserted=upserted,
        ),
        chunks,
    )

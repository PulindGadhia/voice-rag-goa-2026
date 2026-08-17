"""Single entry point for selecting a chunking strategy."""

from __future__ import annotations

from ..documents import NormalizedDocument
from .base import Chunk, ChunkingConfig
from .fixed import FixedSizeChunker
from .metadata import MetadataAwareChunker
from .semantic import EmbeddingProvider, SemanticChunker
from .sentence import SentenceAwareChunker


def chunk_document(
    document: NormalizedDocument,
    *,
    strategy: str | None = None,
    config: ChunkingConfig | None = None,
    embedding_provider: EmbeddingProvider | None = None,
) -> list[Chunk]:
    """Chunk one document through the common strategy interface."""

    selected = strategy or (config.strategy if config else "fixed")
    if selected not in {"fixed", "sentence", "semantic", "metadata"}:
        raise ValueError(f"unsupported chunking strategy: {selected}")
    if config is None:
        config = ChunkingConfig(strategy=selected)
    elif config.strategy != selected:
        config = ChunkingConfig(
            max_chunk_size=config.max_chunk_size,
            overlap=config.overlap,
            min_chunk_size=config.min_chunk_size,
            semantic_similarity_threshold=config.semantic_similarity_threshold,
            strategy=selected,
        )
    chunker = {
        "fixed": FixedSizeChunker,
        "sentence": SentenceAwareChunker,
        "semantic": lambda cfg: SemanticChunker(cfg, embedding_provider=embedding_provider),
        "metadata": MetadataAwareChunker,
    }[selected](config)
    return chunker.chunk(document)

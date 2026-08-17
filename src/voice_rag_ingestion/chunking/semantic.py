"""Injectable semantic-boundary chunking."""

from __future__ import annotations

import hashlib
import math
from typing import Protocol, Sequence

from ..documents import NormalizedDocument
from .base import Chunk, ChunkingConfig, ChunkingStrategy, token_count
from .sentence import split_oversized_sentence, split_sentences


class EmbeddingProvider(Protocol):
    """Minimal interface for future model-backed sentence embeddings."""

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        ...


class DeterministicEmbedding:
    """Small dependency-free embedding useful for development and tests."""

    def __init__(self, dimensions: int = 32) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be > 0")
        self.dimensions = dimensions

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vector = [0.0] * self.dimensions
            for token in text.casefold().split():
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                index = int.from_bytes(digest[:4], "big") % self.dimensions
                sign = 1.0 if digest[4] % 2 else -1.0
                vector[index] += sign
            vectors.append(vector)
        return vectors


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("embedding dimensions do not match")
    numerator = sum(a * b for a, b in zip(left, right))
    denominator = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
    return numerator / denominator if denominator else 0.0


class SemanticChunker(ChunkingStrategy):
    name = "semantic"

    def __init__(
        self,
        config: ChunkingConfig | None = None,
        *,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> None:
        super().__init__(config or ChunkingConfig(strategy="semantic"))
        self.embedding_provider = embedding_provider or DeterministicEmbedding()

    def chunk(self, document: NormalizedDocument) -> list[Chunk]:
        sentences = split_sentences(document.text)
        if not sentences:
            return []
        embeddings = self.embedding_provider.embed(sentences)
        if len(embeddings) != len(sentences):
            raise ValueError("embedding provider returned the wrong number of vectors")
        similarities = [
            cosine_similarity(embeddings[index - 1], embeddings[index])
            for index in range(1, len(sentences))
        ]
        boundaries: set[int] = set()
        threshold = self.config.semantic_similarity_threshold
        for index, similarity in enumerate(similarities, start=1):
            significant_drop = index > 1 and similarity < similarities[index - 2] - 0.15
            if similarity < threshold or significant_drop:
                boundaries.add(index)

        groups: list[str] = []
        current: list[str] = []
        current_size = 0
        for index, sentence in enumerate(sentences):
            sentence_parts = split_oversized_sentence(sentence, self.config.max_chunk_size)
            for part_index, part in enumerate(sentence_parts):
                part_size = token_count(part)
                if current and (index in boundaries and part_index == 0 or current_size + part_size > self.config.max_chunk_size):
                    groups.append(" ".join(current))
                    current = []
                    current_size = 0
                current.append(part)
                current_size += part_size
        if current:
            groups.append(" ".join(current))
        return self._build_chunks(document, groups)

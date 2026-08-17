"""Fixed-size word-approximation chunking."""

from __future__ import annotations

from ..documents import NormalizedDocument
from .base import Chunk, ChunkingConfig, ChunkingStrategy, word_tokens


class FixedSizeChunker(ChunkingStrategy):
    name = "fixed"

    def __init__(self, config: ChunkingConfig | None = None) -> None:
        super().__init__(config or ChunkingConfig(strategy="fixed"))

    def chunk(self, document: NormalizedDocument) -> list[Chunk]:
        words = word_tokens(document.text)
        if not words:
            return []
        max_size = self.config.max_chunk_size
        step = max_size - self.config.overlap
        spans: list[tuple[int, int]] = []
        start = 0
        while start < len(words):
            end = min(start + max_size, len(words))
            spans.append((start, end))
            if end == len(words):
                break
            start += step

        if len(spans) > 1 and spans[-1][1] - spans[-1][0] < self.config.min_chunk_size:
            last_start = max(0, len(words) - max_size)
            if last_start != spans[-2][0]:
                spans[-1] = (last_start, len(words))
            else:
                spans.pop()
        texts = [" ".join(words[start:end]) for start, end in spans]
        return self._build_chunks(document, texts)

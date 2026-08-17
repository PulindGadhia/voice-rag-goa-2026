"""Sentence-aware chunking with long-sentence fallback."""

from __future__ import annotations

import re

from ..cleaning import clean_text
from ..documents import NormalizedDocument
from .base import Chunk, ChunkingConfig, ChunkingStrategy, word_tokens


_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+|(?<=[。！？।॥])\s*", re.UNICODE)


def split_sentences(text: str) -> list[str]:
    return [part for part in (clean_text(item) for item in _SENTENCE_BOUNDARY.split(text)) if part]


def split_oversized_sentence(sentence: str, max_size: int) -> list[str]:
    words = word_tokens(sentence)
    if len(words) <= max_size:
        return [sentence]
    if len(words) == 1:
        # A no-whitespace script/string still needs a bounded fallback. Slicing
        # only happens when a sentence itself exceeds the configured maximum.
        return [sentence[index : index + max_size] for index in range(0, len(sentence), max_size)]
    return [" ".join(words[index : index + max_size]) for index in range(0, len(words), max_size)]


class SentenceAwareChunker(ChunkingStrategy):
    name = "sentence"

    def __init__(self, config: ChunkingConfig | None = None) -> None:
        super().__init__(config or ChunkingConfig(strategy="sentence"))

    def chunk(self, document: NormalizedDocument) -> list[Chunk]:
        sentences = split_sentences(document.text)
        if not sentences:
            return []
        expanded: list[str] = []
        for sentence in sentences:
            expanded.extend(split_oversized_sentence(sentence, self.config.max_chunk_size))

        chunks: list[str] = []
        current: list[str] = []
        current_size = 0
        for sentence in expanded:
            size = len(word_tokens(sentence))
            if current and current_size + size > self.config.max_chunk_size:
                chunks.append(" ".join(current))
                # Preserve only complete trailing sentences for sentence-aware
                # overlap; never cut a sentence to manufacture overlap.
                overlap_sentences: list[str] = []
                overlap_size = 0
                for previous in reversed(current):
                    previous_size = len(word_tokens(previous))
                    if overlap_size + previous_size > self.config.overlap:
                        break
                    overlap_sentences.insert(0, previous)
                    overlap_size += previous_size
                current = overlap_sentences
                current_size = overlap_size
            current.append(sentence)
            current_size += size
        if current:
            chunks.append(" ".join(current))
        return self._build_chunks(document, chunks)

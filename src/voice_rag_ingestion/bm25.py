"""Persistent, provider-independent BM25 retrieval over existing chunks."""

from __future__ import annotations

import json
import logging
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

from .chunking.base import Chunk
from .qdrant_store import RetrievedChunk
from .tokenization import Tokenizer, TokenizerConfig, UnicodeWordTokenizer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BM25Config:
    k1: float = 1.5
    b: float = 0.75
    tokenizer: TokenizerConfig = TokenizerConfig()

    def __post_init__(self) -> None:
        if self.k1 < 0:
            raise ValueError("BM25 k1 must be >= 0")
        if not 0 <= self.b <= 1:
            raise ValueError("BM25 b must be between 0 and 1")


class BM25Index:
    """An in-memory BM25 index that can be saved as safe JSON."""

    def __init__(
        self,
        config: BM25Config | None = None,
        *,
        tokenizer: Tokenizer | None = None,
    ) -> None:
        self.config = config or BM25Config()
        self.tokenizer = tokenizer or UnicodeWordTokenizer(self.config.tokenizer)
        self._chunks: dict[str, Chunk] = {}
        self._tokens: dict[str, list[str]] = {}
        self._document_frequency: Counter[str] = Counter()
        self._average_document_length = 0.0

    @property
    def chunks(self) -> list[Chunk]:
        return list(self._chunks.values())

    @property
    def size(self) -> int:
        return len(self._chunks)

    def rebuild(self, chunks: Iterable[Chunk]) -> int:
        self._chunks.clear()
        self._tokens.clear()
        return self.add(chunks)

    def add(self, chunks: Iterable[Chunk]) -> int:
        added = 0
        for chunk in chunks:
            if not chunk.chunk_id or not chunk.text or not chunk.text.strip():
                continue
            if chunk.chunk_id not in self._chunks:
                added += 1
            self._chunks[chunk.chunk_id] = chunk
        self._recompute_statistics()
        return added

    def _recompute_statistics(self) -> None:
        self._tokens = {
            chunk_id: self.tokenizer.tokenize(chunk.text)
            for chunk_id, chunk in self._chunks.items()
        }
        self._document_frequency = Counter()
        for tokens in self._tokens.values():
            self._document_frequency.update(set(tokens))
        total_length = sum(len(tokens) for tokens in self._tokens.values())
        self._average_document_length = total_length / self.size if self.size else 0.0

    def _score(self, query_tokens: list[str], chunk_id: str) -> float:
        tokens = self._tokens[chunk_id]
        if not tokens:
            return 0.0
        counts = Counter(tokens)
        document_length = len(tokens)
        score = 0.0
        for token in query_tokens:
            term_frequency = counts.get(token, 0)
            if not term_frequency:
                continue
            document_frequency = self._document_frequency[token]
            idf = math.log(
                1.0
                + (self.size - document_frequency + 0.5)
                / (document_frequency + 0.5)
            )
            denominator = term_frequency + self.config.k1 * (
                1.0
                - self.config.b
                + self.config.b * document_length / self._average_document_length
            )
            score += idf * (term_frequency * (self.config.k1 + 1.0)) / denominator
        return score

    def search(self, query: str, *, top_k: int = 10) -> list[RetrievedChunk]:
        if top_k <= 0:
            raise ValueError("top_k must be > 0")
        query_tokens = self.tokenizer.tokenize(query)
        if not query_tokens:
            return []
        scored = [
            (self._score(query_tokens, chunk_id), chunk_id)
            for chunk_id in self._chunks
        ]
        scored = [item for item in scored if item[0] > 0.0]
        scored.sort(key=lambda item: (-item[0], item[1]))
        results: list[RetrievedChunk] = []
        for score, chunk_id in scored[:top_k]:
            chunk = self._chunks[chunk_id]
            metadata = dict(chunk.source)
            metadata.update(
                {
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "query_id": chunk.query_id,
                    "language": chunk.language,
                    "chunk_strategy": chunk.chunk_strategy,
                    "chunk_index": chunk.chunk_index,
                    "text": chunk.text,
                    "document_metadata": dict(chunk.metadata),
                }
            )
            results.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    text=chunk.text,
                    score=float(score),
                    metadata=metadata,
                )
            )
        return results

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "config": {
                "k1": self.config.k1,
                "b": self.config.b,
                "tokenizer": {
                    "lowercase": self.config.tokenizer.lowercase,
                    "unicode_normalization": self.config.tokenizer.unicode_normalization,
                    "min_token_length": self.config.tokenizer.min_token_length,
                },
            },
            "chunks": [chunk.to_dict() for chunk in self.chunks],
        }
        destination.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BM25Index":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("version") != 1:
            raise ValueError("unsupported BM25 index version")
        raw_config = payload.get("config", {})
        tokenizer_config = TokenizerConfig(**raw_config.get("tokenizer", {}))
        index = cls(
            BM25Config(
                k1=float(raw_config.get("k1", 1.5)),
                b=float(raw_config.get("b", 0.75)),
                tokenizer=tokenizer_config,
            )
        )
        chunks = [Chunk(**item) for item in payload.get("chunks", [])]
        index.rebuild(chunks)
        return index


class BM25Retriever:
    """Retrieval facade over a reusable BM25 index."""

    def __init__(self, index: BM25Index) -> None:
        self.index = index

    def retrieve(self, query: str, *, top_k: int = 10) -> list[RetrievedChunk]:
        if not query or not query.strip():
            return []
        return self.index.search(query.strip(), top_k=top_k)

    def retrieve_with_timing(
        self, query: str, *, top_k: int = 10
    ) -> tuple[list[RetrievedChunk], float]:
        started = perf_counter()
        results = self.retrieve(query, top_k=top_k)
        return results, perf_counter() - started

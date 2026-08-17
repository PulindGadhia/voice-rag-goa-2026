"""Shared chunk schema, configuration, and validation."""

from __future__ import annotations

import hashlib
import json
import os
import re
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

from ..cleaning import clean_text, stable_text_key
from ..documents import NormalizedDocument


@dataclass(frozen=True)
class Chunk:
    """A traceable fragment of one normalized document."""

    chunk_id: str
    document_id: str
    text: str
    language: str | None
    chunk_index: int
    chunk_strategy: str
    query_id: str | None = None
    source: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def source_metadata(self) -> dict[str, Any]:
        return self.source

    @property
    def original_document_metadata(self) -> dict[str, Any]:
        return self.metadata

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ChunkingConfig:
    """Configuration shared by every chunking strategy.

    ``max_chunk_size`` and ``overlap`` are approximate whitespace-token/word
    counts. This deliberately avoids coupling ingestion to a future model
    tokenizer. The tokenizer can be replaced by a strategy implementation.
    """

    max_chunk_size: int = 256
    overlap: int = 32
    min_chunk_size: int = 1
    semantic_similarity_threshold: float = 0.70
    strategy: str = "fixed"

    @classmethod
    def from_env(cls) -> "ChunkingConfig":
        """Build chunking settings from ``CHUNK_*`` environment variables."""

        return cls(
            max_chunk_size=int(os.getenv("CHUNK_MAX_SIZE", "256")),
            overlap=int(os.getenv("CHUNK_OVERLAP", "32")),
            min_chunk_size=int(os.getenv("CHUNK_MIN_SIZE", "1")),
            semantic_similarity_threshold=float(
                os.getenv("CHUNK_SEMANTIC_THRESHOLD", "0.70")
            ),
            strategy=os.getenv("CHUNK_STRATEGY", "fixed"),
        )

    def __post_init__(self) -> None:
        if self.max_chunk_size <= 0:
            raise ValueError("max_chunk_size must be > 0")
        if self.overlap < 0 or self.overlap >= self.max_chunk_size:
            raise ValueError("overlap must be >= 0 and smaller than max_chunk_size")
        if self.min_chunk_size <= 0 or self.min_chunk_size > self.max_chunk_size:
            raise ValueError("min_chunk_size must be > 0 and <= max_chunk_size")
        if not 0.0 <= self.semantic_similarity_threshold <= 1.0:
            raise ValueError("semantic_similarity_threshold must be between 0 and 1")
        if self.strategy not in {"fixed", "sentence", "semantic", "metadata"}:
            raise ValueError(f"unsupported strategy: {self.strategy}")


class ChunkValidationError(ValueError):
    """Raised when generated chunks violate the common contract."""


def chunk_id_for(document_id: str, strategy: str, index: int, text: str) -> str:
    payload = json.dumps(
        [document_id, strategy, index, text], ensure_ascii=False, separators=(",", ":")
    )
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:20]
    return f"{document_id}--{strategy}-{index:04d}-{digest}"


def word_tokens(text: str) -> list[str]:
    """Whitespace tokens preserving all Unicode characters."""

    return re.findall(r"\S+", text, flags=re.UNICODE)


def token_count(text: str) -> int:
    return len(word_tokens(text))


class ChunkingStrategy(ABC):
    """Common interface implemented by all chunkers."""

    name: str

    def __init__(self, config: ChunkingConfig | None = None) -> None:
        self.config = config or ChunkingConfig()

    @abstractmethod
    def chunk(self, document: NormalizedDocument) -> list[Chunk]:
        """Convert one normalized document into traceable chunks."""

    def _build_chunks(
        self, document: NormalizedDocument, texts: Sequence[str]
    ) -> list[Chunk]:
        unique_texts: list[str] = []
        seen: set[str] = set()
        for raw_text in texts:
            text = clean_text(raw_text)
            if text is None:
                continue
            key = stable_text_key(text)
            if key in seen:
                continue
            seen.add(key)
            unique_texts.append(text)

        chunks: list[Chunk] = []
        for index, text in enumerate(unique_texts):
            source = deepcopy(document.source)
            source.update(
                {
                    "document_id": document.document_id,
                    "dataset_name": document.dataset_name,
                    "query_id": document.query_id,
                    "chunk_index": index,
                    "chunk_strategy": self.name,
                }
            )
            metadata = deepcopy(document.metadata)
            metadata.setdefault("original_document_id", document.document_id)
            chunks.append(
                Chunk(
                    chunk_id=chunk_id_for(document.document_id, self.name, index, text),
                    document_id=document.document_id,
                    text=text,
                    language=document.language,
                    chunk_index=index,
                    chunk_strategy=self.name,
                    query_id=document.query_id,
                    source=source,
                    metadata=metadata,
                )
            )
        validate_chunks(chunks, expected_strategy=self.name)
        return chunks


def validate_chunks(
    chunks: Sequence[Chunk], *, expected_strategy: str | None = None
) -> None:
    """Validate traceability, deterministic IDs, indexes, and metadata."""

    seen_ids: set[str] = set()
    for expected_index, chunk in enumerate(chunks):
        if not clean_text(chunk.text):
            raise ChunkValidationError("every chunk must have non-empty text")
        if not chunk.chunk_id:
            raise ChunkValidationError("every chunk must have a chunk_id")
        if not chunk.document_id:
            raise ChunkValidationError("every chunk must point to a document_id")
        if chunk.chunk_index != expected_index:
            raise ChunkValidationError("chunk_index must be deterministic and contiguous")
        if expected_strategy and chunk.chunk_strategy != expected_strategy:
            raise ChunkValidationError(
                f"expected strategy {expected_strategy!r}, got {chunk.chunk_strategy!r}"
            )
        expected_id = chunk_id_for(
            chunk.document_id, chunk.chunk_strategy, chunk.chunk_index, chunk.text
        )
        if chunk.chunk_id != expected_id:
            raise ChunkValidationError("chunk_id is not deterministic")
        if chunk.chunk_id in seen_ids:
            raise ChunkValidationError("duplicate chunk_id")
        seen_ids.add(chunk.chunk_id)
        if chunk.source.get("document_id") != chunk.document_id:
            raise ChunkValidationError("source metadata lost document relationship")

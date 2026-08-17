"""Qdrant collection and payload operations."""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from typing import Any, Sequence

from .chunking.base import Chunk


@dataclass(frozen=True)
class VectorStoreConfig:
    url: str = ":memory:"
    collection_name: str = "msmarco_xi_dev"
    api_key: str | None = None
    recreate_collection: bool = True
    timeout: float = 60.0
    upsert_batch_size: int = 64

    @classmethod
    def from_env(cls) -> "VectorStoreConfig":
        return cls(
            url=os.getenv("QDRANT_URL", cls.url),
            collection_name=os.getenv("QDRANT_COLLECTION", cls.collection_name),
            api_key=os.getenv("QDRANT_API_KEY") or None,
            recreate_collection=os.getenv("QDRANT_RECREATE", "true").lower()
            in {"1", "true", "yes", "on"},
            timeout=float(os.getenv("QDRANT_TIMEOUT", str(cls.timeout))),
            upsert_batch_size=int(
                os.getenv("QDRANT_UPSERT_BATCH_SIZE", str(cls.upsert_batch_size))
            ),
        )


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    document_id: str
    text: str
    score: float
    metadata: dict[str, Any]


def chunk_payload(chunk: Chunk) -> dict[str, Any]:
    source = dict(chunk.source)
    return {
        "chunk_id": chunk.chunk_id,
        "document_id": chunk.document_id,
        "query_id": chunk.query_id,
        "language": chunk.language,
        "chunk_strategy": chunk.chunk_strategy,
        "chunk_index": chunk.chunk_index,
        "passage_index": source.get("passage_index"),
        "source_lang": source.get("source_lang"),
        "target_lang": source.get("target_lang"),
        "text_source": source.get("text_source"),
        "dataset_name": source.get("dataset_name"),
        "split": source.get("split"),
        "text": chunk.text,
        "source": source,
        "document_metadata": dict(chunk.metadata),
    }


class QdrantVectorStore:
    """Thin Qdrant adapter with injectable client for unit tests."""

    def __init__(
        self,
        config: VectorStoreConfig | None = None,
        *,
        client: object | None = None,
    ) -> None:
        self.config = config or VectorStoreConfig.from_env()
        if self.config.upsert_batch_size <= 0:
            raise ValueError("upsert_batch_size must be > 0")
        if client is not None:
            self.client = client
            return
        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:
            raise RuntimeError("qdrant-client is required for vector storage") from exc
        if self.config.url in {":memory:", "memory"}:
            self.client = QdrantClient(":memory:")
        else:
            kwargs: dict[str, Any] = {"url": self.config.url, "timeout": self.config.timeout}
            if self.config.api_key:
                kwargs["api_key"] = self.config.api_key
            self.client = QdrantClient(**kwargs)

    def collection_exists(self) -> bool:
        if hasattr(self.client, "collection_exists"):
            return bool(self.client.collection_exists(self.config.collection_name))
        collections = self.client.get_collections().collections
        return any(item.name == self.config.collection_name for item in collections)

    def ensure_collection(self, vector_size: int, *, recreate: bool | None = None) -> None:
        if vector_size <= 0:
            raise ValueError("vector_size must be > 0")
        should_recreate = self.config.recreate_collection if recreate is None else recreate
        exists = self.collection_exists()
        if exists and should_recreate:
            self.client.delete_collection(self.config.collection_name)
            exists = False
        if not exists:
            from qdrant_client.models import Distance, VectorParams

            self.client.create_collection(
                collection_name=self.config.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )

    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> int:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        from qdrant_client.models import PointStruct

        total = 0
        for offset in range(0, len(chunks), self.config.upsert_batch_size):
            points = [
                PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, chunk.chunk_id)),
                    vector=list(vector),
                    payload=chunk_payload(chunk),
                )
                for chunk, vector in zip(
                    chunks[offset : offset + self.config.upsert_batch_size],
                    vectors[offset : offset + self.config.upsert_batch_size],
                )
            ]
            if points:
                self.client.upsert(
                    collection_name=self.config.collection_name,
                    points=points,
                    wait=True,
                )
                total += len(points)
        return total

    def search(self, vector: Sequence[float], *, top_k: int = 5) -> list[RetrievedChunk]:
        if top_k <= 0:
            raise ValueError("top_k must be > 0")
        if hasattr(self.client, "query_points"):
            response = self.client.query_points(
                collection_name=self.config.collection_name,
                query=list(vector),
                limit=top_k,
                with_payload=True,
            )
            points = getattr(response, "points", response)
        else:
            points = self.client.search(
                collection_name=self.config.collection_name,
                query_vector=list(vector),
                limit=top_k,
                with_payload=True,
            )
        results: list[RetrievedChunk] = []
        for point in points:
            payload = getattr(point, "payload", None) or {}
            results.append(
                RetrievedChunk(
                    chunk_id=str(payload.get("chunk_id", getattr(point, "id", ""))),
                    document_id=str(payload.get("document_id", "")),
                    text=str(payload.get("text", "")),
                    score=float(getattr(point, "score", 0.0)),
                    metadata=payload,
                )
            )
        return results

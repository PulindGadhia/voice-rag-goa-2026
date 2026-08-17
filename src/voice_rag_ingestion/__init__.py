"""Dataset ingestion primitives for the Voice RAG project."""

from .config import LoaderConfig
from .bm25 import BM25Config, BM25Index, BM25Retriever
from .documents import NormalizedDocument, normalize_record
from .hybrid import HybridConfig, HybridRetriever
from .loader import DatasetLoader, LoadStats
from .qdrant_store import QdrantVectorStore, RetrievedChunk, VectorStoreConfig
from .retrieval import RetrievalTiming, VectorRetriever
from .rrf import HybridResult, RRFFuser
from .reranking import (
    CrossEncoderReranker,
    HybridRerankRetriever,
    LexicalOverlapReranker,
    RerankPhaseTiming,
    RerankResult,
    RerankerConfig,
    detect_inference_device,
)

__all__ = [
    "DatasetLoader",
    "BM25Config",
    "BM25Index",
    "BM25Retriever",
    "HybridConfig",
    "HybridRetriever",
    "HybridResult",
    "LoadStats",
    "LoaderConfig",
    "NormalizedDocument",
    "QdrantVectorStore",
    "RetrievedChunk",
    "RetrievalTiming",
    "VectorRetriever",
    "VectorStoreConfig",
    "RRFFuser",
    "normalize_record",
    "CrossEncoderReranker",
    "HybridRerankRetriever",
    "LexicalOverlapReranker",
    "RerankResult",
    "RerankPhaseTiming",
    "RerankerConfig",
    "detect_inference_device",
]

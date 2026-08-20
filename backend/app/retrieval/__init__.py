"""Provider-neutral application retrieval boundary."""

from .contract import RetrievalCandidate, RetrievalEngine, RetrievalResponse
from .existing import ExistingRetrievalEngine, build_existing_retrieval_engine

__all__ = [
    "ExistingRetrievalEngine",
    "RetrievalCandidate",
    "RetrievalEngine",
    "RetrievalResponse",
    "build_existing_retrieval_engine",
]


"""Environment-backed reranker configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env_bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class RerankerConfig:
    """Runtime settings for the selected multilingual cross-encoder."""

    model_name: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    batch_size: int = 8
    max_length: int = 256
    device: str | None = None
    trust_remote_code: bool = True
    candidate_top_k: int = 5
    final_top_k: int = 5
    batch_all_candidates: bool = True
    warmup_enabled: bool = True
    warmup_candidates: int = 1

    @classmethod
    def from_env(cls) -> "RerankerConfig":
        device = os.getenv("RERANKER_DEVICE") or None
        return cls(
            model_name=os.getenv(
                "RERANKER_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
            ),
            batch_size=int(os.getenv("RERANKER_BATCH_SIZE", "8")),
            max_length=int(os.getenv("RERANKER_MAX_LENGTH", "256")),
            device=device,
            trust_remote_code=_env_bool("RERANKER_TRUST_REMOTE_CODE", True),
            candidate_top_k=int(os.getenv("RERANKER_CANDIDATE_TOP_K", "5")),
            final_top_k=int(os.getenv("RERANKER_FINAL_TOP_K", "5")),
            batch_all_candidates=_env_bool("RERANKER_BATCH_ALL_CANDIDATES", True),
            warmup_enabled=_env_bool("RERANKER_WARMUP_ENABLED", True),
            warmup_candidates=int(os.getenv("RERANKER_WARMUP_CANDIDATES", "1")),
        )

    def __post_init__(self) -> None:
        if not self.model_name.strip():
            raise ValueError("reranker model_name must not be empty")
        if self.batch_size <= 0 or self.max_length <= 0:
            raise ValueError("reranker batch_size and max_length must be > 0")
        if self.candidate_top_k <= 0 or self.final_top_k <= 0:
            raise ValueError("reranker top_k values must be > 0")
        if self.warmup_candidates <= 0:
            raise ValueError("reranker warmup_candidates must be > 0")

"""Environment-backed application configuration.

The application layer deliberately keeps retrieval-specific construction out of
the API handlers.  Values here are deployment choices only; secrets are read
from the environment and are never committed to the repository.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _load_local_env() -> None:
    """Load the existing local .env file without overriding process values."""

    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(override=False)


@dataclass(frozen=True)
class Settings:
    llm_provider: str = "groq"
    llm_model: str = "openai/gpt-oss-20b"
    groq_api_key: str = ""
    gemini_api_key: str = ""
    stt_provider: str = "sarvam"
    sarvam_api_key: str = ""
    qdrant_url: str = ":memory:"
    qdrant_collection: str = "msmarco_xi_dev"
    qdrant_api_key: str | None = None
    top_k: int = 3
    candidate_top_k: int = 5
    app_auto_index: bool = True
    app_index_sample_size: int = 2
    app_index_recreate: bool = True
    local_model_path: str = "models/qwen-0.5b"
    two_tier_enabled: bool = True
    local_confidence_threshold: float = 0.20
    extractive_enabled: bool = True
    extractive_confidence_threshold: float = 0.50
    adaptive_routing_enabled: bool = True
    fast_path_rrf_threshold: float = 0.025
    fast_path_min_agreeing_sources: int = 2
    fast_path_bm25_threshold: float = 5.0
    fast_path_extractive_threshold: float = 0.015
    multilingual_enabled: bool = True
    translation_provider: str = "sarvam"
    translation_cache_size: int = 512
    cors_origins: tuple[str, ...] = ("http://localhost:3000", "http://localhost:5173", "http://localhost:8000", "http://127.0.0.1:8000", "*")
    host: str = "0.0.0.0"
    port: int = 8000

    @classmethod
    def from_env(cls) -> "Settings":
        process_provider = os.getenv("LLM_PROVIDER")
        process_model = os.getenv("LLM_MODEL")
        _load_local_env()
        llm_provider = os.getenv("LLM_PROVIDER", cls.llm_provider).lower()
        if llm_provider == "gemini":
            default_llm_model = "gemini-3.6-flash"
        elif llm_provider == "local":
            default_llm_model = "models/smollm2-135m"
        else:
            default_llm_model = cls.llm_model
        if process_provider is not None and process_model is None:
            # An explicit provider override should not inherit a model from a
            # different provider in the local .env file.
            llm_model = default_llm_model
        else:
            llm_model = os.getenv("LLM_MODEL", default_llm_model)
        origins = tuple(
            item.strip()
            for item in os.getenv(
                "CORS_ORIGINS", "http://localhost:3000,http://localhost:5173"
            ).split(",") or "http://localhost:3000,http://localhost:5173,http://localhost:8000,http://127.0.0.1:8000,*"
            if item.strip()
        )
        return cls(
            llm_provider=llm_provider,
            llm_model=llm_model,
            groq_api_key=os.getenv("GROQ_API_KEY", ""),
            gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
            stt_provider=os.getenv("STT_PROVIDER", cls.stt_provider).lower(),
            sarvam_api_key=os.getenv("SARVAM_API_KEY", ""),
            qdrant_url=os.getenv("QDRANT_URL", cls.qdrant_url),
            qdrant_collection=os.getenv("QDRANT_COLLECTION", cls.qdrant_collection),
            qdrant_api_key=os.getenv("QDRANT_API_KEY") or None,
            top_k=int(os.getenv("APP_TOP_K", str(cls.top_k))),
            candidate_top_k=int(os.getenv("APP_CANDIDATE_TOP_K", str(cls.candidate_top_k))),
            app_auto_index=_bool("APP_AUTO_INDEX", cls.app_auto_index),
            app_index_sample_size=int(os.getenv("APP_INDEX_SAMPLE_SIZE", str(cls.app_index_sample_size))),
            app_index_recreate=_bool("APP_INDEX_RECREATE", cls.app_index_recreate),
            local_model_path=os.getenv("LOCAL_MODEL_PATH", cls.local_model_path),
            two_tier_enabled=_bool("TWO_TIER_ROUTING", cls.two_tier_enabled),
            local_confidence_threshold=float(os.getenv("LOCAL_CONFIDENCE_THRESHOLD", str(cls.local_confidence_threshold))),
            extractive_enabled=_bool("EXTRACTIVE_ENABLED", cls.extractive_enabled),
            extractive_confidence_threshold=float(os.getenv("EXTRACTIVE_CONFIDENCE_THRESHOLD", str(cls.extractive_confidence_threshold))),
            adaptive_routing_enabled=_bool("ADAPTIVE_ROUTING_ENABLED", cls.adaptive_routing_enabled),
            fast_path_rrf_threshold=float(os.getenv("FAST_PATH_RRF_THRESHOLD", str(cls.fast_path_rrf_threshold))),
            fast_path_min_agreeing_sources=int(os.getenv("FAST_PATH_MIN_AGREEING_SOURCES", str(cls.fast_path_min_agreeing_sources))),
            fast_path_bm25_threshold=float(os.getenv("FAST_PATH_BM25_THRESHOLD", str(cls.fast_path_bm25_threshold))),
            fast_path_extractive_threshold=float(os.getenv("FAST_PATH_EXTRACTIVE_THRESHOLD", str(cls.fast_path_extractive_threshold))),
            multilingual_enabled=_bool("MULTILINGUAL_ENABLED", cls.multilingual_enabled),
            translation_provider=os.getenv("TRANSLATION_PROVIDER", cls.translation_provider).lower(),
            translation_cache_size=int(os.getenv("TRANSLATION_CACHE_SIZE", str(cls.translation_cache_size))),
            cors_origins=origins,
            host=os.getenv("APP_HOST", cls.host),
            port=int(os.getenv("APP_PORT", str(cls.port))),
        )

    @property
    def active_llm_api_key(self) -> str:
        if self.llm_provider == "gemini":
            return self.gemini_api_key
        if self.llm_provider == "local":
            return "local"
        return self.groq_api_key


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings.from_env()
    return _settings

"""Sentence-Transformers cross-encoder adapter with profiled batching."""

from __future__ import annotations

import logging
from time import perf_counter
from typing import Any, Sequence

from ..rrf import HybridResult
from .base import RerankPhaseTiming, RerankResult, deduplicate_candidates
from .config import RerankerConfig

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Score query/passage pairs with one persistent local model instance."""

    def __init__(self, config: RerankerConfig | None = None, *, model: Any = None) -> None:
        self.config = config or RerankerConfig.from_env()
        load_started = perf_counter()
        self.model = model if model is not None else self._load_model()
        self.model_load_seconds = perf_counter() - load_started
        self.device = (
            getattr(self, "device", None)
            or getattr(self.model, "device", None)
            or self.config.device
            or "injected"
        )
        self.last_timing = RerankPhaseTiming(
            model_load_seconds=self.model_load_seconds,
            tokenizer_seconds=0.0,
            preprocessing_seconds=0.0,
            inference_seconds=0.0,
            postprocessing_seconds=0.0,
            total_seconds=0.0,
            device=self.device,
            pair_count=0,
            model_calls=0,
            batched=self.config.batch_all_candidates,
        )

    def _load_model(self) -> Any:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "sentence-transformers is required for CrossEncoderReranker"
            ) from exc

        self.device = detect_inference_device(self.config.device)
        logger.info(
            "Loading reranker model=%s max_length=%s device=%s",
            self.config.model_name,
            self.config.max_length,
            self.device,
        )
        model = CrossEncoder(
            self.config.model_name,
            max_length=self.config.max_length,
            trust_remote_code=self.config.trust_remote_code,
            device=self.device,
        )
        self._repair_gte_position_buffer(model)
        return model

    @staticmethod
    def _repair_gte_position_buffer(model: Any) -> None:
        """Repair the known GTE custom-model non-persistent buffer issue."""

        underlying = getattr(model, "model", model)
        new_model = getattr(underlying, "new", None)
        embeddings = getattr(new_model, "embeddings", None)
        position_ids = getattr(embeddings, "position_ids", None)
        if position_ids is None:
            return
        try:
            import torch

            expected = torch.arange(position_ids.numel(), device=position_ids.device)
            if not torch.equal(position_ids, expected):
                embeddings.position_ids = expected
                logger.warning("Repaired corrupted GTE position_ids buffer")
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("Unable to validate the GTE position_ids buffer") from exc

    @staticmethod
    def _sync_device(device: str) -> None:
        try:
            import torch

            if device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.synchronize()
            elif device.startswith("mps") and hasattr(torch, "mps"):
                torch.mps.synchronize()
        except Exception:  # pragma: no cover
            logger.debug("device synchronization unavailable", exc_info=True)

    @staticmethod
    def _direct_model_available(model: Any) -> bool:
        try:
            return getattr(model[0], "tokenizer", None) is not None
        except (IndexError, KeyError, TypeError, AttributeError):
            return False

    @staticmethod
    def _as_scores(raw_scores: Any, expected: int) -> list[float]:
        values = raw_scores.tolist() if hasattr(raw_scores, "tolist") else raw_scores
        if isinstance(values, (int, float)):
            values = [values]
        flattened: list[float] = []
        for value in values:
            if isinstance(value, (list, tuple)):
                if len(value) != 1:
                    raise ValueError("reranker must return one scalar score per candidate")
                value = value[0]
            flattened.append(float(value))
        if len(flattened) != expected:
            raise RuntimeError(
                f"reranker returned {len(flattened)} scores for {expected} candidates"
            )
        return flattened

    def _score_pairs_direct(
        self, pairs: list[tuple[str, str]]
    ) -> tuple[list[float], dict[str, float | int]]:
        """Run explicit tokenizer, device preparation, and model phases."""

        import torch
        from sentence_transformers.util import batch_to_device

        tokenizer = self.model[0].tokenizer
        batch_size = len(pairs) if self.config.batch_all_candidates else self.config.batch_size
        tokenizer_seconds = 0.0
        preprocessing_seconds = 0.0
        inference_seconds = 0.0
        postprocessing_seconds = 0.0
        scores: list[float] = []
        model_calls = 0
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start : start + batch_size]
            token_started = perf_counter()
            features = tokenizer(
                [pair[0] for pair in batch],
                [pair[1] for pair in batch],
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="pt",
            )
            tokenizer_seconds += perf_counter() - token_started

            preprocessing_started = perf_counter()
            features = batch_to_device(features, self.device)
            preprocessing_seconds += perf_counter() - preprocessing_started

            inference_started = perf_counter()
            with torch.inference_mode():
                output = self.model(features)
            self._sync_device(self.device)
            inference_seconds += perf_counter() - inference_started
            model_calls += 1

            post_started = perf_counter()
            scores.extend(self._as_scores(output["scores"], len(batch)))
            postprocessing_seconds += perf_counter() - post_started

        return scores, {
            "tokenizer_seconds": tokenizer_seconds,
            "preprocessing_seconds": preprocessing_seconds,
            "inference_seconds": inference_seconds,
            "postprocessing_seconds": postprocessing_seconds,
            "model_calls": model_calls,
        }

    def warmup(
        self,
        *,
        query: str = "warmup query",
        passage: str = "warmup passage",
    ) -> RerankPhaseTiming:
        """Initialize tokenizer/model/device kernels before timed requests."""

        if not self.config.warmup_enabled:
            return self.last_timing
        warmup_candidates = [
            HybridResult(
                chunk_id=f"__warmup_{i}__",
                document_id=f"__warmup_{i}__",
                text=f"{passage} {i}",
                rrf_score=0.0,
                vector_score=None,
                bm25_score=None,
                metadata={},
            )
            for i in range(max(self.config.warmup_candidates, self.config.candidate_top_k, 5))
        ]
        self.rerank(query, warmup_candidates, top_k=min(self.config.candidate_top_k, 3))
        return self.last_timing

    def rerank(
        self,
        query: str,
        candidates: Sequence[HybridResult],
        top_k: int,
    ) -> list[RerankResult]:
        if top_k <= 0:
            raise ValueError("top_k must be > 0")
        if not query or not query.strip() or not candidates:
            return []

        bounded = deduplicate_candidates(candidates)[: self.config.candidate_top_k]
        if not bounded:
            return []
        started = perf_counter()
        pairs = [(query.strip(), candidate.text or "") for candidate in bounded]
        if self._direct_model_available(self.model):
            scores, phase = self._score_pairs_direct(pairs)
        else:
            inference_started = perf_counter()
            raw_scores = self.model.predict(
                pairs,
                batch_size=len(pairs) if self.config.batch_all_candidates else self.config.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            scores = self._as_scores(raw_scores, len(bounded))
            phase = {
                "tokenizer_seconds": 0.0,
                "preprocessing_seconds": 0.0,
                "inference_seconds": perf_counter() - inference_started,
                "postprocessing_seconds": 0.0,
                "model_calls": 1 if self.config.batch_all_candidates else len(pairs),
            }

        post_started = perf_counter()
        scored = list(zip(bounded, scores))
        scored.sort(key=lambda item: (-item[1], -item[0].rrf_score, item[0].chunk_id))
        results = [
            RerankResult.from_candidate(candidate, rerank_score=score, rerank_rank=rank)
            for rank, (candidate, score) in enumerate(scored[:top_k], start=1)
        ]
        phase["postprocessing_seconds"] = float(phase["postprocessing_seconds"]) + (
            perf_counter() - post_started
        )
        self.last_timing = RerankPhaseTiming(
            model_load_seconds=self.model_load_seconds,
            tokenizer_seconds=float(phase["tokenizer_seconds"]),
            preprocessing_seconds=float(phase["preprocessing_seconds"]),
            inference_seconds=float(phase["inference_seconds"]),
            postprocessing_seconds=float(phase["postprocessing_seconds"]),
            total_seconds=perf_counter() - started,
            device=self.device,
            pair_count=len(bounded),
            model_calls=int(phase["model_calls"]),
            batched=self.config.batch_all_candidates,
        )
        return results


def detect_inference_device(requested: str | None = None) -> str:
    """Choose CUDA, MPS, or CPU, or validate an explicitly requested device."""

    try:
        import torch
    except ImportError:  # pragma: no cover
        return requested or "cpu"
    if requested:
        normalized = requested.lower()
        if normalized.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"requested device {requested!r} is unavailable")
        mps_backend = getattr(torch.backends, "mps", None)
        if normalized.startswith("mps") and (mps_backend is None or not mps_backend.is_available()):
            raise RuntimeError(f"requested device {requested!r} is unavailable")
        return requested
    if torch.cuda.is_available():
        return "cuda"
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return "mps"
    return "cpu"

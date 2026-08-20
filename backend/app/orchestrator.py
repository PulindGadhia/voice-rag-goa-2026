"""Application orchestrator with three-tier generation architecture (Extractive / Local / Groq)."""

from __future__ import annotations

import asyncio
import logging
import uuid
from time import perf_counter

from .config import Settings, get_settings
from .context import build_context
from .generation import GenerationService, parse_generated_output
from .generation.local import LocalCausalLMGenerator
from .guardrails import InputGuardrails, OutputGuardrails
from .observability.metrics import MetricsCollector, RequestRecord
from .retrieval import RetrievalEngine, build_existing_retrieval_engine
from .retrieval.contract import RetrievalResponse
from .schemas.requests import TextQueryRequest
from .schemas.responses import (
    CitationResponse,
    GenerationInfo,
    LatencyInfo,
    QueryResponse,
    RetrievalInfo,
)


logger = logging.getLogger(__name__)


class ResponseCache:
    """Thread-safe bounded in-memory cache for deterministic query responses."""

    def __init__(self, max_size: int = 128) -> None:
        self.max_size = max_size
        self._cache: dict[str, QueryResponse] = {}

    def _key(self, query: str, language: str, top_k: int | None) -> str:
        return f"{query.strip().lower()}:{language}:{top_k or 0}"

    def get(self, query: str, language: str = "en", top_k: int | None = None) -> QueryResponse | None:
        key = self._key(query, language, top_k)
        return self._cache.get(key)

    def set(
        self, query: str, response: QueryResponse, language: str = "en", top_k: int | None = None
    ) -> None:
        if len(self._cache) >= self.max_size:
            first_key = next(iter(self._cache))
            del self._cache[first_key]
        key = self._key(query, language, top_k)
        self._cache[key] = response

    def clear(self) -> None:
        self._cache.clear()


class Orchestrator:
    """Three-tier RAG orchestrator: Extractive (Tier 0) -> Local (Tier 1) -> Groq (Tier 2)."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        retrieval_engine: RetrievalEngine | None = None,
        generator: GenerationService | None = None,
        local_generator: LocalCausalLMGenerator | None = None,
        stt_provider: object | None = None,
        metrics: MetricsCollector | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.retrieval_engine = retrieval_engine
        self.generator = generator
        self.local_generator = local_generator
        self.stt_provider = stt_provider
        self.metrics = metrics or MetricsCollector()
        self.input_guardrails = InputGuardrails()
        self.output_guardrails = OutputGuardrails()
        self.response_cache = ResponseCache()
        self._translator = None
        self._translation_cache = None
        self._ready = False

    @property
    def is_ready(self) -> bool:
        return self._ready

    def get_provider_status(self) -> dict[str, bool]:
        groq_ready = bool(self.settings.groq_api_key or self.settings.llm_provider != "groq")
        local_ready = self.local_generator is not None and getattr(self.local_generator, "model", None) is not None
        return {
            "local": local_ready,
            "groq": groq_ready,
            "extractive": self.settings.extractive_enabled,
            "cache": True,
        }

    async def startup(self) -> None:
        startup_start = perf_counter()
        if self.retrieval_engine is None:
            self.retrieval_engine = await asyncio.to_thread(
                build_existing_retrieval_engine, app_settings=self.settings
            )
        warmup = getattr(self.retrieval_engine, "warmup", None)
        if warmup is not None:
            try:
                await warmup()
            except Exception as exc:
                self._ready = False
                raise RuntimeError(
                    "application startup failed: reranker warmup could not complete"
                ) from exc
        if self.generator is None:
            self.generator = GenerationService(
                provider=self.settings.llm_provider,
                model_name=self.settings.llm_model,
                api_key=self.settings.active_llm_api_key,
            )
        gen_warmup_async = getattr(self.generator, "warmup_async", None)
        if gen_warmup_async is not None:
            try:
                await gen_warmup_async()
            except Exception:
                pass
        else:
            gen_warmup = getattr(self.generator, "warmup", None)
            if gen_warmup is not None:
                try:
                    await asyncio.to_thread(gen_warmup)
                except Exception:
                    pass

        if self.local_generator is None and self.settings.two_tier_enabled:
            try:
                self.local_generator = LocalCausalLMGenerator(
                    model_path_or_id=self.settings.local_model_path
                )
                self.local_generator.warmup()
            except Exception as exc:
                logger.warning("Local generator initialization skipped/failed: %s", exc)
                self.local_generator = None

        if self.stt_provider is None and self.settings.stt_provider == "sarvam":
            from .providers.stt.sarvam import SarvamSTTProvider

            self.stt_provider = SarvamSTTProvider(api_key=self.settings.sarvam_api_key)
        self._ready = True

        # Initialize multilingual translation support
        if self.settings.multilingual_enabled:
            try:
                from voice_rag_ingestion.translation import build_translator
                from voice_rag_ingestion.translation_cache import TranslationCache

                self._translator = build_translator(
                    provider=self.settings.translation_provider,
                    api_key=self.settings.sarvam_api_key,
                )
                self._translation_cache = TranslationCache(
                    max_size=self.settings.translation_cache_size
                )
                logger.info(
                    "Multilingual support enabled: provider=%s, cache_size=%d",
                    self.settings.translation_provider,
                    self.settings.translation_cache_size,
                )
            except Exception as exc:
                logger.warning("Multilingual init failed: %s; disabled", exc)
                self._translator = None
                self._translation_cache = None

        startup_elapsed_ms = (perf_counter() - startup_start) * 1000.0
        self.metrics.set_startup_time(startup_elapsed_ms)
        logger.info("Application startup completed in %.2f ms", startup_elapsed_ms)

    async def warmup(self) -> dict[str, float]:
        """Explicit warmup hook for production readiness probes and endpoints."""
        warmup_start = perf_counter()
        if self.retrieval_engine is not None:
            warmup = getattr(self.retrieval_engine, "warmup", None)
            if warmup is not None:
                await warmup()
        if self.local_generator is not None:
            try:
                self.local_generator.warmup()
            except Exception:
                pass
        return {"warmup_ms": round((perf_counter() - warmup_start) * 1000.0, 2)}

    async def shutdown(self) -> None:
        if self.generator is not None:
            await self.generator.close()
        self.local_generator = None
        self._ready = False

    def decide_route(
        self,
        retrieval: RetrievalResponse,
        query: str,
        language: str,
    ) -> str:
        """Classify query evidence: EXTRACTIVE_ELIGIBLE, LOCAL_ELIGIBLE, LOCAL_UNCERTAIN, or GROQ_REQUIRED."""
        if not retrieval.found or not retrieval.results:
            return "GROQ_REQUIRED"
        top_candidate = retrieval.results[0]
        top_score = top_candidate.score

        # Use route-aware thresholds: RRF scores (~0.01-0.03) are on a different
        # scale than reranker scores (~0.5-1.0).
        is_fast_path = retrieval.retrieval_route == "fast"
        extractive_threshold = (
            self.settings.fast_path_extractive_threshold
            if is_fast_path
            else self.settings.extractive_confidence_threshold
        )
        local_threshold = (
            0.005 if is_fast_path else self.settings.local_confidence_threshold
        )

        if (
            self.settings.extractive_enabled
            and top_score >= extractive_threshold
            and top_candidate.text.strip()
        ):
            return "EXTRACTIVE_ELIGIBLE"
        elif top_score >= local_threshold:
            return "LOCAL_ELIGIBLE"
        elif top_score >= 0.0:
            return "LOCAL_UNCERTAIN"
        else:
            return "GROQ_REQUIRED"

    def extract_answer_from_passage(self, passage: str, query: str) -> str:
        """Extract high-confidence direct answer passage without LLM generation."""
        text = passage.strip()
        sentences = [s.strip() for s in text.split(". ") if s.strip()]
        if len(sentences) <= 2:
            return text
        return ". ".join(sentences[:2]) + ("." if not sentences[1].endswith(".") else "")

    async def process_text_query(
        self, request: TextQueryRequest, *, use_cache: bool = True
    ) -> QueryResponse:
        if not self._ready or self.retrieval_engine is None or self.generator is None:
            raise RuntimeError("orchestrator is not ready")
        request_id = str(uuid.uuid4())[:8]
        started = perf_counter()

        if use_cache:
            cached = self.response_cache.get(
                request.query, language=request.language, top_k=request.top_k
            )
            if cached is not None:
                total_ms = (perf_counter() - started) * 1000.0
                resp = QueryResponse(
                    request_id=request_id,
                    query=cached.query,
                    answer=cached.answer,
                    status=cached.status,
                    grounded=cached.grounded,
                    citations=cached.citations,
                    retrieval=cached.retrieval,
                    generation=GenerationInfo(
                        provider="cache",
                        model=None,
                        route="cache",
                        confidence_decision="CACHE_HIT",
                        fallback_used=False,
                    ),
                    latency=LatencyInfo(
                        retrieval_ms=0.0,
                        generation_ms=0.0,
                        extractive_generation_ms=0.0,
                        local_generation_ms=0.0,
                        groq_generation_ms=0.0,
                        grounding_ms=0.0,
                        total_ms=total_ms,
                    ),
                    guardrail_status=cached.guardrail_status,
                    guardrail_reason=cached.guardrail_reason,
                )
                self.metrics.record(
                    RequestRecord(
                        total_ms=resp.latency.total_ms,
                        status=resp.status,
                        route="cache",
                    )
                )
                return resp

        verdicts = self.input_guardrails.check(request.query)
        blocked = next((v for v in verdicts if not v.passed), None)
        if blocked is not None:
            response = QueryResponse(
                request_id=request_id,
                query=request.query,
                answer=None,
                status="refused",
                generation=GenerationInfo(
                    provider=None,
                    model=None,
                    route=None,
                    confidence_decision="INPUT_BLOCKED",
                    fallback_used=False,
                ),
                latency=LatencyInfo(total_ms=(perf_counter() - started) * 1000.0),
                guardrail_status="blocked",
                guardrail_reason=blocked.reason,
            )
            self.metrics.record(
                RequestRecord(
                    total_ms=response.latency.total_ms,
                    status=response.status,
                    route="blocked",
                )
            )
            return response

        # --- Language detection ---
        from voice_rag_ingestion.language import detect_language as _detect_lang
        lang_result = _detect_lang(request.query)
        detected_language = lang_result.language_code

        try:
            retrieve_multilingual = getattr(self.retrieval_engine, "retrieve_multilingual", None)
            if (
                self.settings.multilingual_enabled
                and self.settings.adaptive_routing_enabled
                and retrieve_multilingual is not None
            ):
                retrieval = await retrieve_multilingual(
                    request.query,
                    top_k=request.top_k or self.settings.top_k,
                    candidate_top_k=self.settings.candidate_top_k,
                    rrf_threshold=self.settings.fast_path_rrf_threshold,
                    min_agreeing_sources=self.settings.fast_path_min_agreeing_sources,
                    bm25_threshold=self.settings.fast_path_bm25_threshold,
                    source_language=detected_language,
                    translator=self._translator,
                    translation_cache=self._translation_cache,
                )
            elif self.settings.adaptive_routing_enabled and getattr(self.retrieval_engine, "retrieve_adaptive", None) is not None:
                retrieval = await self.retrieval_engine.retrieve_adaptive(
                    request.query,
                    top_k=request.top_k or self.settings.top_k,
                    candidate_top_k=self.settings.candidate_top_k,
                    rrf_threshold=self.settings.fast_path_rrf_threshold,
                    min_agreeing_sources=self.settings.fast_path_min_agreeing_sources,
                    bm25_threshold=self.settings.fast_path_bm25_threshold,
                )
            else:
                retrieval = await self.retrieval_engine.retrieve(
                    request.query,
                    top_k=request.top_k or self.settings.top_k,
                    candidate_top_k=self.settings.candidate_top_k,
                )
        except Exception:
            response = QueryResponse(
                request_id=request_id,
                query=request.query,
                answer=None,
                status="error",
                generation=GenerationInfo(
                    provider=None,
                    model=None,
                    route=None,
                    confidence_decision="RETRIEVAL_ERROR",
                    fallback_used=False,
                ),
                latency=LatencyInfo(total_ms=(perf_counter() - started) * 1000.0),
                guardrail_status="error",
                guardrail_reason="Retrieval failed",
            )
            self.metrics.record(
                RequestRecord(
                    total_ms=response.latency.total_ms,
                    status=response.status,
                    route="error",
                )
            )
            return response

        if not retrieval.found:
            response = QueryResponse(
                request_id=request_id,
                query=request.query,
                answer=None,
                status="refused",
                retrieval=RetrievalInfo(
                    top_k=retrieval.top_k,
                    candidate_top_k=retrieval.candidate_top_k,
                    num_results=0,
                    total_latency_ms=retrieval.total_latency_ms,
                    hybrid_latency_ms=retrieval.hybrid_latency_ms,
                    rerank_latency_ms=retrieval.rerank_latency_ms,
                    embedding_latency_ms=retrieval.embedding_latency_ms,
                    vector_search_latency_ms=retrieval.vector_search_latency_ms,
                    bm25_latency_ms=retrieval.bm25_latency_ms,
                    rrf_latency_ms=retrieval.rrf_latency_ms,
                    model_name=retrieval.model_name,
                    device=retrieval.device,
                    retrieval_route=retrieval.retrieval_route,
                ),
                generation=GenerationInfo(
                    provider=None,
                    model=None,
                    route=None,
                    confidence_decision="NO_PASSAGES",
                    fallback_used=False,
                ),
                latency=LatencyInfo(
                    retrieval_ms=retrieval.total_latency_ms,
                    total_ms=(perf_counter() - started) * 1000.0,
                ),
                guardrail_status="low_confidence",
                guardrail_reason="No relevant passages found",
            )
            self.metrics.record(
                RequestRecord(
                    total_ms=response.latency.total_ms,
                    status=response.status,
                    retrieval_ms=retrieval.total_latency_ms,
                    rerank_ms=retrieval.rerank_latency_ms,
                    route="refused",
                )
            )
            return response

        built_context = build_context(request.query, retrieval.results)
        passages = [item.text for item in retrieval.results]
        
        # Three-tier route decision
        route_decision = self.decide_route(retrieval, request.query, request.language)

        extractive_gen_ms = 0.0
        local_gen_ms = 0.0
        groq_gen_ms = 0.0
        chosen_answer: str | None = None
        active_route = "groq"
        fallback_used = False
        active_provider = getattr(self.generator, "provider", "groq")
        active_model = getattr(self.generator, "model_name", "openai/gpt-oss-20b")
        structured_source_ids: set[str] | None = None
        extractive_grounded: bool | None = None

        # --- TIER 0: DIRECT EXTRACTIVE RAG (<200 ms Fast Path) ---
        if route_decision == "EXTRACTIVE_ELIGIBLE" and retrieval.results:
            ext_start = perf_counter()
            top_passage = retrieval.results[0].text
            extracted = self.extract_answer_from_passage(top_passage, request.query)
            extractive_gen_ms = (perf_counter() - ext_start) * 1000.0

            grounding_test = self.output_guardrails.check_grounding_substring(extracted, top_passage)
            quality_test = self.output_guardrails.check_answer_quality(extracted)
            if grounding_test.passed and quality_test.passed:
                chosen_answer = extracted
                active_route = "extractive"
                active_provider = "retrieval"
                active_model = None
                extractive_grounded = True

                # Translate extractive answer back to user's language
                if (
                    detected_language != "en"
                    and self._translator is not None
                    and self.settings.multilingual_enabled
                ):
                    try:
                        # Check answer translation cache
                        cached_answer = None
                        if self._translation_cache is not None:
                            cached_answer = self._translation_cache.get(
                                chosen_answer, "en", detected_language
                            )
                        if cached_answer is not None:
                            chosen_answer = cached_answer
                        else:
                            translated_answer = await asyncio.to_thread(
                                self._translator.translate,
                                chosen_answer,
                                "en",
                                detected_language,
                            )
                            if translated_answer and translated_answer != chosen_answer:
                                if self._translation_cache is not None:
                                    self._translation_cache.put(
                                        chosen_answer, "en", detected_language, translated_answer
                                    )
                                chosen_answer = translated_answer
                    except Exception as exc:
                        logger.warning("Answer translation failed: %s", exc)
            else:
                logger.info("Extractive candidate failed validation; falling back to local/groq")
                fallback_used = True
                route_decision = "LOCAL_ELIGIBLE"

        # --- TIER 1: LOCAL LLM GENERATION ---
        if (
            chosen_answer is None
            and self.settings.two_tier_enabled
            and self.local_generator is not None
            and route_decision in {"LOCAL_ELIGIBLE", "LOCAL_UNCERTAIN"}
        ):
            local_start = perf_counter()
            try:
                raw_local = await asyncio.to_thread(
                    self.local_generator.generate_rag_sync,
                    request.query,
                    built_context.text,
                    48,
                )
                local_gen_ms = (perf_counter() - local_start) * 1000.0
                clean_local = raw_local.strip()
                if "INSUFFICIENT_CONTEXT" in clean_local.upper() or not clean_local:
                    fallback_used = True
                else:
                    grounding_test = self.output_guardrails.check_grounding(clean_local, passages)
                    quality_test = self.output_guardrails.check_answer_quality(clean_local)
                    if grounding_test.passed and quality_test.passed:
                        chosen_answer = clean_local
                        active_route = "local"
                        active_provider = "local"
                        active_model = self.settings.local_model_path
                    else:
                        fallback_used = True
            except Exception as exc:
                local_gen_ms = (perf_counter() - local_start) * 1000.0
                logger.warning("Local generation error: %s; falling back to Groq", exc)
                fallback_used = True

        # --- TIER 2: GROQ CLOUD FALLBACK ---
        if chosen_answer is None:
            groq_start = perf_counter()
            try:
                raw_answer, groq_gen_ms = await self.generator.generate(
                    question=request.query,
                    context=built_context.text,
                    passages=passages,
                    language=request.language,
                )
                source_ids = {source.source_id for source in built_context.sources}
                source_aliases = {
                    alias: source.source_id
                    for source in built_context.sources
                    for alias in (source.source_id, source.chunk_id, source.document_id)
                }
                structured = parse_generated_output(
                    raw_answer, source_ids, source_aliases=source_aliases
                )
                structured_source_ids = set(structured.source_ids)
                chosen_answer = structured.answer
                active_route = "groq"
                active_provider = getattr(self.generator, "provider", "groq")
                active_model = getattr(self.generator, "model_name", "openai/gpt-oss-20b")
            except Exception as exc:
                api_key = getattr(self.generator, "api_key", "")
                safe_message = str(exc).replace(api_key, "<redacted>")
                logger.warning(
                    "generation_request_failed provider=%s model=%s "
                    "error_type=%s error=%s",
                    getattr(self.generator, "provider", "unknown"),
                    getattr(self.generator, "model_name", "unknown"),
                    type(exc).__name__,
                    safe_message[:500],
                )
                response = QueryResponse(
                    request_id=request_id,
                    query=request.query,
                    answer=None,
                    status="error",
                    retrieval=RetrievalInfo(
                        top_k=retrieval.top_k,
                        candidate_top_k=retrieval.candidate_top_k,
                        num_results=len(retrieval.results),
                        total_latency_ms=retrieval.total_latency_ms,
                        hybrid_latency_ms=retrieval.hybrid_latency_ms,
                        rerank_latency_ms=retrieval.rerank_latency_ms,
                        embedding_latency_ms=retrieval.embedding_latency_ms,
                        vector_search_latency_ms=retrieval.vector_search_latency_ms,
                        bm25_latency_ms=retrieval.bm25_latency_ms,
                        rrf_latency_ms=retrieval.rrf_latency_ms,
                        model_name=retrieval.model_name,
                        device=retrieval.device,
                        retrieval_route=retrieval.retrieval_route,
                    ),
                    generation=GenerationInfo(
                        provider=getattr(self.generator, "provider", "groq"),
                        model=getattr(self.generator, "model_name", "openai/gpt-oss-20b"),
                        route=active_route,
                        confidence_decision=route_decision,
                        fallback_used=fallback_used,
                    ),
                    latency=LatencyInfo(
                        retrieval_ms=retrieval.total_latency_ms,
                        generation_ms=(perf_counter() - groq_start) * 1000.0,
                        extractive_generation_ms=extractive_gen_ms,
                        local_generation_ms=local_gen_ms,
                        groq_generation_ms=(perf_counter() - groq_start) * 1000.0,
                        total_ms=(perf_counter() - started) * 1000.0,
                    ),
                    guardrail_status="error",
                    guardrail_reason=f"Generation failed: {type(exc).__name__}: {safe_message[:200]}",
                )
                self.metrics.record(
                    RequestRecord(
                        total_ms=response.latency.total_ms,
                        status=response.status,
                        retrieval_ms=retrieval.total_latency_ms,
                        rerank_ms=retrieval.rerank_latency_ms,
                        generation_ms=response.latency.generation_ms or 0.0,
                        route=active_route,
                    )
                )
                return response

        if not chosen_answer.strip():
            response = QueryResponse(
                request_id=request_id,
                query=request.query,
                answer=None,
                status="refused",
                retrieval=RetrievalInfo(
                    top_k=retrieval.top_k,
                    candidate_top_k=retrieval.candidate_top_k,
                    num_results=len(retrieval.results),
                    total_latency_ms=retrieval.total_latency_ms,
                    hybrid_latency_ms=retrieval.hybrid_latency_ms,
                    rerank_latency_ms=retrieval.rerank_latency_ms,
                    embedding_latency_ms=retrieval.embedding_latency_ms,
                    vector_search_latency_ms=retrieval.vector_search_latency_ms,
                    bm25_latency_ms=retrieval.bm25_latency_ms,
                    rrf_latency_ms=retrieval.rrf_latency_ms,
                    model_name=retrieval.model_name,
                    device=retrieval.device,
                    retrieval_route=retrieval.retrieval_route,
                ),
                generation=GenerationInfo(
                    provider=active_provider,
                    model=active_model,
                    route=active_route,
                    confidence_decision=route_decision,
                    fallback_used=fallback_used,
                ),
                latency=LatencyInfo(
                    retrieval_ms=retrieval.total_latency_ms,
                    generation_ms=groq_gen_ms if active_route == "groq" else (local_gen_ms if active_route == "local" else extractive_gen_ms),
                    extractive_generation_ms=extractive_gen_ms,
                    local_generation_ms=local_gen_ms,
                    groq_generation_ms=groq_gen_ms,
                    total_ms=(perf_counter() - started) * 1000.0,
                ),
                guardrail_status="low_confidence",
                guardrail_reason="Model declined to answer: insufficient context",
            )
            if use_cache:
                self.response_cache.set(
                    request.query, response, language=request.language, top_k=request.top_k
                )
            self.metrics.record(
                RequestRecord(
                    total_ms=response.latency.total_ms,
                    status=response.status,
                    retrieval_ms=retrieval.total_latency_ms,
                    rerank_ms=retrieval.rerank_latency_ms,
                    generation_ms=response.latency.generation_ms or 0.0,
                    route=active_route,
                )
            )
            return response

        grounding_started = perf_counter()
        if extractive_grounded is not None:
            # Extractive path already validated grounding — skip redundant check
            grounded = extractive_grounded
        else:
            grounding = self.output_guardrails.check_grounding(chosen_answer, passages)
            quality = self.output_guardrails.check_answer_quality(chosen_answer)
            grounded = grounding.passed and quality.passed
        citations = [
            CitationResponse(
                chunk_id=item.chunk_id,
                document_id=item.document_id,
                text=item.text[:500],
                score=round(item.score, 6),
                metadata=item.metadata,
            )
            for item in retrieval.results
        ]
        
        gen_duration = (
            extractive_gen_ms if active_route == "extractive"
            else (local_gen_ms if active_route == "local" else groq_gen_ms)
        )

        response = QueryResponse(
            request_id=request_id,
            query=request.query,
            answer=chosen_answer,
            status="answered",
            grounded=grounded,
            citations=citations,
            retrieval=RetrievalInfo(
                top_k=retrieval.top_k,
                candidate_top_k=retrieval.candidate_top_k,
                num_results=len(retrieval.results),
                total_latency_ms=retrieval.total_latency_ms,
                hybrid_latency_ms=retrieval.hybrid_latency_ms,
                rerank_latency_ms=retrieval.rerank_latency_ms,
                embedding_latency_ms=retrieval.embedding_latency_ms,
                vector_search_latency_ms=retrieval.vector_search_latency_ms,
                bm25_latency_ms=retrieval.bm25_latency_ms,
                rrf_latency_ms=retrieval.rrf_latency_ms,
                model_name=retrieval.model_name,
                device=retrieval.device,
                retrieval_route=retrieval.retrieval_route,
                translation_used=getattr(retrieval, "translation_used", False),
                detected_language=getattr(retrieval, "detected_language", None),
            ),
            generation=GenerationInfo(
                provider=active_provider,
                model=active_model,
                route=active_route,
                confidence_decision=route_decision,
                fallback_used=fallback_used,
            ),
            latency=LatencyInfo(
                retrieval_ms=retrieval.total_latency_ms,
                generation_ms=gen_duration,
                extractive_generation_ms=extractive_gen_ms,
                local_generation_ms=local_gen_ms,
                groq_generation_ms=groq_gen_ms,
                grounding_ms=(perf_counter() - grounding_started) * 1000.0,
                total_ms=(perf_counter() - started) * 1000.0,
            ),
            guardrail_status="passed" if grounded else "ungrounded",
            guardrail_reason=None if grounded else grounding.reason,
            source_language=detected_language,
            answer_language=detected_language if active_route == "extractive" and detected_language != "en" else "en",
            translation_used=getattr(retrieval, "translation_used", False) or (detected_language != "en" and active_route == "extractive"),
        )
        if use_cache:
            self.response_cache.set(
                request.query, response, language=request.language, top_k=request.top_k
            )
        self.metrics.record(
            RequestRecord(
                total_ms=response.latency.total_ms,
                status=response.status,
                retrieval_ms=retrieval.total_latency_ms,
                rerank_ms=retrieval.rerank_latency_ms,
                generation_ms=gen_duration,
                grounding_ms=response.latency.grounding_ms or 0.0,
                route=active_route,
            )
        )
        return response

    async def process_voice_query(
        self,
        audio_bytes: bytes,
        *,
        content_type: str = "audio/wav",
        language: str = "en",
        top_k: int | None = None,
    ) -> QueryResponse:
        if self.stt_provider is None:
            raise RuntimeError("STT provider is not configured")
        result = await self.stt_provider.transcribe(audio_bytes, content_type=content_type)
        return await self.process_text_query(
            TextQueryRequest(query=result.text, language=language, top_k=top_k)
        )

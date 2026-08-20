"""Provider-neutral LLM generation service adapted from Om's backend."""

from __future__ import annotations

import asyncio
import logging
from time import perf_counter


logger = logging.getLogger(__name__)


class GenerationService:
    def __init__(
        self, *, provider: str, model_name: str, api_key: str = "", max_retries: int = 1
    ) -> None:
        self.provider = provider.lower()
        self.model_name = model_name
        self.api_key = api_key
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        self.max_retries = max_retries
        self._client = None
        self._local_generator = None

    async def warmup_async(self) -> None:
        """Warm up persistent client, connection pool, or local model."""
        if self.provider == "groq" and self.api_key:
            from groq import AsyncGroq

            if self._client is None or not isinstance(self._client, AsyncGroq):
                self._client = AsyncGroq(api_key=self.api_key)
            try:
                await self._client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": "ping"}],
                    max_tokens=10,
                )
            except Exception:
                pass
        elif self.provider == "local":
            from .local import LocalCausalLMGenerator

            if self._local_generator is None:
                self._local_generator = LocalCausalLMGenerator(model_path_or_id=self.model_name)
            self._local_generator.warmup()

    def warmup(self) -> None:
        """Warm up local generator or connection pool."""
        if self.provider == "local":
            from .local import LocalCausalLMGenerator

            if self._local_generator is None:
                self._local_generator = LocalCausalLMGenerator(model_path_or_id=self.model_name)
            self._local_generator.warmup()

    def _prompt(self, question: str, context: str, language: str) -> str:
        return (
            f"Context:\n{context}\n\n"
            f"Question ({language}): {question}\n\n"
            'Respond ONLY with a JSON object: {"answer": "concise answer", "grounded": true, "source_ids": ["source-1"], "confidence": 0.9}. '
            'If the context does not contain the answer, set "answer": "", "grounded": false, "source_ids": [], "confidence": 0.'
        )

    def _generate_sync(self, prompt: str) -> str:
        if self.provider == "local":
            from .local import LocalCausalLMGenerator

            if self._local_generator is None:
                self._local_generator = LocalCausalLMGenerator(model_path_or_id=self.model_name)
            return self._local_generator.generate_sync(prompt, max_new_tokens=256)
        if not self.api_key:
            raise RuntimeError(f"No API key configured for LLM provider {self.provider!r}")
        if self.provider == "groq":
            from groq import Groq

            self._client = self._client or Groq(api_key=self.api_key)
            extra_body = {"reasoning_effort": "low"} if "gpt-oss" in self.model_name else None
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=256,
                response_format={"type": "json_object"},
                extra_body=extra_body,
            )
            return response.choices[0].message.content or ""
        if self.provider == "gemini":
            from google import genai

            self._client = self._client or genai.Client(api_key=self.api_key)
            request: dict[str, object] = {
                "model": self.model_name,
                "contents": prompt,
            }
            try:
                from google.genai import types

                request["config"] = types.GenerateContentConfig(
                    temperature=0.1,
                    response_mime_type="application/json",
                )
            except ImportError:
                # The client itself will provide the useful dependency error
                # if the optional Gemini SDK is unavailable.
                pass
            response = self._client.models.generate_content(**request)
            return response.text or ""
        raise ValueError(f"Unsupported LLM provider: {self.provider}")

    async def _generate_async(self, prompt: str) -> str:
        if self.provider == "local":
            return await asyncio.to_thread(self._generate_sync, prompt)
        if not self.api_key:
            raise RuntimeError(f"No API key configured for LLM provider {self.provider!r}")
        if self.provider == "groq":
            from groq import AsyncGroq

            if self._client is None or not isinstance(self._client, AsyncGroq):
                self._client = AsyncGroq(api_key=self.api_key)
            extra_body = {"reasoning_effort": "low"} if "gpt-oss" in self.model_name else None
            response = await self._client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=256,
                response_format={"type": "json_object"},
                extra_body=extra_body,
            )
            return response.choices[0].message.content or ""
        return await asyncio.to_thread(self._generate_sync, prompt)

    async def generate(
        self, *, question: str, passages: list[str] | None = None,
        context: str | None = None, language: str = "en"
    ) -> tuple[str, float]:
        started = perf_counter()
        prompt_context = context or "\n\n".join(passages or [])
        prompt = self._prompt(question, prompt_context, language)
        error: Exception | None = None
        for _ in range(self.max_retries + 1):
            try:
                answer = await self._generate_async(prompt)
                break
            except Exception as exc:
                error = exc
                safe_message = str(exc).replace(self.api_key, "<redacted>")
                logger.warning(
                    "llm_generation_attempt_failed provider=%s model=%s "
                    "error_type=%s error=%s",
                    self.provider,
                    self.model_name,
                    type(exc).__name__,
                    safe_message[:500],
                )
        else:
            assert error is not None
            raise error
        return answer, (perf_counter() - started) * 1000.0

    async def close(self) -> None:
        if self._client is not None:
            close_method = getattr(self._client, "close", None)
            if close_method is not None:
                res = close_method()
                if asyncio.iscoroutine(res):
                    await res
            self._client = None
        self._local_generator = None

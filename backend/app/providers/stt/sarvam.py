"""Lazy Sarvam STT adapter; importing the app does not require the SDK."""

from __future__ import annotations

import asyncio
from time import perf_counter

from .base import STTResponse


class SarvamSTTProvider:
    def __init__(self, *, api_key: str, model: str = "saaras:v4") -> None:
        self.api_key = api_key
        self.model = model
        self._client = None

    def _transcribe_sync(self, audio_bytes: bytes, content_type: str) -> str:
        if not self.api_key:
            raise RuntimeError("SARVAM_API_KEY is not configured")
        try:
            from sarvamai import SarvamAI
        except ImportError as exc:
            raise RuntimeError("sarvamai is required for Sarvam STT") from exc
        self._client = self._client or SarvamAI(api_subscription_key=self.api_key)
        # The SDK has changed its upload surface across releases. Keep this
        # adapter intentionally small and fail with a useful message if the
        # installed version does not expose the expected transcription API.
        if not hasattr(self._client, "speech_to_text"):
            raise RuntimeError("installed sarvamai SDK lacks speech_to_text")
        result = self._client.speech_to_text.transcribe(
            file=("audio", audio_bytes, content_type), model=self.model
        )
        return getattr(result, "transcript", None) or getattr(result, "text", "")

    async def transcribe(self, audio_bytes: bytes, *, content_type: str) -> STTResponse:
        started = perf_counter()
        text = await asyncio.to_thread(self._transcribe_sync, audio_bytes, content_type)
        return STTResponse(text=text, duration_ms=(perf_counter() - started) * 1000.0)


from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class STTResponse:
    text: str
    language: str | None = None
    duration_ms: float = 0.0


class STTProvider(Protocol):
    async def transcribe(self, audio_bytes: bytes, *, content_type: str) -> STTResponse:
        ...


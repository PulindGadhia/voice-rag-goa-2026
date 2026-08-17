"""Deterministic embedding provider for offline development tests."""

from __future__ import annotations

import hashlib
from typing import Sequence


class HashEmbeddingProvider:
    """Stable hashed bag-of-tokens vector; not a quality production model."""

    def __init__(self, dimensions: int = 384) -> None:
        self._dimension = dimensions

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_text(self, text: str, *, input_type: str = "passage") -> list[float]:
        return self.embed_batch([text], input_type=input_type)[0]

    def embed_batch(
        self, texts: Sequence[str], *, input_type: str = "passage"
    ) -> list[list[float]]:
        result: list[list[float]] = []
        for text in texts:
            vector = [0.0] * self._dimension
            for token in f"{input_type}: {text}".casefold().split():
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                index = int.from_bytes(digest[:4], "big") % self._dimension
                vector[index] += 1.0 if digest[4] % 2 else -1.0
            norm = sum(value * value for value in vector) ** 0.5
            result.append([value / norm for value in vector] if norm else vector)
        return result

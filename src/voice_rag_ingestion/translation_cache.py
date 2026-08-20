"""Thread-safe LRU translation cache.

Avoids repeated translation API calls for identical queries.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field


@dataclass
class TranslationCacheStats:
    """Cache hit/miss statistics."""
    hits: int = 0
    misses: int = 0

    @property
    def total(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        return self.hits / max(self.total, 1)


class TranslationCache:
    """Bounded in-memory LRU cache for translations.

    Key: (text, source_lang, target_lang)
    Thread-safe for single-writer scenarios (Python GIL).
    """

    def __init__(self, max_size: int = 512) -> None:
        self._max_size = max_size
        self._cache: OrderedDict[tuple[str, str, str], str] = OrderedDict()
        self.stats = TranslationCacheStats()

    def get(self, text: str, source_lang: str, target_lang: str) -> str | None:
        """Look up a cached translation. Returns None on miss."""
        key = (text.strip(), source_lang, target_lang)
        result = self._cache.get(key)
        if result is not None:
            self._cache.move_to_end(key)
            self.stats.hits += 1
            return result
        self.stats.misses += 1
        return None

    def put(self, text: str, source_lang: str, target_lang: str, translated: str) -> None:
        """Store a translation in the cache."""
        key = (text.strip(), source_lang, target_lang)
        if key in self._cache:
            self._cache.move_to_end(key)
            self._cache[key] = translated
        else:
            if len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)
            self._cache[key] = translated

    @property
    def size(self) -> int:
        return len(self._cache)

    def clear(self) -> None:
        self._cache.clear()
        self.stats = TranslationCacheStats()


__all__ = ["TranslationCache", "TranslationCacheStats"]

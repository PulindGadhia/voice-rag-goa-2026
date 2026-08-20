"""Tests for translation cache."""

from __future__ import annotations

import pytest

from voice_rag_ingestion.translation_cache import TranslationCache


class TestTranslationCache:
    def test_put_and_get(self) -> None:
        cache = TranslationCache(max_size=10)
        cache.put("कंपनी क्या है?", "hi", "en", "What is a company?")
        result = cache.get("कंपनी क्या है?", "hi", "en")
        assert result == "What is a company?"

    def test_miss_returns_none(self) -> None:
        cache = TranslationCache(max_size=10)
        assert cache.get("unknown", "hi", "en") is None

    def test_different_lang_pairs_are_different_keys(self) -> None:
        cache = TranslationCache(max_size=10)
        cache.put("hello", "en", "hi", "नमस्ते")
        cache.put("hello", "en", "gu", "નમસ્તે")
        assert cache.get("hello", "en", "hi") == "नमस्ते"
        assert cache.get("hello", "en", "gu") == "નમસ્તે"

    def test_lru_eviction(self) -> None:
        cache = TranslationCache(max_size=3)
        cache.put("a", "en", "hi", "A")
        cache.put("b", "en", "hi", "B")
        cache.put("c", "en", "hi", "C")
        # Cache is full; adding a new entry should evict "a"
        cache.put("d", "en", "hi", "D")
        assert cache.get("a", "en", "hi") is None
        assert cache.get("b", "en", "hi") == "B"
        assert cache.get("d", "en", "hi") == "D"

    def test_access_refreshes_lru(self) -> None:
        cache = TranslationCache(max_size=3)
        cache.put("a", "en", "hi", "A")
        cache.put("b", "en", "hi", "B")
        cache.put("c", "en", "hi", "C")
        # Access "a" to refresh it
        cache.get("a", "en", "hi")
        # Now "b" is the oldest — should be evicted
        cache.put("d", "en", "hi", "D")
        assert cache.get("a", "en", "hi") == "A"
        assert cache.get("b", "en", "hi") is None

    def test_stats_tracking(self) -> None:
        cache = TranslationCache(max_size=10)
        cache.put("x", "en", "hi", "X")
        cache.get("x", "en", "hi")  # hit
        cache.get("y", "en", "hi")  # miss
        cache.get("z", "en", "hi")  # miss
        assert cache.stats.hits == 1
        assert cache.stats.misses == 2
        assert cache.stats.total == 3
        assert abs(cache.stats.hit_rate - 1 / 3) < 0.01

    def test_size_property(self) -> None:
        cache = TranslationCache(max_size=10)
        assert cache.size == 0
        cache.put("a", "en", "hi", "A")
        assert cache.size == 1
        cache.put("b", "en", "hi", "B")
        assert cache.size == 2

    def test_clear(self) -> None:
        cache = TranslationCache(max_size=10)
        cache.put("a", "en", "hi", "A")
        cache.put("b", "en", "hi", "B")
        cache.clear()
        assert cache.size == 0
        assert cache.get("a", "en", "hi") is None
        assert cache.stats.hits == 0

    def test_strips_whitespace(self) -> None:
        cache = TranslationCache(max_size=10)
        cache.put("  hello  ", "en", "hi", "नमस्ते")
        assert cache.get("hello", "en", "hi") == "नमस्ते"

    def test_update_existing_entry(self) -> None:
        cache = TranslationCache(max_size=10)
        cache.put("hello", "en", "hi", "old")
        cache.put("hello", "en", "hi", "new")
        assert cache.get("hello", "en", "hi") == "new"
        assert cache.size == 1

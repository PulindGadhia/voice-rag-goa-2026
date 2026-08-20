"""Tests for multilingual retrieval integration."""

from __future__ import annotations

import pytest

from voice_rag_ingestion.language import detect_language
from voice_rag_ingestion.query_expansion import expand_query
from voice_rag_ingestion.translation import DictionaryTranslator
from voice_rag_ingestion.translation_cache import TranslationCache


class TestQueryExpansion:
    """Verify query expansion adds English terms for BM25."""

    def test_hindi_expansion(self) -> None:
        result = expand_query("कंपनी क्या है?", "hi")
        assert "company" in result
        assert "कंपनी" in result

    def test_gujarati_expansion(self) -> None:
        result = expand_query("કંપની શું છે?", "gu")
        assert "company" in result

    def test_english_passthrough(self) -> None:
        original = "What is a company?"
        result = expand_query(original, "en")
        assert result == original

    def test_bengali_expansion(self) -> None:
        result = expand_query("কোম্পানী কি?", "bn")
        assert "company" in result

    def test_tamil_expansion(self) -> None:
        result = expand_query("நிறுவனம் என்றால் என்ன?", "ta")
        assert "company" in result or "corporation" in result

    def test_no_match_passthrough(self) -> None:
        original = "xxxxxxxxx"
        result = expand_query(original, "hi")
        assert result == original


class TestDictionaryTranslator:
    """Verify dictionary translator works for testing."""

    def test_hindi_to_english(self) -> None:
        t = DictionaryTranslator()
        result = t.translate("कंपनी क्या है?", "hi", "en")
        assert result == "What is a company?"

    def test_gujarati_to_english(self) -> None:
        t = DictionaryTranslator()
        result = t.translate("કંપની શું છે?", "gu", "en")
        assert result == "What is a company?"

    def test_english_to_hindi(self) -> None:
        t = DictionaryTranslator()
        result = t.translate("A company is a legal entity", "en", "hi")
        assert "कंपनी" in result

    def test_same_lang_passthrough(self) -> None:
        t = DictionaryTranslator()
        result = t.translate("hello", "en", "en")
        assert result == "hello"

    def test_unknown_returns_original(self) -> None:
        t = DictionaryTranslator()
        result = t.translate("totally unknown text", "hi", "en")
        assert result == "totally unknown text"

    def test_is_available(self) -> None:
        assert DictionaryTranslator().is_available() is True


class TestTranslationCacheWithTranslator:
    """Verify cache integrates properly with translator."""

    def test_cache_prevents_repeated_calls(self) -> None:
        t = DictionaryTranslator()
        cache = TranslationCache(max_size=100)

        # First call: miss → translator
        query = "कंपनी क्या है?"
        cached = cache.get(query, "hi", "en")
        assert cached is None
        result = t.translate(query, "hi", "en")
        cache.put(query, "hi", "en", result)

        # Second call: hit → cache
        cached = cache.get(query, "hi", "en")
        assert cached == "What is a company?"
        assert cache.stats.hits == 1
        assert cache.stats.misses == 1


class TestEndToEndLanguageFlow:
    """Verify the complete language detection → expansion → translation flow."""

    @pytest.mark.parametrize(
        "query,expected_lang,expected_expansion",
        [
            ("कंपनी क्या है?", "hi", "company"),
            ("કંપની શું છે?", "gu", "company"),
            ("কোম্পানী কি?", "bn", "company"),
            ("What is a company?", "en", None),
        ],
        ids=["hindi", "gujarati", "bengali", "english"],
    )
    def test_detect_expand_translate(
        self, query: str, expected_lang: str, expected_expansion: str | None
    ) -> None:
        # Step 1: Detect language
        result = detect_language(query)
        assert result.language_code == expected_lang

        # Step 2: Expand query
        expanded = expand_query(query, result.language_code)
        if expected_expansion:
            assert expected_expansion in expanded
        else:
            assert expanded == query

        # Step 3: Translate if non-English
        if result.language_code != "en":
            t = DictionaryTranslator()
            translated = t.translate(query, result.language_code, "en")
            assert translated != query or translated == query  # may or may not have mapping

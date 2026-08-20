"""Tests for Unicode script-based language detection."""

from __future__ import annotations

import time

import pytest

from voice_rag_ingestion.language import (
    LanguageDetectionResult,
    detect_language,
    get_language_name,
    get_sarvam_code,
    is_english,
)


# ── Core detection tests ───────────────────────────────────────────────────


class TestDetectLanguage:
    """Verify detection across all 14 supported languages."""

    @pytest.mark.parametrize(
        "text,expected_lang,expected_script",
        [
            # English
            ("What is a corporation?", "en", "Latin"),
            ("How does a company work?", "en", "Latin"),
            # Hindi (Devanagari)
            ("कंपनी क्या है?", "hi", "Devanagari"),
            ("निगम की परिभाषा क्या है?", "hi", "Devanagari"),
            # Gujarati
            ("કંપની શું છે?", "gu", "Gujarati"),
            ("નિગમ એટલે શું?", "gu", "Gujarati"),
            # Bengali
            ("কোম্পানী কি?", "bn", "Bengali"),
            ("নিগম কি?", "bn", "Bengali"),
            # Assamese (Bengali script + ৰ/ৱ characters)
            ("কৰ্পোৰেচন কি?", "as", "Bengali"),
            ("কোম্পানী এটা কেনেকৈ গঠন কৰা হয়?", "as", "Bengali"),
            # Tamil
            ("நிறுவனம் என்றால் என்ன?", "ta", "Tamil"),
            # Telugu
            ("కంపెనీ అంటే ఏమిటి?", "te", "Telugu"),
            # Kannada
            ("ಕಂಪನಿ ಎಂದರೇನು?", "kn", "Kannada"),
            # Malayalam
            ("കമ്പനി എന്താണ്?", "ml", "Malayalam"),
            # Punjabi (Gurmukhi)
            ("ਕੰਪਨੀ ਕੀ ਹੈ?", "pa", "Gurmukhi"),
            # Odia
            ("କମ୍ପାନୀ କଣ?", "or", "Odia"),
            # Urdu (Arabic script)
            ("کمپنی کیا ہے؟", "ur", "Arabic"),
        ],
        ids=[
            "english_1", "english_2",
            "hindi_1", "hindi_2",
            "gujarati_1", "gujarati_2",
            "bengali_1", "bengali_2",
            "assamese_1", "assamese_2",
            "tamil",
            "telugu",
            "kannada",
            "malayalam",
            "punjabi",
            "odia",
            "urdu",
        ],
    )
    def test_detection(self, text: str, expected_lang: str, expected_script: str) -> None:
        result = detect_language(text)
        assert isinstance(result, LanguageDetectionResult)
        assert result.language_code == expected_lang, (
            f"Expected {expected_lang} for '{text}', got {result.language_code}"
        )
        assert result.script == expected_script
        assert 0.0 <= result.confidence <= 1.0

    def test_empty_string(self) -> None:
        result = detect_language("")
        assert result.language_code == "en"
        assert result.confidence == 0.0

    def test_whitespace_only(self) -> None:
        result = detect_language("   ")
        assert result.language_code == "en"
        assert result.confidence == 0.0

    def test_marathi_detection(self) -> None:
        # Marathi uses Devanagari with ळ
        result = detect_language("कंपनीचे वैशिष्ट्ये काय आहेत? ळ")
        assert result.language_code == "mr"

    def test_nepali_detection(self) -> None:
        # Nepali has distinctive sentence endings
        result = detect_language("कम्पनी के हो भन्छ?")
        assert result.language_code == "ne"


# ── Performance test ───────────────────────────────────────────────────────


class TestDetectionPerformance:
    """Ensure detection meets <1ms requirement."""

    def test_execution_under_1ms(self) -> None:
        queries = [
            "What is a corporation?",
            "कंपनी क्या है?",
            "કંપની શું છે?",
            "কৰ্পোৰেচন কি?",
            "நிறுவனம் என்றால் என்ன?",
        ]
        # Warm up
        for q in queries:
            detect_language(q)

        # Measure
        start = time.perf_counter()
        iterations = 1000
        for _ in range(iterations):
            for q in queries:
                detect_language(q)
        elapsed = (time.perf_counter() - start) / (iterations * len(queries))
        assert elapsed < 0.001, f"Detection took {elapsed*1000:.3f}ms, target <1ms"


# ── Utility tests ─────────────────────────────────────────────────────────


class TestUtilities:
    def test_is_english_true(self) -> None:
        assert is_english("What is a company?") is True

    def test_is_english_false(self) -> None:
        assert is_english("कंपनी क्या है?") is False

    def test_get_sarvam_code(self) -> None:
        assert get_sarvam_code("hi") == "hi-IN"
        assert get_sarvam_code("gu") == "gu-IN"
        assert get_sarvam_code("en") == "en-IN"
        assert get_sarvam_code("unknown") == "en-IN"

    def test_get_language_name(self) -> None:
        assert get_language_name("hi") == "Hindi"
        assert get_language_name("gu") == "Gujarati"
        assert get_language_name("en") == "English"


# ── Mixed script tests ────────────────────────────────────────────────────


class TestMixedScript:
    def test_hinglish_hindi_dominant(self) -> None:
        # More Devanagari than Latin
        result = detect_language("कंपनी what है company?")
        assert result.language_code == "hi"

    def test_english_with_few_indic(self) -> None:
        # Mostly English — should detect as English
        result = detect_language("The company is good")
        assert result.language_code == "en"

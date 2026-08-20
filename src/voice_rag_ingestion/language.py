"""Lightweight Unicode script-based language detection for Indic languages.

Uses Unicode block ranges to identify scripts — no ML model needed.
Execution time: <0.1ms per query. Thread-safe (pure functions on immutable data).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class LanguageDetectionResult:
    """Result of language detection."""
    language_code: str
    script: str
    confidence: float


# Unicode block ranges for Indic scripts.
# Each entry: (compiled regex, script name, default language code)
_SCRIPT_PATTERNS: list[tuple[re.Pattern[str], str, str]] = [
    (re.compile(r"[\u0980-\u09FF]"), "Bengali", "bn"),
    (re.compile(r"[\u0A80-\u0AFF]"), "Gujarati", "gu"),
    (re.compile(r"[\u0A00-\u0A7F]"), "Gurmukhi", "pa"),
    (re.compile(r"[\u0B80-\u0BFF]"), "Tamil", "ta"),
    (re.compile(r"[\u0C00-\u0C7F]"), "Telugu", "te"),
    (re.compile(r"[\u0C80-\u0CFF]"), "Kannada", "kn"),
    (re.compile(r"[\u0D00-\u0D7F]"), "Malayalam", "ml"),
    (re.compile(r"[\u0B00-\u0B7F]"), "Odia", "or"),
    (re.compile(r"[\u0600-\u06FF]"), "Arabic", "ur"),
    # Devanagari last — it's a shared script, needs disambiguation
    (re.compile(r"[\u0900-\u097F]"), "Devanagari", "hi"),
]

# Assamese-specific characters that distinguish from Bengali:
# ৰ (U+09F0) and ৱ (U+09F1) are uniquely Assamese in the Bengali block.
_ASSAMESE_CHARS = re.compile(r"[\u09F0\u09F1]")

# Marathi-specific: ळ (U+0933) is common in Marathi but rare in Hindi.
# Also check for Marathi-specific vocabulary patterns.
_MARATHI_CHAR = re.compile(r"[\u0933]")

# Nepali uses Devanagari but has distinctive particles.
# छ/छन् sentence endings are characteristic of Nepali.
_NEPALI_ENDINGS = re.compile(r"(?:छ|छन्|हुन्छ|गर्छ|भन्छ)\s*[?।]?\s*$")


# Sarvam AI language code mapping
SARVAM_LANG_MAP: dict[str, str] = {
    "en": "en-IN",
    "hi": "hi-IN",
    "gu": "gu-IN",
    "bn": "bn-IN",
    "as": "as-IN",
    "ta": "ta-IN",
    "te": "te-IN",
    "mr": "mr-IN",
    "ml": "ml-IN",
    "kn": "kn-IN",
    "pa": "pa-IN",
    "or": "or-IN",
    "ur": "ur-IN",
    "ne": "ne-IN",
}

# IndicTrans2 BCP-47 style language code mapping
INDICTRANS_LANG_MAP: dict[str, str] = {
    "en": "eng_Latn",
    "hi": "hin_Deva",
    "gu": "guj_Gujr",
    "bn": "ben_Beng",
    "as": "asm_Beng",
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "mr": "mar_Deva",
    "ml": "mal_Mlym",
    "kn": "kan_Knda",
    "pa": "pan_Guru",
    "or": "ory_Orya",
    "ur": "urd_Arab",
    "ne": "nep_Deva",
}

SUPPORTED_LANGUAGES: set[str] = set(SARVAM_LANG_MAP.keys())

LANGUAGE_NAMES: dict[str, str] = {
    "en": "English",
    "hi": "Hindi",
    "gu": "Gujarati",
    "bn": "Bengali",
    "as": "Assamese",
    "ta": "Tamil",
    "te": "Telugu",
    "mr": "Marathi",
    "ml": "Malayalam",
    "kn": "Kannada",
    "pa": "Punjabi",
    "or": "Odia",
    "ur": "Urdu",
    "ne": "Nepali",
}


def detect_language(text: str) -> LanguageDetectionResult:
    """Detect the language of the input text using Unicode script analysis.

    Returns the detected language code, script name, and confidence score.
    Execution: <0.1ms. Thread-safe.
    """
    if not text or not text.strip():
        return LanguageDetectionResult("en", "Latin", 0.0)

    cleaned = text.strip()
    total_chars = len(cleaned)

    # Count characters per script
    script_counts: dict[str, int] = {}
    matched_script: str | None = None
    matched_lang: str | None = None

    for pattern, script_name, lang_code in _SCRIPT_PATTERNS:
        count = len(pattern.findall(cleaned))
        if count > 0:
            script_counts[script_name] = count
            if matched_script is None or count > script_counts.get(matched_script, 0):
                matched_script = script_name
                matched_lang = lang_code

    # No Indic script detected — assume English/Latin
    if matched_script is None:
        return LanguageDetectionResult("en", "Latin", 1.0)

    dominant_count = script_counts[matched_script]
    confidence = min(dominant_count / max(total_chars, 1), 1.0)

    # Disambiguate shared scripts
    if matched_script == "Bengali":
        matched_lang = _disambiguate_bengali_assamese(cleaned)
    elif matched_script == "Devanagari":
        matched_lang = _disambiguate_devanagari(cleaned)

    return LanguageDetectionResult(matched_lang, matched_script, round(confidence, 3))


def _disambiguate_bengali_assamese(text: str) -> str:
    """Distinguish Assamese from Bengali using script-specific characters."""
    if _ASSAMESE_CHARS.search(text):
        return "as"
    return "bn"


# Marathi-specific patterns: ळ (U+0933) and common Marathi grammatical particles/words
_MARATHI_PATTERN = re.compile(r"(?:[\u0933]|\b(?:म्हणजे|काय|कशी|कसे|आहे|आहेत|झाले|केले|नाही|कशा)\b)")


def _disambiguate_devanagari(text: str) -> str:
    """Distinguish Hindi, Marathi, and Nepali — all use Devanagari script."""
    # Check for Nepali sentence-ending patterns
    if _NEPALI_ENDINGS.search(text):
        return "ne"
    # Check for Marathi distinctive characters and common vocabulary
    if _MARATHI_PATTERN.search(text):
        return "mr"
    # Default to Hindi (most common Devanagari language)
    return "hi"


def is_english(text: str) -> bool:
    """Quick check: is the text predominantly English/Latin?"""
    result = detect_language(text)
    return result.language_code == "en"


def get_sarvam_code(lang_code: str) -> str:
    """Convert internal language code to Sarvam AI API code."""
    return SARVAM_LANG_MAP.get(lang_code, "en-IN")


def get_language_name(lang_code: str) -> str:
    """Get human-readable language name."""
    return LANGUAGE_NAMES.get(lang_code, "Unknown")


__all__ = [
    "LanguageDetectionResult",
    "detect_language",
    "is_english",
    "get_sarvam_code",
    "get_language_name",
    "SUPPORTED_LANGUAGES",
    "LANGUAGE_NAMES",
    "SARVAM_LANG_MAP",
    "INDICTRANS_LANG_MAP",
]

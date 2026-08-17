"""Conservative multilingual text cleaning."""

from __future__ import annotations

import re
import unicodedata
from typing import Any


_WHITESPACE = re.compile(r"\s+", flags=re.UNICODE)


def clean_text(value: Any) -> str | None:
    """Normalize Unicode and whitespace without transliterating or stripping scripts."""

    if value is None:
        return None
    if not isinstance(value, str):
        value = str(value)
    text = unicodedata.normalize("NFC", value)
    text = _WHITESPACE.sub(" ", text).strip()
    return text or None


def stable_text_key(text: str) -> str:
    """Return the deduplication key for already-cleaned text."""

    return unicodedata.normalize("NFC", text).casefold()

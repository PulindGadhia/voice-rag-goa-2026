"""Replaceable Unicode-aware tokenization for lexical retrieval."""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Protocol


class Tokenizer(Protocol):
    def tokenize(self, text: str | None) -> list[str]:
        ...


@dataclass(frozen=True)
class TokenizerConfig:
    lowercase: bool = True
    unicode_normalization: str = "NFC"
    min_token_length: int = 1

    def __post_init__(self) -> None:
        if self.unicode_normalization not in {"NFC", "NFKC"}:
            raise ValueError("unicode_normalization must be NFC or NFKC")
        if self.min_token_length < 1:
            raise ValueError("min_token_length must be >= 1")


class UnicodeWordTokenizer:
    """Keep Unicode word characters while treating punctuation as boundaries."""

    def __init__(self, config: TokenizerConfig | None = None) -> None:
        self.config = config or TokenizerConfig()

    @staticmethod
    def _is_word_character(character: str) -> bool:
        category = unicodedata.category(character)
        return category[0] in {"L", "N", "M"} or character == "_"

    def tokenize(self, text: str | None) -> list[str]:
        if not text:
            return []
        normalized = unicodedata.normalize(self.config.unicode_normalization, str(text))
        if self.config.lowercase:
            normalized = normalized.casefold()
        tokens: list[str] = []
        current: list[str] = []
        for character in normalized:
            if self._is_word_character(character):
                current.append(character)
            elif current:
                token = "".join(current)
                if len(token) >= self.config.min_token_length:
                    tokens.append(token)
                current = []
        if current:
            token = "".join(current)
            if len(token) >= self.config.min_token_length:
                tokens.append(token)
        return tokens

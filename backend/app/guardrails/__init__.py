"""Small, deterministic input and output checks used by the application."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class GuardrailVerdict:
    passed: bool
    rule: str
    reason: str
    severity: str = "block"


# Pre-compiled regex patterns (avoid per-call compilation)
_WORD_RE = re.compile(r"\w+", re.UNICODE)


class InputGuardrails:
    def __init__(self, *, min_length: int = 2, max_length: int = 2000) -> None:
        self.min_length = min_length
        self.max_length = max_length
        self._injection = re.compile(
            r"(ignore\s+(all|any|previous)|reveal\s+system\s+prompt|jailbreak)", re.I
        )

    def check(self, query: str) -> list[GuardrailVerdict]:
        value = query or ""
        verdicts = [
            GuardrailVerdict(bool(value.strip()), "min_query_length", "Query is empty")
            if len(value.strip()) < self.min_length
            else GuardrailVerdict(True, "min_query_length", "Query length is valid", "warn"),
            GuardrailVerdict(False, "max_query_length", "Query exceeds maximum length")
            if len(value) > self.max_length
            else GuardrailVerdict(True, "max_query_length", "Query length is valid", "warn"),
            GuardrailVerdict(False, "prompt_injection", "Prompt injection pattern detected")
            if self._injection.search(value)
            else GuardrailVerdict(True, "prompt_injection", "No injection pattern detected", "warn"),
        ]
        return verdicts


class OutputGuardrails:
    def check_grounding(self, answer: str, passages: list[str]) -> GuardrailVerdict:
        answer_tokens = set(_WORD_RE.findall(answer.lower()))
        source_tokens = set(_WORD_RE.findall(" ".join(passages).lower()))
        overlap = len(answer_tokens & source_tokens) / max(len(answer_tokens), 1)
        return GuardrailVerdict(
            overlap >= 0.10,
            "grounding",
            "Answer overlaps retrieved passages" if overlap >= 0.10 else "Insufficient source overlap",
            "warn",
        )

    def check_grounding_substring(self, answer: str, passage: str) -> GuardrailVerdict:
        """Fast path: check if answer is a substring of the passage (guaranteed grounded)."""
        if answer.strip() and answer.strip() in passage:
            return GuardrailVerdict(True, "grounding", "Answer is direct passage substring", "warn")
        # Fall back to token overlap
        return self.check_grounding(answer, [passage])

    def check_answer_quality(self, answer: str) -> GuardrailVerdict:
        passed = bool(answer and answer.strip())
        return GuardrailVerdict(passed, "answer_quality", "Answer is non-empty" if passed else "Answer is empty", "warn")


__all__ = ["GuardrailVerdict", "InputGuardrails", "OutputGuardrails"]

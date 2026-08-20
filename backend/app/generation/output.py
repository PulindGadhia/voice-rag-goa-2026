"""Validated model output used by the application boundary."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, Field, ValidationError


class GeneratedOutput(BaseModel):
    answer: str = Field(default="")
    grounded: bool = False
    source_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


def parse_generated_output(
    raw: str,
    valid_source_ids: set[str],
    source_aliases: Mapping[str, str] | None = None,
) -> GeneratedOutput:
    """Parse JSON output and constrain citations to retrieved source IDs."""

    candidate = raw.strip()
    # Strip markdown code fences if present
    if candidate.startswith("```"):
        first_newline = candidate.find("\n")
        if first_newline != -1:
            candidate = candidate[first_newline + 1 :]
        else:
            candidate = candidate[3:]
    if candidate.endswith("```"):
        candidate = candidate[:-3]
    candidate = candidate.strip()

    # Extract JSON object { ... }
    start_idx = candidate.find("{")
    end_idx = candidate.rfind("}")
    if start_idx != -1 and end_idx != -1 and end_idx >= start_idx:
        candidate = candidate[start_idx : end_idx + 1]

    try:
        value: Any = json.loads(candidate)
        parsed = GeneratedOutput.model_validate(value)
    except (json.JSONDecodeError, ValidationError, TypeError) as exc:
        raise ValueError("LLM returned malformed structured output") from exc
    aliases = source_aliases or {}
    normalized_source_ids: list[str] = []
    for source_id in parsed.source_ids:
        canonical_id = aliases.get(source_id, source_id)
        if canonical_id in valid_source_ids and canonical_id not in normalized_source_ids:
            normalized_source_ids.append(canonical_id)
    parsed.source_ids = normalized_source_ids
    parsed.grounded = bool(parsed.grounded and parsed.source_ids)
    return parsed

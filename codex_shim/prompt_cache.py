"""Prompt cache request fields for logging and ChatGPT passthrough parity."""

from __future__ import annotations

from typing import Any

# Top-level Responses API cache fields mirrored from OpenAI / Codex upstream.
# Passthrough must forward these unchanged; extend when upstream adds fields.
PROMPT_CACHE_BODY_KEYS: tuple[str, ...] = (
    "prompt_cache_key",
    "prompt_cache_options",
    "prompt_cache_breakpoint",
    "prompt_cache_retention",
)


def prompt_cache_fields_from_body(body: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(body, dict):
        return {}
    return {key: body[key] for key in PROMPT_CACHE_BODY_KEYS if key in body}


def format_prompt_cache_req_suffix(body: dict[str, Any] | None) -> str:
    fields = prompt_cache_fields_from_body(body)
    if not fields:
        return ""
    parts = [f"{key}={fields[key]!r}" for key in PROMPT_CACHE_BODY_KEYS if key in fields]
    return " " + " ".join(parts)

from __future__ import annotations

from typing import Any

from ..config import CompactionSettings, compaction_prompt_cache_key
from ..prompts import (
    build_summarization_user_prompt,
    native_compact_instructions,
    summarization_system_instructions,
)
from ..pipeline import PreparedInput


def build_native_compact_body(
    prepared: PreparedInput,
    *,
    body: dict[str, Any],
    upstream_model: str,
    requested_slug: str,
    settings: CompactionSettings,
    session_key: str = "",
) -> dict[str, Any]:
    client_instructions = body.get("instructions")
    if isinstance(client_instructions, str) and settings.use_client_instructions_for_native:
        instructions = native_compact_instructions(client_instructions)
    else:
        instructions = native_compact_instructions(None)
    compact: dict[str, Any] = {
        "model": upstream_model,
        "instructions": instructions,
        "input": prepared.native_input,
    }
    if body.get("tools") is not None:
        compact["tools"] = body.get("tools")
    if body.get("parallel_tool_calls") is not None:
        compact["parallel_tool_calls"] = body.get("parallel_tool_calls")
    if body.get("reasoning") is not None:
        compact["reasoning"] = body.get("reasoning")
    if body.get("service_tier") is not None:
        compact["service_tier"] = body.get("service_tier")
    if body.get("text") is not None:
        compact["text"] = body.get("text")
    compact["prompt_cache_key"] = compaction_prompt_cache_key(settings, session_key or None)
    compact["model"] = requested_slug
    return compact


def build_summarization_compact_body(
    prepared: PreparedInput,
    *,
    body: dict[str, Any],
    upstream_model: str,
    requested_slug: str,
    settings: CompactionSettings,
    session_key: str = "",
    stream: bool = True,
) -> dict[str, Any]:
    user_prompt = build_summarization_user_prompt(
        previous_summary=prepared.previous_summary,
        extra_context=prepared.extra_context,
        recent_user_turns_excluded=prepared.excluded_user_turns,
    )
    input_items = list(prepared.summarization_input)
    input_items.append(
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": user_prompt}],
        }
    )
    compact: dict[str, Any] = {
        "model": upstream_model,
        "instructions": summarization_system_instructions(),
        "input": input_items,
        "stream": stream,
        "max_output_tokens": settings.summary_max_output_tokens,
        "prompt_cache_key": compaction_prompt_cache_key(settings, session_key or None),
    }
    if body.get("tools") is not None:
        compact["tools"] = body.get("tools")
    if body.get("parallel_tool_calls") is not None:
        compact["parallel_tool_calls"] = body.get("parallel_tool_calls")
    if body.get("reasoning") is not None:
        compact["reasoning"] = body.get("reasoning")
    compact["model"] = requested_slug
    return compact


def build_byok_compact_body(
    prepared: PreparedInput,
    *,
    body: dict[str, Any],
    upstream_model: str,
    for_summarization: bool = False,
    settings: CompactionSettings | None = None,
    session_key: str = "",
) -> dict[str, Any]:
    if for_summarization and settings is not None:
        return build_summarization_compact_body(
            prepared,
            body=body,
            upstream_model=upstream_model,
            requested_slug=body.get("model") or upstream_model,
            settings=settings,
            session_key=session_key,
            stream=False,
        )
    client_instructions = body.get("instructions")
    instructions = native_compact_instructions(
        client_instructions if isinstance(client_instructions, str) else None
    )
    compact: dict[str, Any] = {
        "model": upstream_model,
        "instructions": instructions,
        "input": prepared.native_input,
        "max_output_tokens": (settings.summary_max_output_tokens if settings else 4096),
        "stream": False,
    }
    if body.get("tools") is not None:
        compact["tools"] = body.get("tools")
    if body.get("parallel_tool_calls") is not None:
        compact["parallel_tool_calls"] = body.get("parallel_tool_calls")
    if settings is not None:
        compact["prompt_cache_key"] = compaction_prompt_cache_key(settings, session_key or None)
    return compact

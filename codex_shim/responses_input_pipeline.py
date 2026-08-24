"""Responses API input preparation: cache expansion and orphan tool repair.

**Expansion** replays ``previous_response_id`` deltas from the session cache and
strips the field before upstream HTTP (or WS after a new connect / prev_id error).

**Orphan synthesis** runs when expanding or on stateless surfaces. It is skipped for
native ChatGPT Codex WS deltas (reused connection + ``previous_response_id``) because
the upstream connection already holds the prior tool call.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any, Protocol

from .compaction.input_audit import CompactionSanitizationAudit, compaction_input_item_ref
from .compaction.logging import log_compaction_cache_expansion
from .tool_translate import (
    apply_function_call_ids,
    responses_function_call_ids,
    strip_function_call_output_item_id,
)

UNKNOWN_FUNCTION_TOOL_NAME = "unknown_tool"
UNKNOWN_CUSTOM_TOOL_NAME = "unknown_custom_tool"
STALE_TOOL_OUTPUT_MESSAGE = (
    "This tool output is old and no longer available. "
    "Do not assume a previous result is still valid; continue without it or retry the tool if still needed."
)

_TOOL_CALL_ITEM_TYPES = frozenset({"function_call", "custom_tool_call"})
_TOOL_OUTPUT_ITEM_TYPES = frozenset({"function_call_output", "custom_tool_call_output"})


class ConversationCache(Protocol):
    def get(self, session_key: str, previous_response_id: str) -> list[Any] | None: ...


def responses_input_items(input_value: Any) -> list[Any]:
    if isinstance(input_value, list):
        return copy.deepcopy(input_value)
    if input_value is None:
        return []
    return [{"type": "message", "role": "user", "content": copy.deepcopy(input_value)}]


def _tool_item_call_id(item: dict[str, Any]) -> str | None:
    call_id = item.get("call_id")
    if isinstance(call_id, str) and call_id:
        return call_id
    return None


def _synthetic_tool_call_item(
    call_id: str,
    *,
    output_type: str,
    name: str | None = None,
) -> dict[str, Any]:
    if output_type == "custom_tool_call_output":
        return {
            "type": "custom_tool_call",
            "call_id": call_id,
            "name": name or UNKNOWN_CUSTOM_TOOL_NAME,
            "input": "",
        }
    return {
        "type": "function_call",
        "call_id": call_id,
        "name": name or UNKNOWN_FUNCTION_TOOL_NAME,
        "arguments": "{}",
    }


def synthesize_orphan_tool_calls(
    input_items: list[Any],
    *,
    name_resolver: Callable[[str], str | None] | None = None,
) -> tuple[list[Any], list[str]]:
    """Insert synthetic tool calls before orphan tool outputs in Responses input.

    Bridge results arrive detached from the ``function_call`` the shim emitted on an
    earlier, early-completed stream. ``name_resolver`` recovers the real tool name so
    the model does not see the result attributed to a placeholder.
    """
    if not input_items:
        return [], []

    warnings: list[str] = []
    seen_call_ids: set[str] = set()
    for raw in input_items:
        if not isinstance(raw, dict):
            continue
        if raw.get("type") in _TOOL_CALL_ITEM_TYPES:
            call_id = _tool_item_call_id(raw)
            if call_id:
                seen_call_ids.add(call_id)

    repaired: list[Any] = []
    for index, raw in enumerate(input_items):
        if not isinstance(raw, dict):
            repaired.append(raw)
            continue
        item_type = raw.get("type")
        if item_type in _TOOL_CALL_ITEM_TYPES:
            call_id = _tool_item_call_id(raw)
            if call_id:
                seen_call_ids.add(call_id)
            repaired.append(raw)
            continue
        if item_type in _TOOL_OUTPUT_ITEM_TYPES:
            call_id = _tool_item_call_id(raw)
            if call_id and call_id not in seen_call_ids:
                resolved = name_resolver(call_id) if name_resolver else None
                synth = _synthetic_tool_call_item(
                    call_id, output_type=item_type, name=resolved
                )
                repaired.append(copy.deepcopy(synth))
                seen_call_ids.add(call_id)
                ref = compaction_input_item_ref(index, raw)
                warnings.append(
                    f"synthesized {synth.get('type')} before orphan {ref.label()} "
                    f"(name={synth.get('name')!r})"
                )
            repaired.append(raw)
            continue
        repaired.append(raw)

    return repaired, warnings


def _synthetic_tool_output_item(call_id: str, *, call_type: str) -> dict[str, Any]:
    if call_type == "custom_tool_call":
        return {
            "type": "custom_tool_call_output",
            "call_id": call_id,
            "output": STALE_TOOL_OUTPUT_MESSAGE,
        }
    return {
        "type": "function_call_output",
        "call_id": call_id,
        "output": STALE_TOOL_OUTPUT_MESSAGE,
    }


def align_responses_tool_call_ids(input_items: list[Any]) -> list[Any]:
    """Canonicalize tool ``call_id``s so calls and outputs still pair after expand.

    Cache snapshots from other providers (bare UUIDs) are prepended *after*
    ChatGPT passthrough sanitization. Rewrite them here so pairing/stubs see the
    same ``call_*`` ids the Responses API uses. Do not treat a provider switch as
    a compaction retry.
    """
    aligned: list[Any] = []
    for raw in input_items:
        if not isinstance(raw, dict):
            aligned.append(raw)
            continue
        item_type = raw.get("type")
        if item_type == "function_call":
            aligned.append(apply_function_call_ids(raw))
            continue
        if item_type in _TOOL_OUTPUT_ITEM_TYPES:
            aligned.append(strip_function_call_output_item_id(raw))
            continue
        if item_type == "custom_tool_call":
            item = dict(raw)
            raw_call = item.get("call_id")
            if isinstance(raw_call, str) and raw_call.strip():
                _, call_id = responses_function_call_ids(raw_call)
                item["call_id"] = call_id
            aligned.append(item)
            continue
        aligned.append(raw)
    return aligned


def synthesize_missing_tool_outputs(
    input_items: list[Any],
) -> tuple[list[Any], list[str]]:
    """Insert a stale-output stub after function calls that have no result.

    ChatGPT Responses rejects ``input`` that contains a ``function_call`` without a
    matching ``function_call_output``. Stub instead of dropping the call so the
    model can continue; keep the real output when ids can be aligned.
    """
    if not input_items:
        return [], []

    answered: set[str] = set()
    for raw in input_items:
        if not isinstance(raw, dict) or raw.get("type") not in _TOOL_OUTPUT_ITEM_TYPES:
            continue
        call_id = _tool_item_call_id(raw)
        if call_id:
            answered.add(call_id)

    warnings: list[str] = []
    repaired: list[Any] = []
    for raw in input_items:
        repaired.append(raw)
        if not isinstance(raw, dict) or raw.get("type") not in _TOOL_CALL_ITEM_TYPES:
            continue
        call_id = _tool_item_call_id(raw)
        if not call_id or call_id in answered:
            continue
        synth = _synthetic_tool_output_item(call_id, call_type=str(raw.get("type") or ""))
        repaired.append(synth)
        answered.add(call_id)
        warnings.append(
            f"synthesized {synth.get('type')} for unanswered {raw.get('type')} "
            f"call_id={call_id!r} (tool output old/unavailable)"
        )
    return repaired, warnings


def expand_cached_responses_input(
    *,
    cache: ConversationCache | None,
    session_key: str,
    previous_response_id: Any,
    delta_input: list[Any],
    expand_enabled: bool,
    context: str,
) -> list[Any]:
    if not expand_enabled or cache is None:
        return delta_input
    if not isinstance(previous_response_id, str) or not previous_response_id:
        return delta_input

    from .compaction.input_audit import summarize_compaction_input_items

    _, delta_summary = summarize_compaction_input_items(delta_input, tail=6)
    cached = cache.get(session_key, previous_response_id)
    if cached is None:
        latest_fn = getattr(cache, "latest", None)
        if callable(latest_fn):
            cached = latest_fn(session_key)
        if cached is None:
            if context == "compaction":
                log_compaction_cache_expansion(
                    context=context,
                    session_key=session_key,
                    previous_response_id=previous_response_id,
                    cached_items=None,
                    delta_items=len(delta_input),
                    total_items=None,
                    delta_summary=delta_summary,
                )
            else:
                print(
                    f"[chatgpt-cache] {context} MISS session={session_key} "
                    f"previous_response_id={previous_response_id} delta_items={len(delta_input)}",
                    flush=True,
                )
            return delta_input
        print(
            f"[chatgpt-cache] {context} MISS-FALLBACK session={session_key} "
            f"previous_response_id={previous_response_id} latest_items={len(cached)} "
            f"delta_items={len(delta_input)}",
            flush=True,
        )

    expanded = [*copy.deepcopy(cached), *copy.deepcopy(delta_input)]
    if context == "compaction":
        log_compaction_cache_expansion(
            context=context,
            session_key=session_key,
            previous_response_id=previous_response_id,
            cached_items=len(cached),
            delta_items=len(delta_input),
            total_items=len(expanded),
            delta_summary=delta_summary,
        )
    else:
        print(
            f"[chatgpt-cache] {context} HIT session={session_key} "
            f"previous_response_id={previous_response_id} cached={len(cached)} "
            f"delta={len(delta_input)} total={len(expanded)}",
            flush=True,
        )
    return expanded


def prepare_responses_input_items(
    *,
    cache: ConversationCache | None,
    session_key: str,
    previous_response_id: Any,
    input_items: list[Any],
    expand_enabled: bool,
    context: str,
    orphan_synthesis: bool = True,
    name_resolver: Callable[[str], str | None] | None = None,
) -> tuple[list[Any], list[str]]:
    expanded = expand_cached_responses_input(
        cache=cache,
        session_key=session_key,
        previous_response_id=previous_response_id,
        delta_input=input_items,
        expand_enabled=expand_enabled,
        context=context,
    )
    aligned = align_responses_tool_call_ids(expanded)
    if not orphan_synthesis:
        return aligned, []
    repaired, warnings = synthesize_orphan_tool_calls(aligned, name_resolver=name_resolver)
    stubbed, extra = synthesize_missing_tool_outputs(repaired)
    return stubbed, [*warnings, *extra]


def _log_pipeline_warnings(warnings: list[str]) -> None:
    for warning in warnings:
        if warning:
            print(f"[warn] responses-input: {warning}", flush=True)


def apply_responses_input_pipeline_to_body(
    body: dict[str, Any],
    *,
    cache: ConversationCache | None,
    session_key: str,
    expand_enabled: bool,
    context: str,
    strip_previous_response_id: bool = False,
    orphan_synthesis: bool = True,
    name_resolver: Callable[[str], str | None] | None = None,
) -> dict[str, Any]:
    prepared = dict(body)
    raw_input = prepared.get("input")
    if isinstance(raw_input, list):
        repaired, warnings = prepare_responses_input_items(
            cache=cache,
            session_key=session_key,
            previous_response_id=prepared.get("previous_response_id"),
            input_items=responses_input_items(raw_input),
            expand_enabled=expand_enabled,
            context=context,
            orphan_synthesis=orphan_synthesis,
            name_resolver=name_resolver,
        )
        _log_pipeline_warnings(warnings)
        prepared["input"] = repaired
    if strip_previous_response_id:
        prepared.pop("previous_response_id", None)
    return prepared


def sanitize_compaction_input_with_pipeline(
    input_items: list[Any],
) -> tuple[list[Any], list[str], CompactionSanitizationAudit]:
    audit = CompactionSanitizationAudit(incoming_items=len(input_items))
    if not input_items:
        audit.outgoing_items = 0
        return [], [], audit

    repaired, warnings = synthesize_orphan_tool_calls(input_items)
    repaired, extra = synthesize_missing_tool_outputs(repaired)
    warnings = [*warnings, *extra]
    audit.outgoing_items = len(repaired)
    audit.synthesized = warnings
    return repaired, audit.warning_lines(), audit

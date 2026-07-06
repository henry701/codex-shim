from __future__ import annotations

import copy
from typing import Any, Protocol

from .compaction.input_audit import CompactionSanitizationAudit, compaction_input_item_ref
from .compaction.logging import log_compaction_cache_expansion

UNKNOWN_FUNCTION_TOOL_NAME = "unknown_tool"
UNKNOWN_CUSTOM_TOOL_NAME = "unknown_custom_tool"

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


def _synthetic_tool_call_item(call_id: str, *, output_type: str) -> dict[str, Any]:
    if output_type == "custom_tool_call_output":
        return {
            "type": "custom_tool_call",
            "call_id": call_id,
            "name": UNKNOWN_CUSTOM_TOOL_NAME,
            "input": "",
        }
    return {
        "type": "function_call",
        "call_id": call_id,
        "name": UNKNOWN_FUNCTION_TOOL_NAME,
        "arguments": "{}",
    }


def synthesize_orphan_tool_calls(input_items: list[Any]) -> tuple[list[Any], list[str]]:
    """Insert synthetic tool calls before orphan tool outputs in Responses input."""
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
                synth = _synthetic_tool_call_item(call_id, output_type=item_type)
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
) -> tuple[list[Any], list[str]]:
    expanded = expand_cached_responses_input(
        cache=cache,
        session_key=session_key,
        previous_response_id=previous_response_id,
        delta_input=input_items,
        expand_enabled=expand_enabled,
        context=context,
    )
    return synthesize_orphan_tool_calls(expanded)


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
    audit.outgoing_items = len(repaired)
    audit.synthesized = warnings
    return repaired, audit.warning_lines(), audit

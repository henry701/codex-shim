from __future__ import annotations

import base64
import json
import time
from typing import Any

from .input_audit import CompactionSanitizationAudit, compaction_input_item_ref

SHIM_COMPACTION_PREFIX = "codex-shim-compaction-v1:"

_TOOL_CALL_ITEM_TYPES = frozenset({"function_call", "custom_tool_call"})
_TOOL_OUTPUT_ITEM_TYPES = frozenset({"function_call_output", "custom_tool_call_output"})


class CompactionTriggerError(ValueError):
    def __init__(self, message: str, *, param: str = "input") -> None:
        super().__init__(message)
        self.param = param


def strip_terminal_compaction_trigger(input_items: Any) -> list[Any] | None:
    """Return input without a terminal compaction_trigger, or None if absent."""
    if not isinstance(input_items, list):
        return None
    stripped: list[Any] = []
    trigger_seen = False
    last_index = len(input_items) - 1
    for index, item in enumerate(input_items):
        if isinstance(item, dict) and item.get("type") == "compaction_trigger":
            if trigger_seen or index != last_index:
                raise CompactionTriggerError(
                    "compaction_trigger must appear exactly once as the final top-level input item"
                )
            trigger_seen = True
            continue
        stripped.append(item)
    if not trigger_seen:
        return None
    return stripped


def _tool_item_call_id(item: dict[str, Any]) -> str | None:
    call_id = item.get("call_id")
    if isinstance(call_id, str) and call_id:
        return call_id
    return None


def drop_orphaned_tool_outputs(input_items: list[Any]) -> tuple[list[Any], list[str]]:
    """Drop tool outputs with no preceding tool call for the same call_id."""
    cleaned, audit = _drop_orphaned_tool_outputs_with_audit(input_items)
    return cleaned, audit.warning_lines()


def _drop_orphaned_tool_outputs_with_audit(
    input_items: list[Any],
) -> tuple[list[Any], CompactionSanitizationAudit]:
    audit = CompactionSanitizationAudit(incoming_items=len(input_items))
    if not input_items:
        audit.outgoing_items = 0
        return [], audit
    cleaned: list[Any] = []
    seen_call_ids: set[str] = set()
    for index, raw in enumerate(input_items):
        if not isinstance(raw, dict):
            cleaned.append(raw)
            continue
        item_type = raw.get("type")
        if item_type in _TOOL_CALL_ITEM_TYPES:
            call_id = _tool_item_call_id(raw)
            if call_id:
                seen_call_ids.add(call_id)
            cleaned.append(raw)
            continue
        if item_type in _TOOL_OUTPUT_ITEM_TYPES:
            call_id = _tool_item_call_id(raw)
            if call_id and call_id in seen_call_ids:
                cleaned.append(raw)
                continue
            audit.dropped.append(
                (
                    compaction_input_item_ref(index, raw),
                    _DROP_ORPHAN_IN_BATCH if call_id else "missing call_id",
                )
            )
            continue
        cleaned.append(raw)
    audit.outgoing_items = len(cleaned)
    return cleaned, audit


def _last_tool_call_index(input_items: list[Any]) -> int | None:
    last: int | None = None
    for index, raw in enumerate(input_items):
        if isinstance(raw, dict) and raw.get("type") in _TOOL_CALL_ITEM_TYPES:
            last = index
    return last


_DROP_ORPHAN_IN_BATCH = "no in-batch function_call/custom_tool_call for call_id"
_PRESERVE_DETACHED_TAIL = "detached compaction tail after last in-batch tool call"
_PRESERVE_ORPHAN_ONLY_BATCH = "batch contained only tool outputs; preserving for compaction"


def sanitize_compaction_input_items(
    input_items: list[Any],
) -> tuple[list[Any], list[str], CompactionSanitizationAudit]:
    """Drop in-batch orphan tool outputs, but keep detached compaction tail items.

    Codex compaction v2 often sends only truncated tail ``function_call_output``
    items without their matching ``function_call`` in the same request. ChatGPT
    passthrough should expand those deltas from the conversation cache first; any
    tail outputs that still lack an in-batch call are preserved when they appear
    after the last tool call in the batch.
    """
    audit = CompactionSanitizationAudit(incoming_items=len(input_items))
    if not input_items:
        audit.outgoing_items = 0
        return [], [], audit

    cleaned, drop_audit = _drop_orphaned_tool_outputs_with_audit(input_items)
    audit.dropped.extend(drop_audit.dropped)

    last_call_index = _last_tool_call_index(input_items)
    recovered: list[dict[str, Any]] = []
    kept_output_call_ids = {
        _tool_item_call_id(item)
        for item in cleaned
        if isinstance(item, dict)
        and item.get("type") in _TOOL_OUTPUT_ITEM_TYPES
        and _tool_item_call_id(item)
    }
    for index, raw in enumerate(input_items):
        if not isinstance(raw, dict):
            continue
        item_type = raw.get("type")
        if item_type not in _TOOL_OUTPUT_ITEM_TYPES:
            continue
        call_id = _tool_item_call_id(raw)
        if call_id and call_id in kept_output_call_ids:
            continue
        if last_call_index is None or index <= last_call_index:
            continue
        recovered.append(raw)
        audit.preserved.append(
            (
                compaction_input_item_ref(index, raw),
                _PRESERVE_DETACHED_TAIL,
            )
        )

    if not recovered:
        if cleaned:
            audit.outgoing_items = len(cleaned)
            return cleaned, audit.warning_lines(), audit
        tool_outputs_only = bool(input_items) and all(
            isinstance(item, dict) and item.get("type") in _TOOL_OUTPUT_ITEM_TYPES
            for item in input_items
        )
        if tool_outputs_only:
            for index, raw in enumerate(input_items):
                if isinstance(raw, dict):
                    recovered.append(raw)
                    audit.preserved.append(
                        (
                            compaction_input_item_ref(index, raw),
                            _PRESERVE_ORPHAN_ONLY_BATCH,
                        )
                    )
        if not recovered:
            audit.outgoing_items = len(cleaned)
            return cleaned, audit.warning_lines(), audit

    result = [*cleaned, *recovered]
    audit.outgoing_items = len(result)
    return result, audit.warning_lines(), audit


def is_orphan_tool_call_upstream_error(message: str) -> bool:
    lower = (message or "").lower()
    return "no tool call found" in lower and "function call output" in lower


def encode_shim_compaction_summary(summary: str) -> str:
    blob = base64.urlsafe_b64encode(
        json.dumps({"summary": summary}, separators=(",", ":")).encode("utf-8")
    ).decode("ascii")
    return f"{SHIM_COMPACTION_PREFIX}{blob}"


def decode_shim_compaction_summary(encrypted_content: Any) -> str | None:
    if not isinstance(encrypted_content, str) or not encrypted_content.startswith(SHIM_COMPACTION_PREFIX):
        return None
    blob = encrypted_content[len(SHIM_COMPACTION_PREFIX) :]
    try:
        data = json.loads(base64.urlsafe_b64decode(blob.encode("ascii")).decode("utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    summary = data.get("summary")
    if summary is None:
        return None
    text = str(summary).strip()
    return text or None


def compaction_summary_from_output(output: Any) -> str:
    parts: list[str] = []
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "message":
                content = item.get("content") or []
                if isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("text"):
                            parts.append(str(part["text"]))
            elif item.get("type") == "output_text" and item.get("text"):
                parts.append(str(item["text"]))
            elif item.get("type") in {"compaction", "compaction_summary"}:
                decoded = decode_shim_compaction_summary(item.get("encrypted_content"))
                if decoded:
                    parts.append(decoded)
    return "\n".join(part for part in parts if part).strip()


def compaction_output_item(summary: str, *, item_id: str | None = None) -> dict[str, Any]:
    now = int(time.time() * 1000)
    text = summary.strip() or "No prior conversation state was available to compact."
    return {
        "id": item_id or f"cmp_{now}",
        "type": "compaction",
        "status": "completed",
        "encrypted_content": encode_shim_compaction_summary(text),
    }


def apply_compaction_fallback_notice(
    summary: str,
    native_error: str,
    *,
    provider: str | None = None,
    warnings: list[str] | None = None,
) -> str:
    error = (native_error or "unknown error").strip()
    body = summary.strip() or "No prior conversation state was available to compact."
    warn_lines = [line.strip() for line in (warnings or []) if line and line.strip()]
    warn_block = ""
    if warn_lines:
        warn_block = "Compaction warnings:\n" + "\n".join(f"- {line}" for line in warn_lines) + "\n\n"
    provider_prefix = f"{provider}: " if provider else ""
    return (
        f"Remote native compaction failed ({provider_prefix}{error}). "
        "This handoff was generated by fallback summarization.\n\n"
        f"{warn_block}"
        f"{body}"
    )


def compaction_item_from_response_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    output = payload.get("output")
    if isinstance(output, list):
        for raw_item in output:
            if not isinstance(raw_item, dict):
                continue
            item_type = raw_item.get("type")
            encrypted = raw_item.get("encrypted_content")
            if item_type in {"compaction", "compaction_summary"} and isinstance(encrypted, str):
                item = {"type": "compaction", "encrypted_content": encrypted}
                if raw_item.get("id"):
                    item["id"] = raw_item["id"]
                if raw_item.get("status"):
                    item["status"] = raw_item["status"]
                return item
    summary_obj = payload.get("compaction_summary")
    if isinstance(summary_obj, dict):
        encrypted = summary_obj.get("encrypted_content")
        if isinstance(encrypted, str):
            return {"type": "compaction", "encrypted_content": encrypted}
    return None


def compact_response_payload(model: str, summary: str, usage: Any = None) -> dict[str, Any]:
    now = int(time.time())
    response_id = f"resp_compact_{now}"
    item = compaction_output_item(
        summary or "No prior conversation state was available to compact.",
        item_id=f"cmp_{now}",
    )
    payload: dict[str, Any] = {
        "id": response_id,
        "object": "response",
        "created_at": now,
        "status": "completed",
        "model": model,
        "output": [item],
    }
    if usage is not None:
        payload["usage"] = usage
    return payload

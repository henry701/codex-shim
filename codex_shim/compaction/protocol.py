from __future__ import annotations

import base64
import json
import time
from typing import Any

from ..responses_input_pipeline import sanitize_compaction_input_with_pipeline
from .input_audit import CompactionSanitizationAudit

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
    """Repair tool outputs with no preceding tool call by synthesizing placeholder calls."""
    cleaned, warnings, _audit = sanitize_compaction_input_with_pipeline(input_items)
    return cleaned, warnings


def sanitize_compaction_input_items(
    input_items: list[Any],
) -> tuple[list[Any], list[str], CompactionSanitizationAudit]:
    """Expand-safe compaction sanitization: synthesize missing tool calls for orphan outputs."""
    return sanitize_compaction_input_with_pipeline(input_items)


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


def _reasoning_summary_texts(item: dict[str, Any]) -> list[str]:
    summary = item.get("summary")
    texts: list[str] = []
    if isinstance(summary, list):
        for part in summary:
            if isinstance(part, dict) and part.get("text"):
                texts.append(str(part["text"]))
    elif isinstance(summary, str) and summary.strip():
        texts.append(summary)
    return texts


def compaction_summary_from_output(output: Any) -> str:
    """Visible compact text. Reasoning is used only when message/compaction text is empty.

    NVIDIA NIM Muse Glimmer returns a reasoning-only chat completion (empty
    ``content``). DeepSeek and similar still send thinking in ``reasoning_content``
    alongside a real summary in ``content`` — that thinking must not be stored.
    """
    visible: list[str] = []
    reasoning: list[str] = []
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "message":
                content = item.get("content") or []
                if isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("text"):
                            visible.append(str(part["text"]))
            elif item.get("type") == "reasoning":
                reasoning.extend(_reasoning_summary_texts(item))
            elif item.get("type") == "output_text" and item.get("text"):
                visible.append(str(item["text"]))
            elif item.get("type") in {"compaction", "compaction_summary"}:
                decoded = decode_shim_compaction_summary(item.get("encrypted_content"))
                if decoded:
                    visible.append(decoded)
    primary = "\n".join(part for part in visible if part).strip()
    if primary:
        return primary
    return "\n".join(part for part in reasoning if part).strip()


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

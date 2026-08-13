from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

UPSTREAM_IO_DIR = Path.home() / ".codex-shim" / "upstream-io"
RESPONSE_TEXT_LIMIT = 12_000
REQUEST_BODY_FILE_LIMIT = 1_500_000
_INPUT_TYPE_TAIL = 20

# Catalog / picker / discover paths — keep error artifacts quiet (SSD).
_QUIET_SURFACE_MARKERS = (
    "model",
    "models",
    "discover",
    "catalog",
    "picker",
    "health",
    "sync-desktop",
    "sync_desktop",
)

# Responses / compaction / turn passthrough — verbose errors by default.
_VERBOSE_ERROR_SURFACE_MARKERS = (
    "response",
    "responses",
    "compact",
    "compaction",
    "passthrough",
    "chatgpt",
    "byok",
    "anthropic",
    "cursor",
    "summarization",
)


def shim_io_log_enabled() -> bool:
    for key in ("CODEX_SHIM_REQUEST_LOG", "CODEX_SHIM_PASSTHROUGH_TRACE", "CODEX_SHIM_STREAM_LOG"):
        if os.environ.get(key, "").lower() in {"1", "true", "yes", "on"}:
            return True
    return False


def is_quiet_surface(surface: str) -> bool:
    lower = (surface or "").lower()
    return any(marker in lower for marker in _QUIET_SURFACE_MARKERS)


def is_verbose_error_surface(surface: str) -> bool:
    """True for response/compaction turn paths; false for models/picker/discover."""
    if is_quiet_surface(surface):
        return False
    lower = (surface or "").lower()
    return any(marker in lower for marker in _VERBOSE_ERROR_SURFACE_MARKERS)


def _summarize_input_item(item: Any) -> str:
    if not isinstance(item, dict):
        return "?"
    item_type = str(item.get("type") or item.get("role") or "?")
    extra = ""
    if item_type == "function_call":
        extra = f" name={item.get('name', '?')!r}"
    elif item_type == "function_call_output":
        extra = f" call_id={str(item.get('call_id', ''))[:24]!r}"
    elif item_type == "message":
        role = item.get("role")
        content = item.get("content")
        chars = 0
        if isinstance(content, str):
            chars = len(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    text = part.get("text") or part.get("input_text") or ""
                    chars += len(str(text))
        extra = f" role={role!r} chars={chars}"
    elif item_type == "web_search_call":
        action = item.get("action") or {}
        extra = f" query={str(action.get('query', ''))[:40]!r}"
    elif item_type == "mcp_tool_call":
        extra = f" server={item.get('server', '?')!r} tool={item.get('tool', '?')!r}"
    return f"{item_type}{extra}"


def summarize_upstream_body(body: dict[str, Any], *, full_input_types: bool | None = None) -> dict[str, Any]:
    tools = body.get("tools") or []
    input_items = body.get("input")
    messages = body.get("messages")
    input_types: list[str] = []
    input_count = 0
    if isinstance(input_items, list):
        input_count = len(input_items)
        if full_input_types is None:
            full_input_types = shim_io_log_enabled()
        if full_input_types or input_count <= _INPUT_TYPE_TAIL:
            input_types = [_summarize_input_item(item) for item in input_items]
        else:
            head = input_count - _INPUT_TYPE_TAIL
            input_types = [f"…({head} earlier items)"] + [
                _summarize_input_item(item) for item in input_items[-_INPUT_TYPE_TAIL:]
            ]
    message_count = len(messages) if isinstance(messages, list) else 0
    return {
        "model": body.get("model"),
        "stream": body.get("stream"),
        "tools": len(tools) if isinstance(tools, list) else 0,
        "input_items": input_count,
        "input_item_types": input_types,
        "messages": message_count,
        "instructions_chars": len(str(body.get("instructions") or "")),
        "max_output_tokens": body.get("max_output_tokens"),
        "service_tier": body.get("service_tier"),
        "previous_response_id": body.get("previous_response_id"),
    }


def _body_json_size(body: Any) -> int:
    try:
        return len(json.dumps(body, default=str))
    except Exception:
        return 0


def _body_for_error_file(body: dict[str, Any] | None, *, verbose: bool) -> Any:
    if body is None:
        return None
    if not verbose:
        return summarize_upstream_body(body, full_input_types=False)
    if _body_json_size(body) <= REQUEST_BODY_FILE_LIMIT:
        return body
    summary = summarize_upstream_body(body, full_input_types=True)
    summary["_truncated"] = True
    summary["_full_size"] = _body_json_size(body)
    return summary


def record_upstream_error(
    surface: str,
    url: str,
    status: int,
    response_text: str,
    *,
    request_body: dict[str, Any] | None = None,
) -> None:
    """Persist upstream failures.

    Response/compaction turn surfaces: verbose ``last-error.json`` (request +
    response) by default. Models/picker/discover stay summary-only.

    Timestamped ``error-*.json`` files are still debug-only — a 429 retry storm
    on ChatGPT passthrough would otherwise rewrite gigabytes.
    """
    try:
        UPSTREAM_IO_DIR.mkdir(parents=True, exist_ok=True)
        debug = shim_io_log_enabled()
        verbose = debug or is_verbose_error_surface(surface)
        artifact = {
            "ts": time.time(),
            "surface": surface,
            "url": url,
            "status": status,
            "response_text": (response_text or "")[: (RESPONSE_TEXT_LIMIT if verbose else 800)],
            "request_summary": summarize_upstream_body(request_body) if request_body else None,
            "request_body": _body_for_error_file(request_body, verbose=verbose),
        }
        text = json.dumps(artifact, indent=2, default=str)
        (UPSTREAM_IO_DIR / "last-error.json").write_text(text)
        if debug:
            (UPSTREAM_IO_DIR / f"error-{int(time.time())}.json").write_text(text)
    except OSError:
        pass


def _write_request_artifact(surface: str, url: str, body: dict[str, Any]) -> None:
    if not shim_io_log_enabled():
        return
    try:
        UPSTREAM_IO_DIR.mkdir(parents=True, exist_ok=True)
        artifact = {
            "ts": time.time(),
            "surface": surface,
            "url": url,
            "request_summary": summarize_upstream_body(body, full_input_types=True),
            "request_body": _body_for_error_file(body, verbose=True),
        }
        (UPSTREAM_IO_DIR / "last-request.json").write_text(
            json.dumps(artifact, indent=2, default=str)
        )
    except OSError:
        pass


def log_upstream_request(surface: str, url: str, body: dict[str, Any]) -> None:
    summary = summarize_upstream_body(body)
    print(
        f"[upstream-req] {surface} url={url} "
        f"{json.dumps(summary, separators=(',', ':'), default=str)}",
        flush=True,
    )
    _write_request_artifact(surface, url, body)


def log_upstream_response(
    surface: str,
    url: str,
    status: int,
    response_text: str = "",
    *,
    request_body: dict[str, Any] | None = None,
    stream: bool = False,
) -> None:
    preview = (response_text or "")[:500]
    stream_note = " stream=true" if stream else ""
    line = f"[upstream-resp] {surface} url={url} status={status}{stream_note}"
    if status >= 400 and preview:
        line += f" body={preview!r}"
    print(line, flush=True)
    verbose_error = status >= 400 and (shim_io_log_enabled() or is_verbose_error_surface(surface))
    if verbose_error or shim_io_log_enabled():
        payload = {
            "surface": surface,
            "url": url,
            "status": status,
            "stream": stream,
            "response_text": (response_text or "")[:RESPONSE_TEXT_LIMIT],
        }
        if request_body is not None:
            payload["request_summary"] = summarize_upstream_body(
                request_body,
                full_input_types=shim_io_log_enabled() or is_verbose_error_surface(surface),
            )
        print(f"[io-resp] {json.dumps(payload, separators=(',', ':'), default=str)}", flush=True)
    if status >= 400:
        record_upstream_error(surface, url, status, response_text, request_body=request_body)

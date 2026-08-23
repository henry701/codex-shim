from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any

from .settings import ShimModel

DEFAULT_UPSTREAM_COMPAT_PATH = Path.home() / ".codex-shim" / "upstream-compat.json"
CONSOLE_CONTINUE_USER = "Continue."
_CONSOLE_DISCOVER_KINDS = frozenset({"zen", "zen_public"})
_PARALLEL_TOOL_CALLS_UNSUPPORTED = re.compile(
    r"parallel_tool_calls.*(?:not supported|unsupported)|"
    r"(?:not supported|unsupported).*parallel_tool_calls",
    re.IGNORECASE,
)
_CONSOLE_INVALID_PARAMETER = re.compile(
    r"\[(?:1210|1214)\]|"
    r"invalid api parameter|"
    r"messages parameter is illegal",
    re.IGNORECASE,
)
_MESSAGE_KEEP_KEYS = ("role", "content", "name", "tool_calls", "tool_call_id")
_TOOL_CALL_KEEP_KEYS = ("id", "type", "function")
_FUNCTION_KEEP_KEYS = ("name", "arguments")


def is_parallel_tool_calls_unsupported_error(status: int, message: str) -> bool:
    if status not in {400, 422, 502}:
        return False
    text = (message or "").strip()
    if not text:
        return False
    if _PARALLEL_TOOL_CALLS_UNSUPPORTED.search(text):
        return True
    lower = text.lower()
    return "parallel_tool_calls" in lower and "unprocessable" in lower


def is_console_invalid_parameter_error(status: int, message: str) -> bool:
    if status not in {400, 422}:
        return False
    text = (message or "").strip()
    if not text:
        return False
    return _CONSOLE_INVALID_PARAMETER.search(text) is not None


def apply_openai_chat_compat(body: dict[str, Any], *, omit_parallel_tool_calls: bool) -> dict[str, Any]:
    if not omit_parallel_tool_calls or "parallel_tool_calls" not in body:
        return body
    prepared = dict(body)
    prepared.pop("parallel_tool_calls", None)
    return prepared


def apply_console_chat_compat(body: dict[str, Any], *, enabled: bool) -> dict[str, Any]:
    if not enabled:
        return body
    prepared = dict(body)
    prepared.pop("parallel_tool_calls", None)
    prepared.pop("reasoning_effort", None)
    names_by_id = _tool_call_names(prepared.get("messages") or [])
    cleaned: list[dict[str, Any]] = []
    for message in prepared.get("messages") or []:
        if not isinstance(message, dict):
            continue
        current = _console_clean_message(message, names_by_id=names_by_id)
        if current.get("role") == "user" and not _content_nonempty(current.get("content")):
            continue
        cleaned.append(current)
    if not any(message.get("role") == "user" for message in cleaned):
        cleaned.append({"role": "user", "content": CONSOLE_CONTINUE_USER})
    prepared["messages"] = cleaned
    return prepared


def should_omit_parallel_tool_calls(route: ShimModel, *, compat_path: Path | None = None) -> bool:
    raw = route.raw or {}
    if bool(raw.get("omit_parallel_tool_calls")):
        return True
    if raw.get("supports_parallel_tool_calls") is False:
        return True
    store = _load_store(compat_path)
    slug_entry = (store.get("by_slug") or {}).get(route.slug) or {}
    if slug_entry.get("omit_parallel_tool_calls"):
        return True
    model_entry = (store.get("by_upstream_model") or {}).get(route.model) or {}
    return bool(model_entry.get("omit_parallel_tool_calls"))


def should_apply_console_chat_compat(route: ShimModel, *, compat_path: Path | None = None) -> bool:
    raw = route.raw or {}
    if bool(raw.get("console_chat_compat")):
        return True
    if str(raw.get("discover_kind") or "") in _CONSOLE_DISCOVER_KINDS:
        return True
    if (route.slug or "").startswith("oc-free-"):
        return True
    store = _load_store(compat_path)
    slug_entry = (store.get("by_slug") or {}).get(route.slug) or {}
    if slug_entry.get("console_chat_compat"):
        return True
    model_entry = (store.get("by_upstream_model") or {}).get(route.model) or {}
    return bool(model_entry.get("console_chat_compat"))


def remember_parallel_tool_calls_unsupported(
    route: ShimModel,
    *,
    reason: str,
    compat_path: Path | None = None,
) -> bool:
    """Persist learned compat for this route. Returns True when newly recorded."""
    if should_omit_parallel_tool_calls(route, compat_path=compat_path):
        return False
    return _remember_flag(
        route,
        flag="omit_parallel_tool_calls",
        reason=reason,
        compat_path=compat_path,
    )


def remember_console_chat_compat(
    route: ShimModel,
    *,
    reason: str,
    compat_path: Path | None = None,
) -> bool:
    """Persist GLM/Z.AI Console message sanitizer for this route."""
    if should_apply_console_chat_compat(route, compat_path=compat_path):
        return False
    return _remember_flag(
        route,
        flag="console_chat_compat",
        reason=reason,
        compat_path=compat_path,
    )


def learn_parallel_tool_calls_compat_if_needed(
    route: ShimModel,
    status: int,
    message: str,
    *,
    compat_path: Path | None = None,
) -> bool:
    if not is_parallel_tool_calls_unsupported_error(status, message):
        return False
    return remember_parallel_tool_calls_unsupported(route, reason=message, compat_path=compat_path)


def learn_console_chat_compat_if_needed(
    route: ShimModel,
    status: int,
    message: str,
    *,
    compat_path: Path | None = None,
) -> bool:
    if not is_console_invalid_parameter_error(status, message):
        return False
    return remember_console_chat_compat(route, reason=message, compat_path=compat_path)


def prepare_openai_chat_body(route: ShimModel, body: dict[str, Any], *, compat_path: Path | None = None) -> dict[str, Any]:
    console = should_apply_console_chat_compat(route, compat_path=compat_path)
    prepared = apply_openai_chat_compat(
        body,
        omit_parallel_tool_calls=console or should_omit_parallel_tool_calls(route, compat_path=compat_path),
    )
    return apply_console_chat_compat(prepared, enabled=console)


def supports_parallel_tool_calls_in_catalog(route: ShimModel, *, compat_path: Path | None = None) -> bool:
    return not should_omit_parallel_tool_calls(route, compat_path=compat_path)


def _console_clean_message(message: dict[str, Any], *, names_by_id: dict[str, str]) -> dict[str, Any]:
    current = {key: message[key] for key in _MESSAGE_KEEP_KEYS if key in message}
    current["content"] = _flatten_console_content(current.get("content"))
    if current.get("role") == "assistant" and current.get("content") is None:
        current["content"] = ""
    tool_calls = current.get("tool_calls")
    if tool_calls:
        copied_calls: list[dict[str, Any]] = []
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            copied = {key: call[key] for key in _TOOL_CALL_KEEP_KEYS if key in call}
            function = copied.get("function")
            if isinstance(function, dict):
                copied["function"] = {key: function[key] for key in _FUNCTION_KEEP_KEYS if key in function}
            copied_calls.append(copied)
        current["tool_calls"] = copied_calls
    if current.get("role") == "tool" and not current.get("name"):
        call_id = str(current.get("tool_call_id") or "")
        name = names_by_id.get(call_id)
        if name:
            current["name"] = name
    return current


def _flatten_console_content(content: Any) -> Any:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)
    texts: list[str] = []
    for part in content:
        if isinstance(part, str):
            if part:
                texts.append(part)
            continue
        if not isinstance(part, dict):
            continue
        part_type = str(part.get("type") or "")
        if part_type == "image_url":
            texts.append("[image omitted]")
            continue
        text = part.get("text")
        if isinstance(text, str) and text:
            texts.append(text)
    return "\n".join(texts)


def _content_nonempty(content: Any) -> bool:
    if content is None:
        return False
    if isinstance(content, str):
        return bool(content.strip())
    if isinstance(content, list):
        return any(_content_nonempty(part if not isinstance(part, dict) else part.get("text")) for part in content)
    return bool(str(content).strip())


def _tool_call_names(messages: list[Any]) -> dict[str, str]:
    names: dict[str, str] = {}
    for message in messages:
        if not isinstance(message, dict):
            continue
        for call in message.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            call_id = str(call.get("id") or "")
            function = call.get("function") if isinstance(call.get("function"), dict) else {}
            name = str(function.get("name") or "")
            if call_id and name:
                names[call_id] = name
    return names


def _remember_flag(
    route: ShimModel,
    *,
    flag: str,
    reason: str,
    compat_path: Path | None = None,
) -> bool:
    path = compat_path or DEFAULT_UPSTREAM_COMPAT_PATH
    store = _load_store(path)
    learned_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    patch = {
        flag: True,
        "learned_reason": reason[:500],
        "learned_at": learned_at,
    }
    by_slug = dict(store.get("by_slug") or {})
    by_model = dict(store.get("by_upstream_model") or {})
    changed = False
    if route.slug:
        merged = {**dict(by_slug.get(route.slug) or {}), **patch}
        if by_slug.get(route.slug) != merged:
            by_slug[route.slug] = merged
            changed = True
    if route.model:
        merged = {**dict(by_model.get(route.model) or {}), **patch}
        if by_model.get(route.model) != merged:
            by_model[route.model] = merged
            changed = True
    if not changed:
        return False
    store["by_slug"] = by_slug
    store["by_upstream_model"] = by_model
    _save_store(path, store)
    return True


def _load_store(path: Path | None = None) -> dict[str, Any]:
    compat_path = path or DEFAULT_UPSTREAM_COMPAT_PATH
    if not compat_path.exists():
        return {"by_slug": {}, "by_upstream_model": {}}
    try:
        payload = json.loads(compat_path.read_text())
    except (OSError, json.JSONDecodeError):
        return {"by_slug": {}, "by_upstream_model": {}}
    if not isinstance(payload, dict):
        return {"by_slug": {}, "by_upstream_model": {}}
    return {
        "by_slug": dict(payload.get("by_slug") or {}),
        "by_upstream_model": dict(payload.get("by_upstream_model") or {}),
    }


def _save_store(path: Path, store: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(store, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)

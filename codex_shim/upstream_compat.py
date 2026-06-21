from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any

from .settings import ShimModel

DEFAULT_UPSTREAM_COMPAT_PATH = Path.home() / ".codex-shim" / "upstream-compat.json"
_PARALLEL_TOOL_CALLS_UNSUPPORTED = re.compile(
    r"parallel_tool_calls.*(?:not supported|unsupported)|"
    r"(?:not supported|unsupported).*parallel_tool_calls",
    re.IGNORECASE,
)


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


def apply_openai_chat_compat(body: dict[str, Any], *, omit_parallel_tool_calls: bool) -> dict[str, Any]:
    if not omit_parallel_tool_calls or "parallel_tool_calls" not in body:
        return body
    prepared = dict(body)
    prepared.pop("parallel_tool_calls", None)
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


def remember_parallel_tool_calls_unsupported(
    route: ShimModel,
    *,
    reason: str,
    compat_path: Path | None = None,
) -> bool:
    """Persist learned compat for this route. Returns True when newly recorded."""
    if should_omit_parallel_tool_calls(route, compat_path=compat_path):
        return False
    path = compat_path or DEFAULT_UPSTREAM_COMPAT_PATH
    store = _load_store(path)
    learned_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    entry = {
        "omit_parallel_tool_calls": True,
        "learned_reason": reason[:500],
        "learned_at": learned_at,
    }
    by_slug = dict(store.get("by_slug") or {})
    by_model = dict(store.get("by_upstream_model") or {})
    changed = False
    if route.slug and by_slug.get(route.slug) != entry:
        by_slug[route.slug] = entry
        changed = True
    if route.model and by_model.get(route.model) != entry:
        by_model[route.model] = entry
        changed = True
    if not changed:
        return False
    store["by_slug"] = by_slug
    store["by_upstream_model"] = by_model
    _save_store(path, store)
    return True


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


def prepare_openai_chat_body(route: ShimModel, body: dict[str, Any], *, compat_path: Path | None = None) -> dict[str, Any]:
    return apply_openai_chat_compat(
        body,
        omit_parallel_tool_calls=should_omit_parallel_tool_calls(route, compat_path=compat_path),
    )


def supports_parallel_tool_calls_in_catalog(route: ShimModel, *, compat_path: Path | None = None) -> bool:
    return not should_omit_parallel_tool_calls(route, compat_path=compat_path)


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

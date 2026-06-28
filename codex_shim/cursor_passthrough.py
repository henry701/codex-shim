from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import subprocess
import time
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass
from typing import Any

from .catalog_slugs import cursor_catalog_slug
from .naming import description_for_route, display_name_from_slug, format_cursor_display_name
from .settings import slugify
from .translate import responses_to_chat, strip_think

CURSOR_MODEL_SLUG = cursor_catalog_slug("composer-2.5")
CURSOR_UPSTREAM_MODEL = "composer-2.5"
CURSOR_DISPLAY_NAME = format_cursor_display_name("Composer 2.5")
_LIST_MODELS_RE = re.compile(r"^(\S+)\s+-\s+(.+)$")
_AUTH_PROBE_TTL_SEC = 30.0
_MODELS_CACHE_TTL_SEC = 300.0
_auth_probe_cache: tuple[float, bool] | None = None
_models_cache: tuple[float, dict[str, "CursorCatalogModel"]] | None = None


@dataclass(frozen=True)
class CursorCatalogModel:
    catalog_slug: str
    upstream_id: str
    display_name: str


def cursor_spawn_env() -> dict[str, str]:
    """Environment for cursor-agent child processes.

    Following open-design's subscription-first pattern: a stale
    ``CURSOR_API_KEY`` in the shell must not override ``cursor-agent login``.
    """
    env = os.environ.copy()
    env.pop("CURSOR_API_KEY", None)
    bin_override = env.get("CURSOR_AGENT_BIN", "").strip()
    if bin_override:
        env["PATH"] = f"{os.path.dirname(bin_override)}:{env.get('PATH', '')}"
    return env


def _cursor_agent_bin() -> str:
    override = os.environ.get("CURSOR_AGENT_BIN", "").strip()
    if override:
        return override
    return shutil.which("cursor-agent") or "cursor-agent"


def _is_cursor_auth_failure(text: str) -> bool:
    value = text.strip()
    if not value:
        return False
    lowered = value.lower()
    return any(
        marker in lowered
        for marker in (
            "authentication required",
            "not authenticated",
            "not logged in",
            "please run",
            "agent login",
            "cursor_api_key",
        )
    )


def _probe_cursor_auth() -> bool:
    if os.environ.get("CODEX_SHIM_DISABLE_CURSOR", "").lower() in {"1", "true", "yes", "on"}:
        return False
    if not shutil.which(_cursor_agent_bin()) and not os.environ.get("CURSOR_AGENT_BIN"):
        return False
    try:
        result = subprocess.run(
            [_cursor_agent_bin(), "status"],
            capture_output=True,
            text=True,
            timeout=15,
            env=cursor_spawn_env(),
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    output = f"{result.stdout}\n{result.stderr}"
    if _is_cursor_auth_failure(output):
        return False
    return "logged in" in output.lower()


def cursor_passthrough_available(*, force_refresh: bool = False) -> bool:
    """Return True when cursor-agent is installed and logged in."""
    global _auth_probe_cache
    now = time.monotonic()
    if not force_refresh and _auth_probe_cache is not None:
        cached_at, cached = _auth_probe_cache
        if now - cached_at < _AUTH_PROBE_TTL_SEC:
            return cached
    available = _probe_cursor_auth()
    _auth_probe_cache = (now, available)
    return available


def _fallback_cursor_models() -> dict[str, CursorCatalogModel]:
    return {
        CURSOR_MODEL_SLUG: CursorCatalogModel(
            catalog_slug=CURSOR_MODEL_SLUG,
            upstream_id=CURSOR_UPSTREAM_MODEL,
            display_name=CURSOR_DISPLAY_NAME,
        )
    }


def _parse_cursor_list_models_output(output: str) -> dict[str, CursorCatalogModel]:
    models: dict[str, CursorCatalogModel] = {}
    for line in output.splitlines():
        match = _LIST_MODELS_RE.match(line.strip())
        if not match:
            continue
        upstream_id, display_name = match.group(1).strip(), match.group(2).strip()
        if not upstream_id or upstream_id.lower() == "auto":
            continue
        catalog_slug = cursor_catalog_slug(upstream_id)
        model = CursorCatalogModel(
            catalog_slug=catalog_slug,
            upstream_id=upstream_id,
            display_name=format_cursor_display_name(display_name),
        )
        models[catalog_slug] = model
        models[upstream_id] = model
        legacy_slug = slugify(upstream_id)
        if legacy_slug != catalog_slug:
            models[legacy_slug] = model
    return models or _fallback_cursor_models()


def _load_cursor_catalog_models(*, force_refresh: bool = False) -> dict[str, CursorCatalogModel]:
    global _models_cache
    now = time.monotonic()
    if not force_refresh and _models_cache is not None:
        cached_at, cached = _models_cache
        if now - cached_at < _MODELS_CACHE_TTL_SEC:
            return cached
    if not shutil.which(_cursor_agent_bin()) and not os.environ.get("CURSOR_AGENT_BIN"):
        models = _fallback_cursor_models()
        _models_cache = (now, models)
        return models
    try:
        result = subprocess.run(
            [_cursor_agent_bin(), "--list-models"],
            capture_output=True,
            text=True,
            timeout=30,
            env=cursor_spawn_env(),
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        models = _fallback_cursor_models()
        _models_cache = (now, models)
        return models
    output = f"{result.stdout}\n{result.stderr}"
    models = _parse_cursor_list_models_output(output)
    _models_cache = (now, models)
    return models


def cursor_catalog_models(*, force_refresh: bool = False) -> list[CursorCatalogModel]:
    by_slug: dict[str, CursorCatalogModel] = {}
    for model in _load_cursor_catalog_models(force_refresh=force_refresh).values():
        by_slug[model.catalog_slug] = model
    return sorted(by_slug.values(), key=lambda item: item.display_name.lower())


def is_cursor_passthrough_slug(slug: str) -> bool:
    return slug in _load_cursor_catalog_models()


def cursor_upstream_model(slug: str) -> str:
    model = _load_cursor_catalog_models().get(slug)
    if model is not None:
        return model.upstream_id
    return slug.replace("-", ".")


_WORKSPACE_PROMPT_PATTERNS = (
    re.compile(r"workspace [`'\"](/[^\s`'\"]+)[`'\"]", re.IGNORECASE),
    re.compile(r"working (?:directory|root|folder)(?: is)?:?\s*[`'\"]?(/[^\s`'\"]+)", re.IGNORECASE),
    re.compile(r"\bcwd:\s*(/\S+)", re.IGNORECASE),
    re.compile(r"Current working directory:\s*(/\S+)", re.IGNORECASE),
)


def _extract_workspace_from_text(text: str) -> str | None:
    for pattern in _WORKSPACE_PROMPT_PATTERNS:
        match = pattern.search(text or "")
        if match:
            path = match.group(1).strip().rstrip(".,;")
            if path.startswith("/"):
                return path
    return None


def _workspace_from_mapping(data: Any) -> str | None:
    if not isinstance(data, dict):
        return None
    for key in ("cwd", "working_directory", "workspace", "workspace_root", "working_root"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def resolve_cursor_workspace(
    body: dict[str, Any] | None = None,
    *,
    request_headers: Mapping[str, str] | None = None,
    prompt: str | None = None,
) -> str:
    override = os.environ.get("CODEX_SHIM_CURSOR_WORKSPACE", "").strip()
    if override:
        return override
    if body is not None:
        resolved = _workspace_from_mapping(body.get("metadata"))
        if resolved:
            return resolved
    if request_headers is not None:
        raw = request_headers.get("x-codex-turn-metadata") or request_headers.get("X-Codex-Turn-Metadata")
        if raw:
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                parsed = None
            resolved = _workspace_from_mapping(parsed)
            if resolved:
                return resolved
    if body is not None:
        instructions = body.get("instructions")
        if isinstance(instructions, str):
            resolved = _extract_workspace_from_text(instructions)
            if resolved:
                return resolved
    if prompt:
        resolved = _extract_workspace_from_text(prompt)
        if resolved:
            return resolved
    if body is not None and prompt is None:
        resolved = _extract_workspace_from_text(build_cursor_prompt(body))
        if resolved:
            return resolved
    return os.getcwd()


def cursor_workspace() -> str:
    return resolve_cursor_workspace()


def cursor_passthrough_display_names() -> dict[str, str]:
    return {model.catalog_slug: model.display_name for model in cursor_catalog_models()}


def cursor_catalog_entry(model: CursorCatalogModel) -> dict[str, Any]:
    display_name = model.display_name
    return {
        "slug": model.catalog_slug,
        "display_name": display_name,
        "description": description_for_route(
            display_name,
            "routed through your Cursor subscription (cursor-agent login)",
        ),
        "context_window": 272_000,
        "max_context_window": 272_000,
        "auto_compact_token_limit": 217_600,
        "truncation_policy": {"mode": "tokens", "limit": 64_000},
        "default_reasoning_level": "medium",
        "supported_reasoning_levels": [
            {"effort": "low", "description": "Faster, lighter reasoning"},
            {"effort": "medium", "description": "Balanced speed and reasoning"},
            {"effort": "high", "description": "Deeper reasoning"},
        ],
        "default_reasoning_summary": "auto",
        "reasoning_summary_format": "auto",
        "supports_reasoning_summaries": True,
        "default_verbosity": "low",
        "support_verbosity": False,
        "apply_patch_tool_type": "freeform",
        "web_search_tool_type": "text_and_image",
        "supports_search_tool": True,
        "supports_parallel_tool_calls": True,
        "experimental_supported_tools": [],
        "input_modalities": ["text", "image"],
        "supports_image_detail_original": True,
        "shell_type": "shell_command",
        "visibility": "list",
        "minimal_client_version": "0.0.1",
        "supported_in_api": True,
        "availability_nux": None,
        "upgrade": None,
        "prefer_websockets": False,
        "available_in_plans": ["free", "plus", "pro", "team", "business", "enterprise"],
        "base_instructions": f"You are Codex, a coding agent powered by {display_name}.",
        "model_messages": {
            "instructions_template": f"You are Codex, a coding agent powered by {display_name}.",
            "instructions_variables": {"model_name": display_name},
        },
    }


def cursor_passthrough_entries() -> list[dict[str, Any]]:
    models = cursor_catalog_models()
    if not models:
        return [cursor_catalog_entry(_fallback_cursor_models()[CURSOR_MODEL_SLUG])]
    return [
        cursor_catalog_entry(model)
        for model in models
    ]


def build_cursor_prompt(body: dict[str, Any]) -> str:
    """Convert a Codex Responses payload into a cursor-agent prompt."""
    chat = responses_to_chat(body, cursor_upstream_model(str(body.get("model") or "")))
    sections: list[str] = []
    for message in chat.get("messages") or []:
        role = str(message.get("role") or "user").upper()
        content = _message_content(message)
        if not content:
            continue
        if role in {"SYSTEM", "DEVELOPER"}:
            sections.append(f"[{role}]\n{content}")
        elif role == "ASSISTANT":
            tool_calls = message.get("tool_calls") or []
            if tool_calls:
                rendered_calls = []
                for call in tool_calls:
                    fn = call.get("function") or {}
                    rendered_calls.append(
                        f"{fn.get('name') or 'tool'}({fn.get('arguments') or ''})"
                    )
                sections.append(f"[ASSISTANT]\n{content}\nTool calls: {', '.join(rendered_calls)}")
            else:
                sections.append(f"[ASSISTANT]\n{content}")
        elif role == "TOOL":
            sections.append(f"[TOOL {message.get('tool_call_id', '')}]\n{content}")
        else:
            sections.append(f"[USER]\n{content}")
    prompt = "\n\n".join(sections).strip()
    return prompt or "Continue."


def _message_content(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return strip_think(content)
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") in {"text", "input_text", "output_text"}:
                    parts.append(str(item.get("text") or ""))
                elif item.get("type") in {"input_image", "image_url"}:
                    parts.append("[image omitted for cursor-agent bridge]")
        return strip_think("\n".join(part for part in parts if part))
    return ""


def _extract_cursor_assistant_text(message: Any) -> str:
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if not isinstance(content, list):
        return ""
    parts = [
        str(block.get("text") or "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text" and block.get("text")
    ]
    return "".join(parts)


def _extract_cursor_thinking_text(obj: dict[str, Any]) -> str:
    message = obj.get("message")
    if isinstance(message, dict):
        text = _extract_cursor_assistant_text(message)
        if text:
            return text
    for key in ("text", "thinking", "content"):
        value = obj.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _assistant_is_streaming_delta(obj: dict[str, Any]) -> bool:
    if obj.get("model_call_id"):
        return False
    return "timestamp_ms" in obj


def _assistant_is_segment_boundary(obj: dict[str, Any]) -> bool:
    if obj.get("type") != "assistant":
        return False
    if obj.get("model_call_id"):
        return True
    return "timestamp_ms" not in obj


CURSOR_TOOL_RESULT_MAX_CHARS = 4096
CURSOR_TOOL_JSON_MAX_CHARS = 4096
_CURSOR_TOOL_NOISE_KEYS = frozenset(
    {
        "hookAdditionalContexts",
        "toolCallId",
        "startedAtMs",
        "completedAtMs",
        "parsingResult",
        "conversationId",
        "closeStdin",
        "fileOutputThresholdBytes",
        "hardTimeout",
        "hasInputRedirect",
        "hasOutputRedirect",
        "isBackground",
        "skipApproval",
        "timeoutBehavior",
        "simpleCommands",
    }
)


def _cursor_tool_verbose_json() -> bool:
    env = os.environ.get("CODEX_SHIM_CURSOR_TOOL_VERBOSE", "")
    return env.lower() in {"1", "true", "yes", "on"}


def _truncate_cursor_text(text: str, *, limit: int = CURSOR_TOOL_RESULT_MAX_CHARS) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n… ({len(text)} chars truncated)"


def _strip_tool_noise(value: Any, *, verbose: bool) -> Any:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if not verbose and key in _CURSOR_TOOL_NOISE_KEYS:
                continue
            out[key] = _strip_tool_noise(item, verbose=verbose)
        return out
    if isinstance(value, list):
        return [_strip_tool_noise(item, verbose=verbose) for item in value]
    return value


def _format_cursor_tool_json(payload: dict[str, Any]) -> str:
    verbose = _cursor_tool_verbose_json()
    cleaned = _strip_tool_noise(payload, verbose=verbose)
    text = json.dumps(cleaned, ensure_ascii=False, indent=2)
    return _truncate_cursor_text(text, limit=CURSOR_TOOL_JSON_MAX_CHARS)


def _cursor_tool_payload(tool_body: dict[str, Any]) -> tuple[str | None, dict[str, Any] | None]:
    if "function" in tool_body and isinstance(tool_body.get("function"), dict):
        return "function", tool_body["function"]
    for key, value in tool_body.items():
        if key in {"hookAdditionalContexts", "toolCallId", "startedAtMs", "completedAtMs"}:
            continue
        if isinstance(value, dict) and ("args" in value or "result" in value or key.endswith("ToolCall")):
            return key, value
    return None, None


def _path_from_args(args: dict[str, Any]) -> str:
    return str(args.get("path") or args.get("targetDirectory") or "?")


def _preview_text(value: str, *, limit: int = 240) -> str:
    if not value:
        return ""
    return _truncate_cursor_text(value, limit=limit).replace("\n", " ")


@dataclass(frozen=True)
class CursorToolSpec:
    kind: str
    format_started: Callable[[dict[str, Any]], str]
    format_result: Callable[[dict[str, Any]], str] | None = None


def _format_read_started(payload: dict[str, Any]) -> str:
    args = payload.get("args") or {}
    path = str(args.get("path") or "?")
    limit = args.get("limit")
    if limit is not None:
        return f"`{path}` (limit={limit})"
    return f"`{path}`"


def _format_write_started(payload: dict[str, Any]) -> str:
    args = payload.get("args") or {}
    path = str(args.get("path") or "?")
    preview = _preview_text(str(args.get("fileText") or ""))
    if preview:
        return f"`{path}` — {preview}"
    return f"`{path}`"


def _format_edit_started(payload: dict[str, Any]) -> str:
    args = payload.get("args") or {}
    path = str(args.get("path") or "?")
    preview = _preview_text(str(args.get("streamContent") or args.get("patch") or ""))
    if preview:
        return f"`{path}` — {preview}"
    return f"`{path}`"


def _format_shell_started(payload: dict[str, Any]) -> str:
    from .cursor_bridge import is_bridge_shell_command, parse_bridge_tool_from_shell

    args = payload.get("args") or {}
    command = str(args.get("command") or args.get("cmd") or args.get("script") or "")
    if command and is_bridge_shell_command(command):
        tool = parse_bridge_tool_from_shell(command)
        if tool:
            return f"→ Codex tool: `{tool}`"
    if command:
        return f"`{_preview_text(command)}`"
    return "`(command)`"


def _format_delete_started(payload: dict[str, Any]) -> str:
    return f"`{_path_from_args(payload.get('args') or {})}`"


def _format_glob_started(payload: dict[str, Any]) -> str:
    args = payload.get("args") or {}
    pattern = str(args.get("globPattern") or args.get("pattern") or "*")
    directory = str(args.get("targetDirectory") or args.get("path") or ".")
    return f"`{pattern}` in `{directory}`"


def _format_grep_started(payload: dict[str, Any]) -> str:
    args = payload.get("args") or {}
    pattern = str(args.get("pattern") or "?")
    path = str(args.get("path") or ".")
    return f"`{_preview_text(pattern)}` in `{path}`"


def _format_function_started(payload: dict[str, Any]) -> str:
    name = str(payload.get("name") or "tool")
    args = str(payload.get("arguments") or "")
    return f"`{_preview_text(args)}`"


def _format_glob_result(payload: dict[str, Any]) -> str:
    result = payload.get("result") or {}
    success = result.get("success") if isinstance(result, dict) else None
    if not isinstance(success, dict):
        return ""
    files = success.get("files") or success.get("paths") or success.get("matches")
    if isinstance(files, list):
        lines = [str(item) for item in files[:20]]
        if len(files) > 20:
            lines.append(f"… ({len(files)} files total)")
        return "\n".join(lines)
    total = success.get("totalFiles")
    if total is not None:
        return f"{total} file(s) matched"
    return ""


def _format_grep_result(payload: dict[str, Any]) -> str:
    result = payload.get("result") or {}
    success = result.get("success") if isinstance(result, dict) else None
    if not isinstance(success, dict):
        return ""
    lines: list[str] = []
    workspace_results = success.get("workspaceResults")
    if isinstance(workspace_results, dict):
        for _root, entry in workspace_results.items():
            if not isinstance(entry, dict):
                continue
            content = entry.get("content")
            if not isinstance(content, dict):
                continue
            matches = content.get("matches")
            if not isinstance(matches, list):
                continue
            for file_match in matches:
                if not isinstance(file_match, dict):
                    continue
                file_path = str(file_match.get("file") or "?")
                file_lines = file_match.get("matches")
                if isinstance(file_lines, list) and file_lines:
                    first = file_lines[0]
                    if isinstance(first, dict):
                        line_no = first.get("lineNumber")
                        snippet = str(first.get("content") or "").strip()
                        prefix = f"{file_path}:{line_no}" if line_no is not None else file_path
                        lines.append(f"{prefix}: {_preview_text(snippet, limit=120)}")
                    else:
                        lines.append(file_path)
                else:
                    lines.append(file_path)
    if lines:
        return "\n".join(lines[:20])
    pattern = success.get("pattern")
    if pattern:
        return f"pattern `{pattern}` — no matches"
    return ""


CURSOR_TOOL_SPECS: dict[str, CursorToolSpec] = {
    "readToolCall": CursorToolSpec("read", _format_read_started),
    "writeToolCall": CursorToolSpec("write", _format_write_started),
    "editToolCall": CursorToolSpec("edit", _format_edit_started),
    "deleteToolCall": CursorToolSpec("delete", _format_delete_started),
    "globToolCall": CursorToolSpec("glob", _format_glob_started, _format_glob_result),
    "grepToolCall": CursorToolSpec("grep", _format_grep_started, _format_grep_result),
    "shellToolCall": CursorToolSpec("shell", _format_shell_started),
    "runTerminalCommand": CursorToolSpec("shell", _format_shell_started),
    "bashToolCall": CursorToolSpec("shell", _format_shell_started),
    "terminalToolCall": CursorToolSpec("shell", _format_shell_started),
}


def _cursor_tool_body(tool_call: dict[str, Any]) -> dict[str, Any]:
    body = tool_call.get("tool_call")
    return body if isinstance(body, dict) else tool_call


def _cursor_tool_kind_and_detail(tool_body: dict[str, Any]) -> tuple[str, str, bool]:
    """Return (kind, detail, known)."""
    tool_key, payload = _cursor_tool_payload(tool_body)
    if tool_key == "function" and payload is not None:
        name = str(payload.get("name") or "tool")
        return name, _format_function_started(payload), True
    if tool_key is not None and payload is not None:
        spec = CURSOR_TOOL_SPECS.get(tool_key)
        if spec is not None:
            return spec.kind, spec.format_started(payload), True
        label = re.sub(r"ToolCall$", "", tool_key, flags=re.IGNORECASE).lower() or "tool"
        return label, f"`{tool_key}`", False
    if tool_body:
        key = next(iter(tool_body))
        label = re.sub(r"ToolCall$", "", key, flags=re.IGNORECASE).lower() or "tool"
        return label, f"`{key}`", False
    return "tool", "`(unknown)`", False


def _cursor_tool_result_text(tool_body: dict[str, Any]) -> str:
    tool_key, payload = _cursor_tool_payload(tool_body)
    if tool_key is not None and payload is not None:
        spec = CURSOR_TOOL_SPECS.get(tool_key)
        if spec is not None and spec.format_result is not None:
            formatted = spec.format_result(payload)
            if formatted:
                return _truncate_cursor_text(formatted)

    def _walk(value: Any) -> str:
        if isinstance(value, str) and value.strip():
            return value
        if isinstance(value, dict):
            for key in ("content", "output", "stdout", "stderr", "result", "message", "text"):
                if key in value:
                    nested = _walk(value[key])
                    if nested:
                        return nested
            if "success" in value:
                nested = _walk(value["success"])
                if nested:
                    return nested
            parts: list[str] = []
            for nested_value in value.values():
                nested = _walk(nested_value)
                if nested:
                    parts.append(nested)
            return "\n".join(parts)
        if isinstance(value, list):
            parts = [_walk(item) for item in value]
            return "\n".join(part for part in parts if part)
        return ""

    if payload is not None:
        result = payload.get("result")
        if result is not None:
            text = _walk(result)
            if text:
                return _truncate_cursor_text(text)
    for key in tool_body:
        nested_payload = tool_body.get(key)
        if not isinstance(nested_payload, dict):
            continue
        result = nested_payload.get("result")
        if result is not None:
            text = _walk(result)
            if text:
                return _truncate_cursor_text(text)
    return ""


def _format_unknown_tool_markdown(tool_body: dict[str, Any], *, phase: str) -> str:
    tool_key, payload = _cursor_tool_payload(tool_body)
    body: dict[str, Any] = {"tool": tool_key or "unknown", "phase": phase}
    if isinstance(payload, dict):
        args = payload.get("args")
        if isinstance(args, dict):
            body["args"] = args
        if phase == "completed":
            result = payload.get("result")
            if result is not None:
                body["result"] = result
    elif tool_body:
        body["raw"] = tool_body
    return (
        "**cursor-agent · unknown**\n\n"
        f"```json\n{_format_cursor_tool_json(body)}\n```\n"
    )


def format_cursor_tool_started_markdown(tool_call: dict[str, Any]) -> str:
    body = _cursor_tool_body(tool_call)
    kind, detail, known = _cursor_tool_kind_and_detail(body)
    if not known:
        return _format_unknown_tool_markdown(body, phase="started")
    return f"**cursor-agent · {kind}**\n\n> {detail}\n"


def format_cursor_tool_completed_markdown(tool_call: dict[str, Any]) -> str:
    body = _cursor_tool_body(tool_call)
    kind, _, known = _cursor_tool_kind_and_detail(body)
    if not known:
        return "\n" + _format_unknown_tool_markdown(body, phase="completed")
    result = _cursor_tool_result_text(body)
    if not result:
        return "\n**Result**\n\n_(completed)_\n"
    return f"\n**Result**\n\n```\n{result}\n```\n"


def cursor_tool_display_kind(tool_call: dict[str, Any]) -> str:
    kind, _, _ = _cursor_tool_kind_and_detail(_cursor_tool_body(tool_call))
    return kind


def cursor_tool_is_known(tool_call: dict[str, Any]) -> bool:
    _, _, known = _cursor_tool_kind_and_detail(_cursor_tool_body(tool_call))
    return known


def format_cursor_thinking_markdown(text: str) -> str:
    body = text.strip()
    if not body:
        return ""
    return f"**cursor-agent · thinking**\n\n{body}\n"


class CursorStreamParser:
    """Parse cursor-agent ``stream-json`` lines into normalized shim events."""

    def __init__(self) -> None:
        self.segment_text = ""
        self.final_text = ""
        self.usage: dict[str, Any] | None = None
        self.error: str | None = None
        self._open_tool_calls: set[str] = set()

    def feed_events(self, line: str) -> list[dict[str, Any]]:
        line = line.strip()
        if not line:
            return []
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            return []
        if not isinstance(obj, dict):
            return []

        obj_type = obj.get("type")
        events: list[dict[str, Any]] = []

        if obj_type == "assistant" and obj.get("message"):
            if _assistant_is_segment_boundary(obj):
                events.append({"type": "segment_boundary"})
                self.segment_text = ""
                return events
            if not _assistant_is_streaming_delta(obj):
                return []
            text = _extract_cursor_assistant_text(obj.get("message"))
            if not text:
                return []
            delta = self._delta_for_segment(text)
            if delta:
                events.append({"type": "text_delta", "delta": delta})
            return events

        if obj_type == "tool_call":
            subtype = str(obj.get("subtype") or "")
            call_id = str(obj.get("call_id") or "")
            tool_call = obj.get("tool_call")
            if not isinstance(tool_call, dict):
                tool_call = {}
            if subtype == "started":
                events.append({"type": "segment_boundary"})
                self.segment_text = ""
                if call_id:
                    self._open_tool_calls.add(call_id)
                events.append(
                    {
                        "type": "tool_started",
                        "call_id": call_id or f"tool_{len(self._open_tool_calls)}",
                        "tool_call": tool_call,
                        "markdown": format_cursor_tool_started_markdown({"tool_call": tool_call}),
                    }
                )
                return events
            if subtype == "completed":
                if call_id:
                    self._open_tool_calls.discard(call_id)
                events.append(
                    {
                        "type": "tool_completed",
                        "call_id": call_id,
                        "tool_call": tool_call,
                        "markdown": format_cursor_tool_completed_markdown({"tool_call": tool_call}),
                    }
                )
                return events
            return []

        if obj_type == "thinking":
            subtype = str(obj.get("subtype") or "")
            if subtype == "delta":
                text = str(obj.get("text") or _extract_cursor_thinking_text(obj) or "")
                if text:
                    events.append({"type": "thinking_delta", "delta": text})
                return events
            if subtype == "completed":
                events.append({"type": "thinking_completed"})
                return events
            text = _extract_cursor_thinking_text(obj)
            if text:
                events.append({"type": "thinking_delta", "delta": text})
            return events

        if obj_type == "connection" and str(obj.get("subtype") or "") == "reconnecting":
            events.append(
                {
                    "type": "connection_interrupted",
                    "message": "> _(interrupted — connection reconnecting)_\n",
                }
            )
            self._open_tool_calls.clear()
            return events

        if obj_type == "result":
            if obj.get("subtype") == "error" or obj.get("is_error"):
                self.error = str(obj.get("result") or obj.get("error") or "cursor-agent failed")
            elif isinstance(obj.get("result"), str) and obj.get("result"):
                self.final_text = str(obj["result"])
            usage = obj.get("usage")
            if isinstance(usage, dict):
                self.usage = {
                    "input_tokens": usage.get("inputTokens"),
                    "output_tokens": usage.get("outputTokens"),
                    "cache_read_input_tokens": usage.get("cacheReadTokens"),
                    "cache_creation_input_tokens": usage.get("cacheWriteTokens"),
                }
            return []

        if obj_type == "error":
            self.error = str(obj.get("message") or obj.get("error") or "cursor-agent error")
        return []

    def feed_line(self, line: str) -> str | None:
        for event in self.feed_events(line):
            if event.get("type") == "text_delta":
                return str(event.get("delta") or "")
        return None

    def _delta_for_segment(self, text: str) -> str | None:
        if not text:
            return None
        if self.segment_text and text.startswith(self.segment_text):
            delta = text[len(self.segment_text) :]
            self.segment_text = text
            return delta or None
        self.segment_text += text
        return text


class CursorResponseCollector:
    """Accumulate normalized cursor events into Responses ``output`` items."""

    def __init__(
        self,
        *,
        tool_types: dict[str, str] | None = None,
        tool_resolve: dict[str, tuple[str | None, str]] | None = None,
    ) -> None:
        self.output: list[dict[str, Any]] = []
        self._current_message = ""
        self._tool_reasoning: dict[str, str] = {}
        self._thinking_buffer = ""
        self._next_id = 0
        self._tool_types = tool_types or {}
        self._tool_resolve = tool_resolve or {}

    def consume(self, event: dict[str, Any]) -> None:
        event_type = event.get("type")
        if event_type == "text_delta":
            self._flush_thinking()
            self._current_message += str(event.get("delta") or "")
            return
        if event_type == "segment_boundary":
            self._flush_thinking()
            self._flush_message()
            return
        if event_type == "tool_started":
            self._flush_thinking()
            self._flush_message()
            call_id = str(event.get("call_id") or "")
            self._tool_reasoning[call_id] = str(event.get("markdown") or "")
            return
        if event_type == "tool_completed":
            call_id = str(event.get("call_id") or "")
            text = self._tool_reasoning.pop(call_id, "")
            text += str(event.get("markdown") or "")
            if text.strip():
                self._append_reasoning(text)
            return
        if event_type == "thinking_delta":
            self._thinking_buffer += str(event.get("delta") or "")
            return
        if event_type == "thinking_completed":
            self._flush_thinking()
            return
        if event_type == "connection_interrupted":
            message = str(event.get("message") or "")
            for call_id in list(self._tool_reasoning):
                self._tool_reasoning[call_id] += f"\n{message}"
            return

    def _flush_thinking(self) -> None:
        if not self._thinking_buffer.strip():
            self._thinking_buffer = ""
            return
        self._append_reasoning(format_cursor_thinking_markdown(self._thinking_buffer))
        self._thinking_buffer = ""

    def build_output(self, *, fallback_text: str = "") -> list[dict[str, Any]]:
        self._flush_message()
        for call_id, text in list(self._tool_reasoning.items()):
            if text.strip():
                self._append_reasoning(text + "\n> _(interrupted — connection reconnecting)_\n")
            self._tool_reasoning.pop(call_id, None)
        if not self.output and fallback_text:
            self.output.append(self._message_item(fallback_text))
        return self.output

    def _flush_message(self) -> None:
        if not self._current_message:
            return
        self.output.append(self._message_item(self._current_message))
        self._current_message = ""

    def _append_reasoning(self, text: str) -> None:
        item_id = f"rs_{self._next_id}"
        self._next_id += 1
        self.output.append(
            {
                "id": item_id,
                "type": "reasoning",
                "status": "completed",
                "summary": [{"type": "summary_text", "text": text}],
            }
        )

    def _message_item(self, text: str) -> dict[str, Any]:
        item_id = f"msg_{self._next_id}"
        self._next_id += 1
        return {
            "id": item_id,
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": text, "annotations": []}],
        }

    def append_function_call(
        self,
        *,
        name: str,
        arguments: dict[str, Any],
        call_id: str,
        namespace: str | None = None,
        chat_name: str | None = None,
        tool_types: dict[str, str] | None = None,
        tool_resolve: dict[str, tuple[str | None, str]] | None = None,
    ) -> None:
        from .translate import function_call_item_from_chat_tool, upstream_chat_tool_name

        self._flush_thinking()
        self._flush_message()
        resolved_tool_types = tool_types if tool_types is not None else self._tool_types
        resolved_tool_resolve = tool_resolve if tool_resolve is not None else self._tool_resolve
        fn_name = chat_name or (
            upstream_chat_tool_name(namespace, name) if namespace else name
        )
        call = {
            "id": call_id,
            "function": {
                "name": fn_name,
                "arguments": json.dumps(arguments, separators=(",", ":")),
            },
        }
        self.output.append(
            function_call_item_from_chat_tool(call, resolved_tool_types, resolved_tool_resolve)
        )


def replay_cursor_ndjson(
    lines: list[str] | str,
    *,
    collect_output: bool = False,
) -> tuple[list[dict[str, Any]], CursorStreamParser, CursorResponseCollector | None]:
    """Replay captured cursor-agent NDJSON through the parser (and optional collector)."""
    parser = CursorStreamParser()
    collector = CursorResponseCollector() if collect_output else None
    events: list[dict[str, Any]] = []
    raw_lines = lines.splitlines() if isinstance(lines, str) else lines
    for line in raw_lines:
        for event in parser.feed_events(line):
            events.append(event)
            if collector is not None:
                collector.consume(event)
    return events, parser, collector


async def iter_cursor_agent_events(
    prompt: str,
    model: str,
    *,
    workspace: str | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Spawn cursor-agent and yield normalized stream events."""
    resolved_workspace = workspace or cursor_workspace()
    cmd = [
        _cursor_agent_bin(),
        "--print",
        "--output-format",
        "stream-json",
        "--stream-partial-output",
        "--force",
        "--trust",
        "--workspace",
        resolved_workspace,
        "--model",
        model,
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=cursor_spawn_env(),
    )
    assert proc.stdout is not None
    stderr_chunks: list[bytes] = []

    async def _drain_stderr() -> None:
        if proc.stderr is None:
            return
        while True:
            chunk = await proc.stderr.read(4096)
            if not chunk:
                break
            stderr_chunks.append(chunk)

    stderr_task = asyncio.create_task(_drain_stderr())
    parser = CursorStreamParser()
    stdout_complete = False
    try:
        if proc.stdin is not None:
            proc.stdin.write(prompt.encode("utf-8"))
            await proc.stdin.drain()
            proc.stdin.close()

        buffer = ""
        while True:
            chunk = await proc.stdout.read(4096)
            if not chunk:
                break
            buffer += chunk.decode("utf-8", errors="replace")
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                for event in parser.feed_events(line):
                    yield event
        if buffer.strip():
            for event in parser.feed_events(buffer):
                yield event
        stdout_complete = True
    finally:
        if not stdout_complete and proc.returncode is None:
            proc.kill()
        await proc.wait()
        try:
            await stderr_task
        except asyncio.CancelledError:
            pass

    stderr = b"".join(stderr_chunks).decode("utf-8", errors="replace")
    if parser.error:
        yield {"type": "error", "message": parser.error}
    elif proc.returncode not in (0, None) and not parser.final_text:
        message = stderr.strip() or f"cursor-agent exited with code {proc.returncode}"
        if _is_cursor_auth_failure(message):
            message = (
                "Cursor Agent is not authenticated. Run `cursor-agent login`, "
                "then `cursor-agent status`, and retry."
            )
        yield {"type": "error", "message": message}
    if parser.usage:
        yield {"type": "usage", "usage": parser.usage}
    if parser.final_text:
        yield {"type": "completed", "text": parser.final_text}

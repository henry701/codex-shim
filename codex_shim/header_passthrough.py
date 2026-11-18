from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any, Mapping
from urllib.parse import urlparse, urlunparse

_CODEX_CLI_VERSION_RE = re.compile(r"(\d+\.\d+\.\d+(?:[-.][0-9A-Za-z]+)*)")

from aiohttp import web

CHATGPT_ACCEPT_ENCODING = "zstd, gzip, deflate"

_HOP_BY_HOP_REQUEST = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-connection",
        "transfer-encoding",
        "te",
        "trailer",
        "upgrade",
        "host",
        "content-length",
        # Client→shim body may be zstd/gzip; we re-serialize plain JSON upstream.
        "content-encoding",
    }
)
_SHIM_INTERNAL_REQUEST = frozenset({"x-codex-shim-picker-token"})
_WS_UPGRADE_REQUEST = frozenset(
    {
        "sec-websocket-key",
        "sec-websocket-version",
        "sec-websocket-extensions",
        "sec-websocket-protocol",
    }
)
_RESPONSE_BLOCKLIST = frozenset(
    {
        "content-length",
        "transfer-encoding",
        "connection",
        "content-encoding",
        "keep-alive",
        "proxy-connection",
    }
)
_USAGE_HEADER_MARKERS = (
    "token",
    "cache",
    "usage",
    "ratelimit",
    "rate-limit",
    "quota",
    "processing",
    "request-id",
)


def upstream_headers_from_response(upstream: Any) -> Mapping[str, str]:
    headers = getattr(upstream, "headers", None)
    if headers is None:
        return {}
    return headers


def _header_present(headers: Mapping[str, str], name: str) -> bool:
    lowered = name.lower()
    return any(key.lower() == lowered for key in headers)


def upstream_header_log_enabled() -> bool:
    env = os.environ.get("CODEX_SHIM_UPSTREAM_HEADER_LOG", "")
    if env.lower() in {"1", "true", "yes", "on"}:
        return True
    return os.environ.get("CODEX_SHIM_STREAM_LOG", "").lower() in {"1", "true", "yes", "on"}


def client_headers_for_upstream(
    request_headers: Mapping[str, str],
    *,
    overrides: Mapping[str, str | None] | None = None,
    setdefaults: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Copy client request headers, apply setdefaults for missing keys, then overrides."""
    merged: dict[str, str] = {}
    for key, value in request_headers.items():
        lowered = key.lower()
        if (
            lowered in _HOP_BY_HOP_REQUEST
            or lowered in _SHIM_INTERNAL_REQUEST
            or lowered in _WS_UPGRADE_REQUEST
        ):
            continue
        merged[key] = value
    if setdefaults:
        for key, value in setdefaults.items():
            if not _header_present(merged, key):
                merged[key] = value
    if overrides:
        for key, value in overrides.items():
            if value is None:
                lowered = key.lower()
                for existing in list(merged):
                    if existing.lower() == lowered:
                        merged.pop(existing, None)
            else:
                merged[key] = value
    return merged


KEYLESS_API_KEY = "public"


def openai_upstream_headers(
    request_headers: Mapping[str, str],
    *,
    api_key: str | None = None,
    extra_headers: Mapping[str, str] | None = None,
    accept: str | None = None,
) -> dict[str, str]:
    overrides: dict[str, str | None] = {"Content-Type": "application/json"}
    token = (api_key or "").strip()
    if token == KEYLESS_API_KEY:
        # Free Zen 401s unrecognized bearers, including "Bearer public".
        # Pop Desktop's ChatGPT Authorization so it cannot leak upstream.
        overrides["Authorization"] = None
    elif token:
        overrides["Authorization"] = f"Bearer {token}"
    if accept:
        overrides["Accept"] = accept
    setdefaults = dict(extra_headers or {})
    if "Content-Type" not in setdefaults:
        setdefaults = {"Content-Type": "application/json", **setdefaults}
    merged = client_headers_for_upstream(
        request_headers,
        setdefaults=setdefaults,
        overrides=overrides,
    )
    return merged


# Zen's free tier, since 2026-10-14, 403s unless the request is the v2 OpenCode
# client: this User-Agent exactly, these headers, and /inference/v1 rather than
# /zen/v1. x-opencode-request is a v1 header and is itself a 403.
# Fallback floor only: the UA version normally tracks the installed opencode CLI.
OPENCODE_FREE_CLIENT_VERSION = "2.0.26"
OPENCODE_FREE_USER_AGENT = f"opencode/latest/{OPENCODE_FREE_CLIENT_VERSION}/cli"
OPENCODE_FREE_CLIENT_VERSION_ENV = "CODEX_SHIM_OPENCODE_FREE_CLIENT_VERSION"
_OPENCODE_FREE_MODELS = frozenset({"big-pickle", "grok-code"})
_BASE62 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_OPENCODE_FINGERPRINT_TOOLS = (
    (
        "read",
        "Read the contents of a file or directory. Supports text files, images, and PDFs. Images and PDFs are presented directly to the model. Each text line is prefixed by its 1-based line number as <line>: <content>. The prefix is for reference and is not part of the file content. Directory entries are returned one per line. Use offset and limit to read large files or directories in sections. Prefer one larger read over many small slices, and use grep to find specific content in large files.",
        {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File or directory to read"},
                "offset": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "The line or directory entry to start reading from (1-based)",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "The maximum number of lines or directory entries to read (defaults to and capped at 2000)",
                },
            },
            "required": ["path"],
            "additionalProperties": False,
        },
    ),
    (
        "shell",
        "Execute a shell command and return its output. Commands run on Linux using bash. Quote file paths containing spaces or special characters. Prefer dedicated tools over shell commands when possible. When output is large, the full result is saved to a file and a truncated preview is returned. Rely on automatic truncation unless filtering the output is more useful. Commands accept an optional timeout, background commands have no timeout by default. Background commands return immediately, and you will be notified when they complete.",
        {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command string to execute"},
                "workdir": {
                    "type": "string",
                    "description": "Working directory to execute the command in. Defaults to the current working directory. When possible, avoid changing directories in the command and set the working directory here instead.",
                },
                "timeout": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "Timeout in milliseconds. Set to 0 to disable the timeout. Defaults to 120000 for foreground commands. Background commands have no timeout by default.",
                },
                "background": {
                    "type": "boolean",
                    "description": "Run the command in the background and return immediately (useful for dev servers and long-running builds). You do not need to use '&' at the end of the command when using this parameter. You will be notified when it completes. DO NOT poll for completion.",
                },
            },
            "required": ["command"],
            "additionalProperties": False,
        },
    ),
)


def is_opencode_free_route(route: Any) -> bool:
    """True for keyless OpenCode Zen free routes (oc-free-* and -free model ids)."""
    slug = str(getattr(route, "slug", "") or "")
    model = str(getattr(route, "model", "") or "")
    base_url = str(getattr(route, "base_url", "") or "")
    if slug.startswith("oc-free-"):
        return True
    parsed = urlparse(base_url)
    host = (parsed.hostname or "").lower().rstrip(".")
    if host != "opencode.ai" and not host.endswith(".opencode.ai"):
        return False
    path = parsed.path or ""
    if path == "/zen/go" or path.startswith("/zen/go/"):
        return False
    return model.endswith("-free") or model in _OPENCODE_FREE_MODELS


def opencode_free_inference_base(base_url: str) -> str:
    """Move a Zen free base from /zen/v1 to /inference/v1. Leave /zen/go alone."""
    parsed = urlparse(base_url)
    host = (parsed.hostname or "").lower().rstrip(".")
    if host != "opencode.ai" and not host.endswith(".opencode.ai"):
        return base_url
    path = parsed.path or ""
    if path == "/zen/go" or path.startswith("/zen/go/"):
        return base_url
    if path != "/zen" and not path.startswith("/zen/"):
        return base_url
    rest = path[len("/zen") :]
    return urlunparse(parsed._replace(path="/inference" + rest))


def opencode_free_session_id(signal: str) -> str:
    """Stable ses_ id in the descending layout Zen's free tier accepts."""
    digest = hashlib.sha256(f"ses\0{signal}".encode()).digest()
    current = (
        digest[0] << 28 | digest[1] << 20 | digest[2] << 12 | digest[3] << 4 | digest[4] >> 4
    ) << 12 | 1
    inverted = (~current) & 0xFFFFFFFFFFFF
    time_hex = f"{inverted:012x}"
    number = int.from_bytes(digest[6:16], "big")
    chars = [""] * 14
    for index in range(13, -1, -1):
        chars[index] = _BASE62[number % 62]
        number //= 62
    return f"ses_{time_hex}{''.join(chars)}"


def _request_session_signal(headers: Mapping[str, str]) -> str:
    for name in ("session-id", "thread-id"):
        for key, value in headers.items():
            if key.lower() == name and str(value).strip():
                return str(value).strip()
    return "codex-shim"


def _drop_header(headers: dict[str, str], name: str) -> None:
    lowered = name.lower()
    for key in list(headers):
        if key.lower() == lowered:
            headers.pop(key, None)


def opencode_free_client_version() -> str:
    """Version for the free-tier fingerprint User-Agent.

    The gate compares the client identity, so an installed ``opencode`` CLI of
    a different version must present its own version or the gate 403s.
    Preference: ``CODEX_SHIM_OPENCODE_FREE_CLIENT_VERSION``, then
    ``opencode --version`` on PATH (lazy import avoids a cycle with
    ``discover``; identity-cached like the codex probe), then the hardcoded
    floor.
    """
    explicit = os.environ.get(OPENCODE_FREE_CLIENT_VERSION_ENV, "").strip()
    if explicit:
        return explicit
    from .discover import detect_opencode_cli_version

    return detect_opencode_cli_version() or OPENCODE_FREE_CLIENT_VERSION


def opencode_free_user_agent() -> str:
    return f"opencode/latest/{opencode_free_client_version()}/cli"


def apply_opencode_free_client_headers(
    headers: Mapping[str, str],
    *,
    session_signal: str,
) -> dict[str, str]:
    """Replace the client identity with the v2 OpenCode free-tier fingerprint."""
    merged = dict(headers)
    session = opencode_free_session_id(session_signal)
    for name in (
        "user-agent",
        "authorization",
        "x-opencode-request",
        "x-opencode-client",
        "x-opencode-session",
        "x-opencode-session-id",
        "x-session-affinity",
        "x-session-id",
        "x-opencode-project",
    ):
        _drop_header(merged, name)
    merged["User-Agent"] = opencode_free_user_agent()
    merged["Authorization"] = "Bearer public"
    merged["x-opencode-client"] = "cli"
    merged["x-opencode-session"] = session
    merged["x-opencode-session-id"] = session
    merged["x-session-affinity"] = session
    merged["x-session-id"] = session
    merged["x-opencode-project"] = "global"
    return merged


def ensure_opencode_free_chat_tools(body: dict[str, Any]) -> dict[str, Any]:
    """Offer the read and shell definitions the free-tier gate compares verbatim."""
    prepared = dict(body)
    tools = [tool for tool in (prepared.get("tools") or []) if isinstance(tool, dict)]
    names: set[str] = set()
    for tool in tools:
        function = tool.get("function") if isinstance(tool.get("function"), dict) else tool
        name = function.get("name") if isinstance(function, dict) else None
        if isinstance(name, str) and name:
            names.add(name)
    for name, description, parameters in _OPENCODE_FINGERPRINT_TOOLS:
        if name in names:
            continue
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": parameters,
                },
            }
        )
    prepared["tools"] = tools
    if prepared.get("tool_choice") == "none":
        prepared.pop("tool_choice", None)
    return prepared


def anthropic_upstream_headers(
    request_headers: Mapping[str, str],
    *,
    api_key: str | None = None,
    extra_headers: Mapping[str, str] | None = None,
    accept: str | None = None,
) -> dict[str, str]:
    overrides: dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        overrides["x-api-key"] = api_key
    if accept:
        overrides["Accept"] = accept
    setdefaults = {
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01",
        **dict(extra_headers or {}),
    }
    return client_headers_for_upstream(
        request_headers,
        setdefaults=setdefaults,
        overrides=overrides,
    )


def chatgpt_client_version_for_upstream() -> str | None:
    """System ``codex --version``. Lazy import avoids a cycle with ``discover``."""
    from .discover import chatgpt_codex_client_version

    version = chatgpt_codex_client_version()
    return version or None


def _version_key(version: str) -> tuple[int, ...]:
    parts: list[int] = []
    for bit in version.split("."):
        digits = ""
        for char in bit:
            if char.isdigit():
                digits += char
            else:
                break
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def _codex_cli_version_in_user_agent(user_agent: str) -> str | None:
    match = _CODEX_CLI_VERSION_RE.search(user_agent or "")
    return match.group(1) if match else None


def _upgrade_stale_chatgpt_client_identity(headers: dict[str, str]) -> None:
    """Present the system CLI identity when Desktop still sends an older Codex.

    ChatGPT rejects ``gpt-6-luna`` for a ChatGPT account when the forwarded
    client is older than that model's ``minimal_client_version`` (0.155.0).
    Desktop bundles its own ``codex`` and can lag ``/usr/bin/codex``.
    """
    system_version = chatgpt_client_version_for_upstream()
    if not system_version:
        return
    ua_key = next((key for key in headers if key.lower() == "user-agent"), None)
    if ua_key is None:
        return
    client_version = _codex_cli_version_in_user_agent(headers[ua_key])
    if client_version is None:
        return
    if _version_key(client_version) >= _version_key(system_version):
        return
    headers["User-Agent"] = f"codex_cli_rs/{system_version}"
    for key in list(headers):
        if key.lower() == "originator":
            headers.pop(key)
    headers["originator"] = "codex_cli_rs"


def chatgpt_passthrough_upstream_headers(
    request_headers: Mapping[str, str],
    *,
    access_token: str,
    account_id: str,
    accept: str,
) -> dict[str, str]:
    setdefaults: dict[str, str] = {}
    if not _header_present(request_headers, "Accept-Encoding"):
        setdefaults["Accept-Encoding"] = CHATGPT_ACCEPT_ENCODING
    if account_id and not _header_present(request_headers, "chatgpt-account-id"):
        setdefaults["chatgpt-account-id"] = account_id
    if not _header_present(request_headers, "OpenAI-Beta"):
        setdefaults["OpenAI-Beta"] = "responses=2026-02-06"
    if not _header_present(request_headers, "originator"):
        setdefaults["originator"] = "codex_cli_rs"
    merged = client_headers_for_upstream(
        request_headers,
        setdefaults=setdefaults,
        overrides={
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "Accept": accept,
            # Codex may send the websocket beta header on HTTP compaction turns.
            "OpenAI-Beta": "responses=2026-02-06",
        },
    )
    _upgrade_stale_chatgpt_client_identity(merged)
    return merged


def chatgpt_passthrough_ws_upstream_headers(
    request_headers: Mapping[str, str],
    *,
    access_token: str,
    account_id: str,
) -> dict[str, str]:
    setdefaults: dict[str, str] = {}
    if account_id and not _header_present(request_headers, "chatgpt-account-id"):
        setdefaults["chatgpt-account-id"] = account_id
    if not _header_present(request_headers, "OpenAI-Beta"):
        setdefaults["OpenAI-Beta"] = "responses_websockets=2026-02-06"
    if not _header_present(request_headers, "originator"):
        setdefaults["originator"] = "codex_cli_rs"
    merged = client_headers_for_upstream(
        request_headers,
        setdefaults=setdefaults,
        overrides={
            "Authorization": f"Bearer {access_token}",
        },
    )
    _upgrade_stale_chatgpt_client_identity(merged)
    return merged


def openai_responses_ws_upstream_headers(
    request_headers: Mapping[str, str],
    *,
    api_key: str | None = None,
    extra_headers: Mapping[str, str] | None = None,
) -> dict[str, str]:
    setdefaults = dict(extra_headers or {})
    if not _header_present(request_headers, "OpenAI-Beta") and not _header_present(setdefaults, "OpenAI-Beta"):
        setdefaults["OpenAI-Beta"] = "responses_websockets=2026-02-06"
    return openai_upstream_headers(
        request_headers,
        api_key=api_key,
        extra_headers=setdefaults,
        accept=None,
    )


_WS_UPGRADE_RESPONSE_BLOCKLIST = _RESPONSE_BLOCKLIST | frozenset(
    {
        "upgrade",
        "sec-websocket-accept",
        "sec-websocket-extensions",
        "sec-websocket-protocol",
    }
)


def forwardable_ws_upgrade_headers(upstream_headers: Mapping[str, str]) -> dict[str, str]:
    forwarded: dict[str, str] = {}
    for key, value in upstream_headers.items():
        if key.lower() in _WS_UPGRADE_RESPONSE_BLOCKLIST:
            continue
        forwarded[key] = value
    return forwarded


def forwardable_upstream_response_headers(upstream_headers: Mapping[str, str]) -> dict[str, str]:
    forwarded: dict[str, str] = {}
    for key, value in upstream_headers.items():
        if key.lower() in _RESPONSE_BLOCKLIST:
            continue
        forwarded[key] = value
    return forwarded


def apply_upstream_headers_to_response(
    response: web.Response | web.StreamResponse,
    upstream_headers: Mapping[str, str],
) -> None:
    for key, value in forwardable_upstream_response_headers(upstream_headers).items():
        response.headers[key] = value


def prepare_downstream_sse_response(upstream: Any) -> web.StreamResponse:
    response = web.StreamResponse(
        status=200,
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
    apply_upstream_headers_to_response(response, upstream_headers_from_response(upstream))
    return response


def _usage_header_subset(headers: Mapping[str, str]) -> dict[str, str]:
    subset: dict[str, str] = {}
    for key, value in headers.items():
        lowered = key.lower()
        if any(marker in lowered for marker in _USAGE_HEADER_MARKERS):
            subset[key] = value
    return subset


def _format_usage_body(usage: Mapping[str, Any] | None) -> str:
    if not usage:
        return ""
    normalized = dict(usage)
    details = normalized.get("input_tokens_details")
    if isinstance(details, dict):
        cached = details.get("cached_tokens") or details.get("cache_read_input_tokens")
        if cached is not None:
            normalized.setdefault("_cached_tokens", cached)
    cache_read = normalized.get("cache_read_input_tokens")
    if cache_read is not None:
        normalized.setdefault("_cache_read_input_tokens", cache_read)
    return json.dumps(normalized, separators=(",", ":"), default=str)


def log_upstream_response_headers(
    source: str,
    upstream_headers: Mapping[str, str],
    *,
    usage: Mapping[str, Any] | None = None,
) -> None:
    if not upstream_header_log_enabled():
        return
    usage_headers = _usage_header_subset(upstream_headers)
    parts = [f"[upstream-headers] source={source}"]
    if usage_headers:
        parts.append(f"headers={json.dumps(usage_headers, separators=(',', ':'))}")
    usage_text = _format_usage_body(usage)
    if usage_text:
        parts.append(f"usage={usage_text}")
    if len(parts) == 1 and upstream_headers:
        parts.append(f"headers={json.dumps(dict(upstream_headers), separators=(',', ':'))}")
    print(" ".join(parts), flush=True)


def observe_upstream_response(
    source: str,
    upstream: Any,
    *,
    usage: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    headers = upstream_headers_from_response(upstream)
    log_upstream_response_headers(source, headers, usage=usage)
    return forwardable_upstream_response_headers(headers)

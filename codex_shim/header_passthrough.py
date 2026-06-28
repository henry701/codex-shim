from __future__ import annotations

import json
import os
from typing import Any, Mapping

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
    overrides: Mapping[str, str] | None = None,
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


def openai_upstream_headers(
    request_headers: Mapping[str, str],
    *,
    api_key: str | None = None,
    extra_headers: Mapping[str, str] | None = None,
    accept: str | None = None,
) -> dict[str, str]:
    overrides: dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        overrides["Authorization"] = f"Bearer {api_key}"
    if accept:
        overrides["Accept"] = accept
    setdefaults = dict(extra_headers or {})
    if "Content-Type" not in setdefaults:
        setdefaults = {"Content-Type": "application/json", **setdefaults}
    return client_headers_for_upstream(
        request_headers,
        setdefaults=setdefaults,
        overrides=overrides,
    )


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
    return client_headers_for_upstream(
        request_headers,
        setdefaults=setdefaults,
        overrides={
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "Accept": accept,
        },
    )


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
    return client_headers_for_upstream(
        request_headers,
        setdefaults=setdefaults,
        overrides={
            "Authorization": f"Bearer {access_token}",
        },
    )


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

from __future__ import annotations

import asyncio
import errno
import json
from typing import Any

RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})
RETRYABLE_ERRNOS = frozenset(
    {
        errno.ETIMEDOUT,
        errno.ECONNRESET,
        errno.EPIPE,
        errno.ECONNABORTED,
    }
)


def errno_of(exc: BaseException) -> int | None:
    value = getattr(exc, "errno", None)
    if isinstance(value, int):
        return value
    os_error = getattr(exc, "os_error", None) or getattr(exc, "__cause__", None)
    nested = getattr(os_error, "errno", None) if os_error is not None else None
    return nested if isinstance(nested, int) else None


def is_retryable_exception(exc: BaseException) -> bool:
    try:
        from aiohttp import ClientConnectorError, ServerTimeoutError
        from aiohttp.client_exceptions import ClientConnectionResetError
    except ImportError:  # pragma: no cover
        ClientConnectorError = ServerTimeoutError = ClientConnectionResetError = tuple()  # type: ignore[misc, assignment]
    timeout_types: tuple[type[BaseException], ...] = (TimeoutError, asyncio.TimeoutError)
    aiohttp_types = tuple(
        cls
        for cls in (ServerTimeoutError, ClientConnectorError, ClientConnectionResetError)
        if isinstance(cls, type)
    )
    if isinstance(exc, timeout_types + aiohttp_types):
        return True
    if errno_of(exc) in RETRYABLE_ERRNOS:
        return True
    lowered = str(exc).lower()
    return "closing transport" in lowered or "connection reset" in lowered


def is_retryable_status(status: int | None) -> bool:
    return status in RETRYABLE_STATUS


THROTTLE_NONE = "none"
THROTTLE_RATE_LIMIT = "rate_limit"
THROTTLE_QUOTA = "quota"
THROTTLE_TRANSPORT = "transport"

_QUOTA_TYPES = frozenset(
    {
        "usage_limit_reached",
        "quota_exceeded",
        "quotaexceeded",
        "usage_limit",
    }
)


def _json_object(body: str | None) -> dict[str, Any]:
    if not body:
        return {}
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _error_object(payload: dict[str, Any]) -> dict[str, Any]:
    err = payload.get("error")
    return err if isinstance(err, dict) else {}


def _error_tokens(body: str | None) -> str:
    payload = _json_object(body)
    err = _error_object(payload)
    return " ".join(
        str(item).lower()
        for item in (
            err.get("type"),
            err.get("code"),
            payload.get("type"),
            payload.get("code"),
            err.get("message"),
            payload.get("message"),
        )
        if item is not None
    )


def parse_resets_in_seconds(body: str | None) -> float | None:
    import time as _time
    from datetime import datetime, timezone

    payload = _json_object(body)
    err = _error_object(payload)
    for src in (err, payload):
        value = src.get("resets_in_seconds")
        if value is None:
            value = src.get("resets_in")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return max(0.0, float(value))
        resets_at = src.get("resets_at")
        if isinstance(resets_at, (int, float)) and not isinstance(resets_at, bool):
            return max(0.0, float(resets_at) - _time.time())
        if isinstance(resets_at, str) and resets_at.strip():
            text = resets_at.strip()
            try:
                parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
            except ValueError:
                parsed = None
            if parsed is not None:
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                return max(0.0, parsed.timestamp() - _time.time())
    return None


def is_quota_limit(status: int | None, body: str | None) -> bool:
    del status
    payload = _json_object(body)
    err = _error_object(payload)
    tokens = _error_tokens(body)
    if any(marker in tokens for marker in _QUOTA_TYPES):
        return True
    if err.get("plan_type") and parse_resets_in_seconds(body) is not None:
        return True
    if "usage limit" in tokens and "rate limit" not in tokens:
        return True
    return False


def is_rate_limit(status: int | None, body: str | None) -> bool:
    if is_quota_limit(status, body):
        return False
    if status == 429:
        return True
    compact = (
        _error_tokens(body)
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
    )
    return any(
        marker in compact
        for marker in (
            "freeusagelimiterror",
            "ratelimit",
            "toomanyrequests",
        )
    )


def exception_http_status(exc: BaseException | None) -> int | None:
    if exc is None:
        return None
    for attr in ("status", "code", "status_code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return None


def classify_throttle(
    *,
    status: int | None = None,
    content_type: str | None = None,
    body: str | None = None,
    exc: BaseException | None = None,
) -> str:
    del content_type
    resolved_status = status if status is not None else exception_http_status(exc)
    blob = body
    if blob is None and exc is not None:
        blob = str(exc)
    if is_quota_limit(resolved_status, blob):
        return THROTTLE_QUOTA
    if is_rate_limit(resolved_status, blob):
        return THROTTLE_RATE_LIMIT
    if exc is not None and is_retryable_exception(exc):
        return THROTTLE_TRANSPORT
    if is_retryable_status(resolved_status):
        return THROTTLE_TRANSPORT
    return THROTTLE_NONE


def classify_ws_event_throttle(event: Any) -> str:
    if not isinstance(event, dict):
        return THROTTLE_NONE
    status = event.get("status") if isinstance(event.get("status"), int) else None
    blobs: list[Any] = [event]
    err = event.get("error")
    if isinstance(err, dict):
        blobs.append({"error": err})
        nested_status = err.get("status") or err.get("code")
        if status is None and isinstance(nested_status, int):
            status = nested_status
    response = event.get("response")
    if isinstance(response, dict):
        blobs.append(response)
        inner = response.get("error")
        if isinstance(inner, dict):
            blobs.append({"error": inner})
            nested_status = inner.get("status") or inner.get("code")
            if status is None and isinstance(nested_status, int):
                status = nested_status
    for blob in blobs:
        kind = classify_throttle(status=status, body=json.dumps(blob))
        if kind != THROTTLE_NONE:
            return kind
    return THROTTLE_NONE


_GENERIC_ERROR_MESSAGES = frozenset({"provider returned error", "error"})


def _nonempty_str(value: Any) -> str | None:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped:
            return stripped
    return None


def _error_code_of(value: Any) -> str | None:
    text = _nonempty_str(value)
    if text is not None:
        return text
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    return None


UPSTREAM_FAILURE_FINISH_REASONS = frozenset(
    {
        "error",
        "network_error",
        "provider_error",
        "internal_error",
        "unknown_error",
    }
)


def chat_chunk_upstream_error(chunk: Any) -> tuple[str, str] | None:
    """Return (code, message) when a chat.completion chunk is an upstream failure."""
    if not isinstance(chunk, dict):
        return None
    if chunk.get("error") is not None:
        return parse_upstream_error(json.dumps(chunk), 502)
    choice = (chunk.get("choices") or [None])[0]
    if not isinstance(choice, dict):
        return None
    native = str(choice.get("native_finish_reason") or "").strip().lower()
    finish = str(choice.get("finish_reason") or "").strip().lower()
    reason = ""
    if native in UPSTREAM_FAILURE_FINISH_REASONS:
        reason = native
    elif finish in UPSTREAM_FAILURE_FINISH_REASONS:
        reason = finish
    if not reason:
        return None
    return reason, f"Upstream finished with {reason}"


def parse_upstream_error(body: str, http_status: int) -> tuple[str, str]:
    text = (body or "").strip()
    code = f"upstream_http_{http_status}"
    message = text or f"Upstream returned HTTP {http_status}"
    if not text:
        return code, message
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return code, text[:2000]
    if not isinstance(payload, dict):
        return code, text[:2000]

    err = payload.get("error")
    if isinstance(err, dict):
        nested_message = _nonempty_str(err.get("message"))
        if nested_message is not None:
            message = nested_message
        nested_type = _nonempty_str(err.get("type"))
        if nested_type is not None:
            code = nested_type
        else:
            nested_code = _error_code_of(err.get("code"))
            if nested_code is not None:
                code = nested_code
        meta = err.get("metadata")
        if isinstance(meta, dict):
            provider_code = _error_code_of(meta.get("provider_error_code"))
            if provider_code is not None and nested_type is None:
                code = provider_code
            raw = _nonempty_str(meta.get("raw"))
            if raw is not None and message.lower() in _GENERIC_ERROR_MESSAGES:
                message = raw
    elif isinstance(err, str) and err.strip():
        message = err.strip()

    top_message = payload.get("message")
    if isinstance(top_message, str) and top_message.strip():
        message = top_message.strip()
    top_code = payload.get("code")
    if isinstance(top_code, str) and top_code.strip():
        code = top_code.strip()

    detail = payload.get("detail")
    if isinstance(detail, str) and detail.strip():
        message = detail.strip()
    elif isinstance(detail, list):
        parts = [str(item).strip() for item in detail if str(item).strip()]
        if parts:
            message = "; ".join(parts)

    return code, message


def close_http_response(response: Any) -> None:
    close = getattr(response, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            pass
    release = getattr(response, "release", None)
    if callable(release):
        try:
            release()
        except Exception:
            pass

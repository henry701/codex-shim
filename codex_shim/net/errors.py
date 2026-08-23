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

from __future__ import annotations

import asyncio
import errno
from dataclasses import dataclass
from typing import Any

from aiohttp import ClientConnectorError, ClientSession, ServerTimeoutError
from aiohttp.client_exceptions import ClientConnectionResetError

RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})
RETRYABLE_ERRNOS = frozenset(
    {
        errno.ETIMEDOUT,
        errno.ECONNRESET,
        errno.EPIPE,
        errno.ECONNABORTED,
    }
)
HTML_SITE_DOWN_MARKERS = (
    "unable to load site",
    "status.openai.com",
    "ray id",
    "cf-ray",
    "just a moment",
    "attention required",
    "cloudflare",
)
EXHAUSTED_EDGE_STATUS = 502
EXHAUSTED_EDGE_MESSAGE = "chatgpt edge unavailable"
EXHAUSTED_EDGE_CONTENT_TYPE = "text/plain"
DEFAULT_ATTEMPTS = 3
DEFAULT_BACKOFF_BASE = 0.5
DEFAULT_BACKOFF_FACTOR = 2.0


@dataclass
class ChatgptEdgePost:
    response: Any
    status: int
    content_type: str
    error_text: str | None = None


def is_html_chatgpt_site_down(
    status: int | None,
    content_type: str | None,
    body: str | None,
) -> bool:
    if status != 403:
        return False
    text = body or ""
    content = (content_type or "").lower()
    lowered = text.lower()
    looks_html = (
        "html" in content
        or lowered.lstrip().startswith("<!doctype")
        or "<html" in lowered[:400]
    )
    if not looks_html:
        return False
    return any(marker in lowered for marker in HTML_SITE_DOWN_MARKERS)


def is_retryable_chatgpt_edge(
    *,
    status: int | None = None,
    content_type: str | None = None,
    body: str | None = None,
    exc: BaseException | None = None,
) -> bool:
    if exc is not None:
        return _is_retryable_exception(exc)
    if status in RETRYABLE_STATUS:
        return True
    return is_html_chatgpt_site_down(status, content_type, body)


def _is_retryable_exception(exc: BaseException) -> bool:
    if isinstance(exc, (TimeoutError, asyncio.TimeoutError, ServerTimeoutError, ClientConnectorError, ClientConnectionResetError)):
        return True
    if _errno_of(exc) in RETRYABLE_ERRNOS:
        return True
    return "closing transport" in str(exc).lower() or "connection reset" in str(exc).lower()


def _errno_of(exc: BaseException) -> int | None:
    value = getattr(exc, "errno", None)
    if isinstance(value, int):
        return value
    os_error = getattr(exc, "os_error", None) or getattr(exc, "__cause__", None)
    nested = getattr(os_error, "errno", None) if os_error is not None else None
    return nested if isinstance(nested, int) else None


async def post_chatgpt_with_retry(
    session: ClientSession,
    url: str,
    *,
    json: dict[str, Any],
    headers: dict[str, str],
    attempts: int = DEFAULT_ATTEMPTS,
    backoff_base: float = DEFAULT_BACKOFF_BASE,
    backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
) -> ChatgptEdgePost:
    """POST ChatGPT Codex, retrying edge blips on a fresh TCP connection each try."""
    total = max(1, int(attempts))
    last_post: ChatgptEdgePost | None = None
    last_exc: BaseException | None = None
    for attempt in range(total):
        try:
            response = await session.post(url, json=json, headers=headers)
        except Exception as exc:
            last_exc = exc
            if not is_retryable_chatgpt_edge(exc=exc) or attempt + 1 >= total:
                raise
            _log_retry(url, attempt + 1, total, status=None, detail=str(exc))
            await _backoff(backoff_base, backoff_factor, attempt)
            continue
        last_exc = None
        posted = await _edge_post_from_response(response)
        last_post = posted
        if posted.status < 400:
            return posted
        retryable = is_retryable_chatgpt_edge(
            status=posted.status,
            content_type=posted.content_type,
            body=posted.error_text,
        )
        _close_failed_response(response)
        if not retryable or attempt + 1 >= total:
            if is_html_chatgpt_site_down(posted.status, posted.content_type, posted.error_text):
                return ChatgptEdgePost(
                    response=response,
                    status=EXHAUSTED_EDGE_STATUS,
                    content_type=EXHAUSTED_EDGE_CONTENT_TYPE,
                    error_text=EXHAUSTED_EDGE_MESSAGE,
                )
            return posted
        _log_retry(url, attempt + 1, total, status=posted.status, detail=posted.error_text)
        await _backoff(backoff_base, backoff_factor, attempt)
    if last_post is not None:
        return last_post
    assert last_exc is not None
    raise last_exc


async def _edge_post_from_response(response: Any) -> ChatgptEdgePost:
    status = int(getattr(response, "status", 0) or 0)
    content_type = str(getattr(response, "content_type", None) or "text/plain")
    if status < 400:
        return ChatgptEdgePost(response=response, status=status, content_type=content_type)
    text_fn = getattr(response, "text", None)
    if callable(text_fn):
        error_text = await text_fn()
    else:
        error_text = ""
    return ChatgptEdgePost(
        response=response,
        status=status,
        content_type=content_type,
        error_text=error_text if isinstance(error_text, str) else str(error_text),
    )


def _close_failed_response(response: Any) -> None:
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


def _log_retry(url: str, attempt: int, total: int, *, status: int | None, detail: str | None) -> None:
    status_note = f" status={status}" if status is not None else ""
    preview = (detail or "").replace("\n", " ")[:180]
    extra = f" detail={preview!r}" if preview else ""
    print(
        f"[chatgpt-edge] retry attempt={attempt}/{total} url={url}{status_note}{extra}",
        flush=True,
    )


async def _backoff(base: float, factor: float, attempt: int) -> None:
    delay = base * (factor**attempt)
    if delay > 0:
        await asyncio.sleep(delay)

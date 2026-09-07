from __future__ import annotations

from collections.abc import Callable
from typing import Any

from aiohttp import ClientSession

from .net.errors import RETRYABLE_ERRNOS, RETRYABLE_STATUS, is_retryable_exception
from .net.retry import (
    DEFAULT_ATTEMPTS,
    DEFAULT_BACKOFF_BASE,
    DEFAULT_BACKOFF_FACTOR,
    HttpPostResult,
    RetryPolicy,
    retry_aiohttp_post,
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


class ChatgptEdgePost:
    def __init__(
        self,
        response: Any,
        status: int,
        content_type: str,
        error_text: str | None = None,
    ) -> None:
        self.response = response
        self.status = status
        self.content_type = content_type
        self.error_text = error_text


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
        return is_retryable_exception(exc)
    if status in RETRYABLE_STATUS:
        return True
    return is_html_chatgpt_site_down(status, content_type, body)


def _chatgpt_retry_policy(attempts: int, backoff_base: float, backoff_factor: float) -> RetryPolicy:
    return RetryPolicy(
        attempts=attempts,
        backoff_base=backoff_base,
        backoff_factor=backoff_factor,
        classify=is_retryable_chatgpt_edge,
    )


def _from_posted(posted: HttpPostResult) -> ChatgptEdgePost:
    return ChatgptEdgePost(
        response=posted.response,
        status=posted.status,
        content_type=posted.content_type,
        error_text=posted.error_text,
    )


async def post_chatgpt_with_retry(
    session: ClientSession,
    url: str,
    *,
    json: dict[str, Any],
    headers: dict[str, str],
    attempts: int = DEFAULT_ATTEMPTS,
    backoff_base: float = DEFAULT_BACKOFF_BASE,
    backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
    disconnect_fn: Callable[[], bool] | None = None,
    ping_fn: Callable[..., Any] | None = None,
    keepalive: float | None = None,
) -> ChatgptEdgePost:
    """POST ChatGPT Codex, retrying edge blips on a fresh TCP connection each try."""
    posted = await retry_aiohttp_post(
        session,
        url,
        json=json,
        headers=headers,
        policy=_chatgpt_retry_policy(attempts, backoff_base, backoff_factor),
        label=url,
        disconnect_fn=disconnect_fn,
        ping_fn=ping_fn,
        keepalive=keepalive,
    )
    if is_html_chatgpt_site_down(posted.status, posted.content_type, posted.error_text):
        return ChatgptEdgePost(
            response=posted.response,
            status=EXHAUSTED_EDGE_STATUS,
            content_type=EXHAUSTED_EDGE_CONTENT_TYPE,
            error_text=EXHAUSTED_EDGE_MESSAGE,
        )
    return _from_posted(posted)


# Re-exported for tests / callers that imported these from chatgpt_edge.
__all__ = [
    "ChatgptEdgePost",
    "DEFAULT_ATTEMPTS",
    "DEFAULT_BACKOFF_BASE",
    "DEFAULT_BACKOFF_FACTOR",
    "EXHAUSTED_EDGE_CONTENT_TYPE",
    "EXHAUSTED_EDGE_MESSAGE",
    "EXHAUSTED_EDGE_STATUS",
    "HTML_SITE_DOWN_MARKERS",
    "RETRYABLE_ERRNOS",
    "RETRYABLE_STATUS",
    "is_html_chatgpt_site_down",
    "is_retryable_chatgpt_edge",
    "post_chatgpt_with_retry",
]

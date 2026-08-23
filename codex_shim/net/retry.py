from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen as _stdlib_urlopen

from .errors import (
    RETRYABLE_STATUS,
    close_http_response,
    is_retryable_exception,
    is_retryable_status,
)

DEFAULT_ATTEMPTS = 3
DEFAULT_BACKOFF_BASE = 0.5
DEFAULT_BACKOFF_FACTOR = 2.0

ClassifyFn = Callable[..., bool]


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return max(0.0, float(raw))
    except ValueError:
        return default


@dataclass
class RetryPolicy:
    attempts: int = DEFAULT_ATTEMPTS
    backoff_base: float = DEFAULT_BACKOFF_BASE
    backoff_factor: float = DEFAULT_BACKOFF_FACTOR
    retryable_statuses: frozenset[int] = field(default_factory=lambda: RETRYABLE_STATUS)
    classify: ClassifyFn | None = None
    retry_json_decode: bool = True

    def delay_for(self, attempt: int) -> float:
        return self.backoff_base * (self.backoff_factor**attempt)

    def is_retryable(
        self,
        *,
        status: int | None = None,
        content_type: str | None = None,
        body: str | None = None,
        exc: BaseException | None = None,
    ) -> bool:
        if self.classify is not None:
            return bool(
                self.classify(
                    status=status,
                    content_type=content_type,
                    body=body,
                    exc=exc,
                )
            )
        if exc is not None:
            if self.retry_json_decode and isinstance(exc, json.JSONDecodeError):
                return True
            return is_retryable_exception(exc)
        return status in self.retryable_statuses or is_retryable_status(status)

    async def sleep(self, attempt: int) -> None:
        delay = self.delay_for(attempt)
        if delay > 0:
            await asyncio.sleep(delay)

    def sleep_sync(self, attempt: int, sleep_fn: Callable[[float], None] | None = None) -> None:
        delay = self.delay_for(attempt)
        if delay > 0:
            (sleep_fn or time.sleep)(delay)


def retry_policy_from_env(**overrides: Any) -> RetryPolicy:
    policy = RetryPolicy(
        attempts=_env_int("CODEX_SHIM_RETRY_ATTEMPTS", DEFAULT_ATTEMPTS),
        backoff_base=_env_float("CODEX_SHIM_RETRY_BACKOFF_BASE", DEFAULT_BACKOFF_BASE),
        backoff_factor=_env_float("CODEX_SHIM_RETRY_BACKOFF_FACTOR", DEFAULT_BACKOFF_FACTOR),
    )
    for key, value in overrides.items():
        setattr(policy, key, value)
    return policy


@dataclass
class HttpPostResult:
    response: Any
    status: int
    content_type: str
    error_text: str | None = None


def log_retry(
    label: str,
    attempt: int,
    total: int,
    *,
    status: int | None,
    detail: str | None,
) -> None:
    status_note = f" status={status}" if status is not None else ""
    preview = (detail or "").replace("\n", " ")[:180]
    extra = f" detail={preview!r}" if preview else ""
    print(f"[net-retry] retry attempt={attempt}/{total} {label}{status_note}{extra}", flush=True)


async def retry_aiohttp_post(
    session: Any,
    url: str,
    *,
    json: dict[str, Any] | None = None,
    headers: Mapping[str, str] | None = None,
    policy: RetryPolicy | None = None,
    label: str = "",
) -> HttpPostResult:
    """POST via aiohttp, retrying transport / gateway failures on a fresh connection."""
    policy = policy or retry_policy_from_env()
    total = max(1, int(policy.attempts))
    last_result: HttpPostResult | None = None
    last_exc: BaseException | None = None
    tag = label or url
    for attempt in range(total):
        try:
            response = await session.post(url, json=json, headers=headers)
        except Exception as exc:
            last_exc = exc
            if not policy.is_retryable(exc=exc) or attempt + 1 >= total:
                raise
            log_retry(tag, attempt + 1, total, status=None, detail=str(exc))
            await policy.sleep(attempt)
            continue
        last_exc = None
        status = int(getattr(response, "status", 0) or 0)
        content_type = str(getattr(response, "content_type", None) or "text/plain")
        if status < 400:
            return HttpPostResult(response=response, status=status, content_type=content_type)
        text_fn = getattr(response, "text", None)
        if callable(text_fn):
            error_text = await text_fn()
        else:
            error_text = ""
        if not isinstance(error_text, str):
            error_text = str(error_text)
        posted = HttpPostResult(
            response=response,
            status=status,
            content_type=content_type,
            error_text=error_text,
        )
        last_result = posted
        retryable = policy.is_retryable(
            status=status,
            content_type=content_type,
            body=error_text,
        )
        close_http_response(response)
        if not retryable or attempt + 1 >= total:
            return posted
        log_retry(tag, attempt + 1, total, status=status, detail=error_text)
        await policy.sleep(attempt)
    if last_result is not None:
        return last_result
    assert last_exc is not None
    raise last_exc


@dataclass
class UrllibResult:
    status: int
    body: bytes
    headers: Mapping[str, str]


def request_urllib(
    url: str,
    *,
    method: str = "GET",
    headers: Mapping[str, str] | None = None,
    data: bytes | None = None,
    timeout: float = 20.0,
    policy: RetryPolicy | None = None,
    raise_on_http_error: bool = True,
    urlopen_fn: Callable[..., Any] | None = None,
    sleep_fn: Callable[[float], None] | None = None,
    label: str = "",
    ensure_json: bool = False,
) -> UrllibResult:
    """Sync urllib request with the shared retry policy.

    Wrappers pass their module-level ``urlopen`` so existing tests that patch
    ``codex_shim.discover.urlopen`` / ``codex_shim.nous_auth.urlopen`` keep working.
    """
    policy = policy or retry_policy_from_env()
    opener = urlopen_fn or _stdlib_urlopen
    total = max(1, int(policy.attempts))
    request = Request(url, data=data, headers=dict(headers or {}), method=method)
    last_error: BaseException | None = None
    tag = label or f"{method} {url}"
    for attempt in range(total):
        try:
            with opener(request, timeout=timeout) as response:
                body = response.read()
                status = int(getattr(response, "status", 200) or 200)
                if ensure_json:
                    json.loads(body.decode("utf-8"))
                return UrllibResult(status=status, body=body, headers=getattr(response, "headers", {}))
        except HTTPError as exc:
            last_error = exc
            body = b""
            try:
                body = exc.read()
            except Exception:
                pass
            retryable = policy.is_retryable(status=exc.code, body=body.decode("utf-8", errors="replace"))
            if not retryable or attempt + 1 >= total:
                if raise_on_http_error:
                    raise
                return UrllibResult(status=int(exc.code), body=body, headers=getattr(exc, "headers", {}))
            log_retry(tag, attempt + 1, total, status=exc.code, detail=str(exc))
        except (OSError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
            # urllib discovery used to retry every transport/parse blip. Keep that
            # here so catalog fetches survive generic URLError; aiohttp stays on
            # is_retryable_exception. OAuth sets retry_json_decode=False.
            retryable = isinstance(exc, (OSError, URLError, TimeoutError))
            if isinstance(exc, json.JSONDecodeError):
                retryable = policy.retry_json_decode
            if not retryable:
                retryable = policy.is_retryable(exc=exc)
            if not retryable or attempt + 1 >= total:
                raise
            log_retry(tag, attempt + 1, total, status=None, detail=str(exc))
        policy.sleep_sync(attempt, sleep_fn)
    assert last_error is not None
    raise last_error

from __future__ import annotations

import asyncio
import inspect
import json
import os
import random
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen as _stdlib_urlopen

from .errors import (
    RETRYABLE_STATUS,
    THROTTLE_NONE,
    THROTTLE_QUOTA,
    THROTTLE_RATE_LIMIT,
    THROTTLE_TRANSPORT,
    classify_throttle,
    close_http_response,
    is_retryable_exception,
    is_retryable_status,
    parse_resets_in_seconds,
)
from .sse import ClientDisconnected, keepalive_interval

DEFAULT_ATTEMPTS = 3
DEFAULT_BACKOFF_BASE = 0.5
DEFAULT_BACKOFF_FACTOR = 2.0
DEFAULT_WAIT_BUDGET = 30.0
DEFAULT_429_RETRY_AFTER = 5.0
DEFAULT_RATE_LIMIT_MIN = 60.0
DEFAULT_RATE_LIMIT_MAX = 3600.0
DEFAULT_RATE_LIMIT_JITTER = 0.2
MAX_RETRY_AFTER = DEFAULT_RATE_LIMIT_MAX
_MIN_EXTENSION_DELAY = 0.25
_MAX_EXTENSION_ATTEMPTS = 8
_MAX_ORIGIN_EXPONENT = 16

ClassifyFn = Callable[..., bool]
DisconnectFn = Callable[[], bool]
PingFn = Callable[[], Any]


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


def _clamp(value: float, lo: float, hi: float) -> float:
    if hi < lo:
        hi = lo
    return max(lo, min(hi, value))


def _apply_jitter(delay: float, fraction: float) -> float:
    if fraction <= 0 or delay <= 0:
        return delay
    span = delay * fraction
    return max(0.0, delay + random.uniform(-span, span))


@dataclass
class RetryPolicy:
    attempts: int = DEFAULT_ATTEMPTS
    backoff_base: float = DEFAULT_BACKOFF_BASE
    backoff_factor: float = DEFAULT_BACKOFF_FACTOR
    wait_budget: float = DEFAULT_WAIT_BUDGET
    retryable_statuses: frozenset[int] = field(default_factory=lambda: RETRYABLE_STATUS)
    classify: ClassifyFn | None = None
    retry_json_decode: bool = True
    rate_limit_min: float = DEFAULT_RATE_LIMIT_MIN
    rate_limit_max: float = DEFAULT_RATE_LIMIT_MAX
    rate_limit_jitter: float = DEFAULT_RATE_LIMIT_JITTER

    def delay_for(self, attempt: int) -> float:
        return self.backoff_base * (self.backoff_factor**attempt)

    def should_continue(
        self,
        attempt: int,
        waited: float,
        retry_after: float | None,
    ) -> bool:
        if attempt + 1 < self.attempts:
            return True
        # One-shot policies (OAuth refresh) never extend past attempts.
        if self.attempts <= 1 or self.wait_budget <= 0 or retry_after is None or retry_after <= 0:
            return False
        return waited < self.wait_budget

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


@dataclass
class _RetryState:
    """Shared wait/budget bookkeeping for aiohttp and urllib adapters."""

    attempt: int = 0
    extension_waited: float = 0.0
    extension_attempts: int = 0

    def next_wait(
        self,
        policy: RetryPolicy,
        *,
        retryable: bool,
        retry_after: float | None,
    ) -> float | None:
        if not retryable:
            return None
        if self.attempt + 1 < policy.attempts:
            return max(policy.delay_for(self.attempt), retry_after or 0.0)
        if policy.attempts <= 1 or policy.wait_budget <= 0 or retry_after is None or retry_after <= 0:
            return None
        if self.extension_attempts >= _MAX_EXTENSION_ATTEMPTS:
            return None
        remaining = policy.wait_budget - self.extension_waited
        if remaining <= 0:
            return None
        wait = max(policy.delay_for(self.attempt), retry_after, _MIN_EXTENSION_DELAY)
        return min(wait, remaining)

    def record(self, wait: float, *, extension: bool) -> None:
        self.attempt += 1
        if extension:
            self.extension_waited += wait
            self.extension_attempts += 1

    def in_extension(self, policy: RetryPolicy) -> bool:
        return self.attempt + 1 >= policy.attempts


@dataclass
class OriginBackoff:
    exponent: int = 0
    last_delay: float = 0.0
    not_before: float = 0.0
    kind: str = THROTTLE_NONE


_ORIGIN_BACKOFF: dict[str, OriginBackoff] = {}


def origin_key(url: str) -> str:
    parsed = urlparse(url)
    scheme = (parsed.scheme or "https").lower()
    if scheme == "wss":
        scheme = "https"
    elif scheme == "ws":
        scheme = "http"
    netloc = (parsed.netloc or "").lower()
    return f"{scheme}://{netloc}"


def reset_origin_backoff() -> None:
    _ORIGIN_BACKOFF.clear()


def get_origin_backoff(url: str) -> OriginBackoff | None:
    return _ORIGIN_BACKOFF.get(origin_key(url))


def _cap_exponent(policy: RetryPolicy) -> int:
    floor = max(policy.rate_limit_min, 1e-9)
    ceiling = max(policy.rate_limit_max, floor)
    exponent = 0
    value = floor
    while value < ceiling and exponent < _MAX_ORIGIN_EXPONENT:
        value *= 2.0
        exponent += 1
    return exponent


def configure_origin_backoff(
    url: str,
    *,
    delay: float,
    kind: str = THROTTLE_RATE_LIMIT,
    exponent: int | None = None,
) -> None:
    key = origin_key(url)
    if exponent is None:
        exponent = 1 if kind == THROTTLE_RATE_LIMIT else _cap_exponent(RetryPolicy())
    _ORIGIN_BACKOFF[key] = OriginBackoff(
        exponent=max(0, int(exponent)),
        last_delay=float(delay),
        not_before=time.monotonic() + float(delay),
        kind=kind,
    )


def mark_origin_success(url: str) -> None:
    _ORIGIN_BACKOFF[origin_key(url)] = OriginBackoff()


def note_origin_failure(
    url: str,
    kind: str,
    policy: RetryPolicy,
    *,
    retry_after: float | None = None,
    resets_in_seconds: float | None = None,
) -> float:
    key = origin_key(url)
    state = _ORIGIN_BACKOFF.get(key) or OriginBackoff()
    if kind == THROTTLE_QUOTA:
        raw = policy.rate_limit_max if resets_in_seconds is None else float(resets_in_seconds)
        delay = _clamp(raw, policy.rate_limit_min, policy.rate_limit_max)
        state.kind = THROTTLE_QUOTA
        state.exponent = _cap_exponent(policy)
    else:
        exponential = policy.rate_limit_min * (2 ** max(0, state.exponent))
        raw = max(exponential, retry_after or 0.0)
        delay = _clamp(raw, policy.rate_limit_min, policy.rate_limit_max)
        delay = _apply_jitter(delay, policy.rate_limit_jitter)
        state.kind = THROTTLE_RATE_LIMIT
        state.exponent = min(state.exponent + 1, _MAX_ORIGIN_EXPONENT)
    state.last_delay = delay
    state.not_before = time.monotonic() + delay
    _ORIGIN_BACKOFF[key] = state
    return delay


def log_throttle(
    *,
    origin: str,
    kind: str,
    wait: float,
    exponent: int,
    http_gate: bool,
    ws_session: Any = None,
) -> None:
    extra = f" ws_session={ws_session}" if ws_session is not None else ""
    print(
        f"[throttle] origin={origin} class={kind} wait={wait:.1f}s "
        f"exponent={exponent} http_gate={int(http_gate)}{extra}",
        flush=True,
    )


def _give_up(origin: str, reason: str = "client_disconnect") -> None:
    print(f"[throttle] giving up origin={origin} reason={reason}", flush=True)


def _kind_for_policy(
    policy: RetryPolicy,
    *,
    status: int | None = None,
    content_type: str | None = None,
    body: str | None = None,
    exc: BaseException | None = None,
) -> str:
    kind = classify_throttle(status=status, content_type=content_type, body=body, exc=exc)
    if kind in {THROTTLE_RATE_LIMIT, THROTTLE_QUOTA}:
        return kind
    retryable = policy.is_retryable(status=status, content_type=content_type, body=body, exc=exc)
    if not retryable:
        return THROTTLE_NONE
    if kind == THROTTLE_NONE:
        return THROTTLE_TRANSPORT
    return kind


def _endless_kind(kind: str, policy: RetryPolicy) -> bool:
    return policy.attempts > 1 and kind in {THROTTLE_RATE_LIMIT, THROTTLE_QUOTA}


async def _maybe_ping(ping_fn: PingFn | None) -> None:
    if ping_fn is None:
        return
    result = ping_fn()
    if inspect.isawaitable(result):
        await result


async def throttle_sleep(
    delay: float,
    *,
    origin: str = "",
    disconnect_fn: DisconnectFn | None = None,
    ping_fn: PingFn | None = None,
    keepalive: float | None = None,
) -> None:
    if disconnect_fn is not None and disconnect_fn():
        _give_up(origin)
        raise ClientDisconnected()
    if delay <= 0:
        return
    chunked = ping_fn is not None or disconnect_fn is not None
    if not chunked:
        await asyncio.sleep(delay)
        if disconnect_fn is not None and disconnect_fn():
            _give_up(origin)
            raise ClientDisconnected()
        return
    interval = keepalive_interval(keepalive)
    remaining = delay
    while remaining > 0:
        if disconnect_fn is not None and disconnect_fn():
            _give_up(origin)
            raise ClientDisconnected()
        chunk = min(remaining, interval)
        try:
            await asyncio.sleep(chunk)
        except asyncio.CancelledError:
            _give_up(origin, "cancelled")
            raise
        remaining -= chunk
        if remaining > 0:
            try:
                await _maybe_ping(ping_fn)
            except ClientDisconnected:
                _give_up(origin)
                raise
            except asyncio.CancelledError:
                _give_up(origin, "cancelled")
                raise


def throttle_sleep_sync(
    delay: float,
    sleep_fn: Callable[[float], None] | None = None,
) -> None:
    if delay > 0:
        (sleep_fn or time.sleep)(delay)


async def wait_http_origin_gate(
    url: str,
    policy: RetryPolicy,
    *,
    disconnect_fn: DisconnectFn | None = None,
    ping_fn: PingFn | None = None,
    keepalive: float | None = None,
    label: str = "",
) -> None:
    del label
    if policy.attempts <= 1:
        return
    key = origin_key(url)
    state = _ORIGIN_BACKOFF.get(key)
    if state is None or state.not_before <= 0:
        return
    remaining = state.not_before - time.monotonic()
    if remaining <= 0:
        return
    log_throttle(
        origin=key,
        kind=state.kind or THROTTLE_RATE_LIMIT,
        wait=remaining,
        exponent=state.exponent,
        http_gate=True,
    )
    await throttle_sleep(
        remaining,
        origin=key,
        disconnect_fn=disconnect_fn,
        ping_fn=ping_fn,
        keepalive=keepalive,
    )
    if state.not_before > time.monotonic():
        state.not_before = time.monotonic()


def wait_http_origin_gate_sync(
    url: str,
    policy: RetryPolicy,
    sleep_fn: Callable[[float], None] | None = None,
) -> None:
    if policy.attempts <= 1:
        return
    key = origin_key(url)
    state = _ORIGIN_BACKOFF.get(key)
    if state is None or state.not_before <= 0:
        return
    remaining = state.not_before - time.monotonic()
    if remaining <= 0:
        return
    log_throttle(
        origin=key,
        kind=state.kind or THROTTLE_RATE_LIMIT,
        wait=remaining,
        exponent=state.exponent,
        http_gate=True,
    )
    throttle_sleep_sync(remaining, sleep_fn)
    if state.not_before > time.monotonic():
        state.not_before = time.monotonic()


async def backoff_ws_origin(
    url: str,
    kind: str,
    policy: RetryPolicy,
    *,
    retry_after: float | None = None,
    resets_in_seconds: float | None = None,
    disconnect_fn: DisconnectFn | None = None,
    ping_fn: PingFn | None = None,
    keepalive: float | None = None,
    ws_session: Any = None,
) -> float:
    delay = note_origin_failure(
        url,
        kind,
        policy,
        retry_after=retry_after,
        resets_in_seconds=resets_in_seconds,
    )
    state = get_origin_backoff(url) or OriginBackoff()
    log_throttle(
        origin=origin_key(url),
        kind=kind,
        wait=delay,
        exponent=max(0, state.exponent - 1) if kind == THROTTLE_RATE_LIMIT else state.exponent,
        http_gate=False,
        ws_session=ws_session,
    )
    await throttle_sleep(
        delay,
        origin=origin_key(url),
        disconnect_fn=disconnect_fn,
        ping_fn=ping_fn,
        keepalive=keepalive,
    )
    return delay


def retry_policy_from_env(**overrides: Any) -> RetryPolicy:
    rate_min = _env_float("CODEX_SHIM_RETRY_RATE_LIMIT_MIN", DEFAULT_RATE_LIMIT_MIN)
    rate_max = _env_float("CODEX_SHIM_RETRY_RATE_LIMIT_MAX", DEFAULT_RATE_LIMIT_MAX)
    if rate_max < rate_min:
        rate_max = rate_min
    policy = RetryPolicy(
        attempts=_env_int("CODEX_SHIM_RETRY_ATTEMPTS", DEFAULT_ATTEMPTS),
        backoff_base=_env_float("CODEX_SHIM_RETRY_BACKOFF_BASE", DEFAULT_BACKOFF_BASE),
        backoff_factor=_env_float("CODEX_SHIM_RETRY_BACKOFF_FACTOR", DEFAULT_BACKOFF_FACTOR),
        wait_budget=_env_float("CODEX_SHIM_RETRY_WAIT_BUDGET", DEFAULT_WAIT_BUDGET),
        rate_limit_min=rate_min,
        rate_limit_max=rate_max,
        rate_limit_jitter=_env_float("CODEX_SHIM_RETRY_RATE_LIMIT_JITTER", DEFAULT_RATE_LIMIT_JITTER),
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
    wait: float | None = None,
) -> None:
    status_note = f" status={status}" if status is not None else ""
    preview = (detail or "").replace("\n", " ")[:180]
    extra = f" detail={preview!r}" if preview else ""
    wait_note = f" wait={wait:.1f}s" if wait is not None else ""
    print(f"[net-retry] retry attempt={attempt}/{total} {label}{status_note}{wait_note}{extra}", flush=True)


def _header_get(headers: Any, name: str) -> str | None:
    if headers is None:
        return None
    getter = getattr(headers, "get", None)
    if not callable(getter):
        return None
    for key in (name, name.lower(), name.title()):
        value = getter(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _parse_retry_after_value(raw: Any, *, now: float | None = None) -> float | None:
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        seconds = float(raw)
    else:
        text = str(raw).strip()
        if not text:
            return None
        try:
            seconds = float(text)
        except ValueError:
            seconds = _http_date_retry_seconds(text, now=now)
            if seconds is None:
                return None
    if seconds < 0:
        return None
    return min(MAX_RETRY_AFTER, seconds)


def _http_date_retry_seconds(text: str, *, now: float | None = None) -> float | None:
    try:
        from email.utils import parsedate_to_datetime

        parsed = parsedate_to_datetime(text)
    except (TypeError, ValueError, OverflowError):
        return None
    if parsed is None:
        return None
    if parsed.tzinfo is None:
        from datetime import timezone

        parsed = parsed.replace(tzinfo=timezone.utc)
    stamp = parsed.timestamp()
    current = time.time() if now is None else now
    return max(0.0, stamp - current)


def parse_retry_after(response: Any = None, body: str | None = None) -> float | None:
    """Seconds to wait from Retry-After / OpenRouter metadata. Caps at MAX_RETRY_AFTER."""
    headers = getattr(response, "headers", None)
    from_header = _parse_retry_after_value(_header_get(headers, "Retry-After"))
    if from_header is not None:
        return from_header
    if not body:
        return None
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    err = payload.get("error")
    meta = err.get("metadata") if isinstance(err, dict) else None
    if not isinstance(meta, dict):
        meta = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else None
    if not isinstance(meta, dict):
        return None
    for key in ("retry_after_seconds", "retry_after", "retry-after"):
        parsed = _parse_retry_after_value(meta.get(key))
        if parsed is not None:
            return parsed
    nested = meta.get("headers")
    if isinstance(nested, dict):
        return _parse_retry_after_value(nested.get("Retry-After") or nested.get("retry-after"))
    return None


def inferred_retry_after(
    status: int | None,
    response: Any = None,
    body: str | None = None,
) -> float | None:
    """Retry-After header/body, else a 5s default for HTTP 429."""
    hinted = parse_retry_after(response, body)
    if hinted is not None:
        return hinted
    if status == 429:
        return DEFAULT_429_RETRY_AFTER
    return None


async def _handle_endless_throttle(
    url: str,
    kind: str,
    policy: RetryPolicy,
    *,
    retry_after: float | None,
    body: str | None,
    disconnect_fn: DisconnectFn | None,
    ping_fn: PingFn | None,
    keepalive: float | None,
    http_gate: bool,
    ws_session: Any = None,
) -> None:
    resets = parse_resets_in_seconds(body) if kind == THROTTLE_QUOTA else None
    delay = note_origin_failure(
        url,
        kind,
        policy,
        retry_after=retry_after,
        resets_in_seconds=resets,
    )
    state = get_origin_backoff(url) or OriginBackoff()
    logged_exponent = state.exponent - 1 if kind == THROTTLE_RATE_LIMIT else state.exponent
    log_throttle(
        origin=origin_key(url),
        kind=kind,
        wait=delay,
        exponent=max(0, logged_exponent),
        http_gate=http_gate,
        ws_session=ws_session,
    )
    if http_gate:
        return
    await throttle_sleep(
        delay,
        origin=origin_key(url),
        disconnect_fn=disconnect_fn,
        ping_fn=ping_fn,
        keepalive=keepalive,
    )


async def retry_aiohttp_post(
    session: Any,
    url: str,
    *,
    json: dict[str, Any] | None = None,
    headers: Mapping[str, str] | None = None,
    policy: RetryPolicy | None = None,
    label: str = "",
    disconnect_fn: DisconnectFn | None = None,
    ping_fn: PingFn | None = None,
    keepalive: float | None = None,
) -> HttpPostResult:
    """POST via aiohttp, retrying transport / gateway failures on a fresh connection."""
    policy = policy or retry_policy_from_env()
    total = max(1, int(policy.attempts))
    tag = label or url
    state = _RetryState()
    while True:
        await wait_http_origin_gate(
            url,
            policy,
            disconnect_fn=disconnect_fn,
            ping_fn=ping_fn,
            keepalive=keepalive,
            label=tag,
        )
        try:
            response = await session.post(url, json=json, headers=headers)
        except Exception as exc:
            kind = _kind_for_policy(policy, exc=exc)
            if _endless_kind(kind, policy):
                await _handle_endless_throttle(
                    url,
                    kind,
                    policy,
                    retry_after=None,
                    body=str(exc),
                    disconnect_fn=disconnect_fn,
                    ping_fn=ping_fn,
                    keepalive=keepalive,
                    http_gate=True,
                )
                continue
            wait = state.next_wait(policy, retryable=kind == THROTTLE_TRANSPORT, retry_after=None)
            if wait is None:
                raise
            log_retry(tag, state.attempt + 1, total, status=None, detail=str(exc), wait=wait)
            if wait > 0:
                await throttle_sleep(
                    wait,
                    origin=origin_key(url),
                    disconnect_fn=disconnect_fn,
                    ping_fn=ping_fn,
                    keepalive=keepalive,
                )
            state.record(wait, extension=state.in_extension(policy))
            continue
        status = int(getattr(response, "status", 0) or 0)
        content_type = str(getattr(response, "content_type", None) or "text/plain")
        if status < 400:
            mark_origin_success(url)
            return HttpPostResult(response=response, status=status, content_type=content_type)
        error_text = ""
        body_error: BaseException | None = None
        try:
            text_fn = getattr(response, "text", None)
            if callable(text_fn):
                error_text = await text_fn()
            if not isinstance(error_text, str):
                error_text = str(error_text)
        except Exception as exc:
            body_error = exc
            error_text = str(exc)
        finally:
            close_http_response(response)
        retry_after = inferred_retry_after(status, response, error_text)
        kind = _kind_for_policy(
            policy,
            status=status,
            content_type=content_type,
            body=error_text,
        )
        if _endless_kind(kind, policy):
            await _handle_endless_throttle(
                url,
                kind,
                policy,
                retry_after=retry_after,
                body=error_text,
                disconnect_fn=disconnect_fn,
                ping_fn=ping_fn,
                keepalive=keepalive,
                http_gate=True,
            )
            if body_error is not None and kind == THROTTLE_TRANSPORT:
                raise body_error
            continue
        if body_error is not None:
            wait = state.next_wait(
                policy,
                retryable=kind == THROTTLE_TRANSPORT,
                retry_after=retry_after,
            )
            if wait is None:
                raise body_error
            log_retry(tag, state.attempt + 1, total, status=status, detail=str(body_error), wait=wait)
            if wait > 0:
                await throttle_sleep(
                    wait,
                    origin=origin_key(url),
                    disconnect_fn=disconnect_fn,
                    ping_fn=ping_fn,
                    keepalive=keepalive,
                )
            state.record(wait, extension=state.in_extension(policy))
            continue
        posted = HttpPostResult(
            response=response,
            status=status,
            content_type=content_type,
            error_text=error_text,
        )
        wait = state.next_wait(
            policy,
            retryable=kind == THROTTLE_TRANSPORT,
            retry_after=retry_after,
        )
        if wait is None:
            return posted
        log_retry(tag, state.attempt + 1, total, status=status, detail=error_text, wait=wait)
        if wait > 0:
            await throttle_sleep(
                wait,
                origin=origin_key(url),
                disconnect_fn=disconnect_fn,
                ping_fn=ping_fn,
                keepalive=keepalive,
            )
        state.record(wait, extension=state.in_extension(policy))


async def retry_aiohttp_ws_connect(
    session: Any,
    url: str,
    *,
    headers: Mapping[str, str] | None = None,
    heartbeat: float | None = None,
    policy: RetryPolicy | None = None,
    label: str = "",
    disconnect_fn: DisconnectFn | None = None,
    ping_fn: PingFn | None = None,
    keepalive: float | None = None,
    ws_session: Any = None,
) -> Any:
    policy = policy or retry_policy_from_env()
    total = max(1, int(policy.attempts))
    tag = label or url
    state = _RetryState()
    connect_kwargs: dict[str, Any] = {}
    if headers is not None:
        connect_kwargs["headers"] = headers
    if heartbeat is not None:
        connect_kwargs["heartbeat"] = heartbeat
    while True:
        if disconnect_fn is not None and disconnect_fn():
            _give_up(origin_key(url))
            raise ClientDisconnected()
        try:
            upstream = await session.ws_connect(url, **connect_kwargs)
        except Exception as exc:
            kind = _kind_for_policy(policy, exc=exc)
            if _endless_kind(kind, policy):
                await _handle_endless_throttle(
                    url,
                    kind,
                    policy,
                    retry_after=inferred_retry_after(getattr(exc, "status", None), exc, str(exc)),
                    body=str(exc),
                    disconnect_fn=disconnect_fn,
                    ping_fn=ping_fn,
                    keepalive=keepalive,
                    http_gate=False,
                    ws_session=ws_session,
                )
                continue
            wait = state.next_wait(policy, retryable=kind == THROTTLE_TRANSPORT, retry_after=None)
            if wait is None:
                raise
            log_retry(tag, state.attempt + 1, total, status=None, detail=str(exc), wait=wait)
            if wait > 0:
                await throttle_sleep(
                    wait,
                    origin=origin_key(url),
                    disconnect_fn=disconnect_fn,
                    ping_fn=ping_fn,
                    keepalive=keepalive,
                )
            state.record(wait, extension=state.in_extension(policy))
            continue
        mark_origin_success(url)
        return upstream


def _close_urllib_error(exc: HTTPError) -> None:
    closer = getattr(exc, "close", None)
    if callable(closer):
        try:
            closer()
        except Exception:
            pass


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
    tag = label or f"{method} {url}"
    state = _RetryState()
    while True:
        wait_http_origin_gate_sync(url, policy, sleep_fn)
        try:
            with opener(request, timeout=timeout) as response:
                body = response.read()
                status = int(getattr(response, "status", 200) or 200)
                if ensure_json:
                    json.loads(body.decode("utf-8"))
                if status < 400:
                    mark_origin_success(url)
                return UrllibResult(status=status, body=body, headers=getattr(response, "headers", {}))
        except HTTPError as exc:
            body = b""
            try:
                body = exc.read()
            except Exception:
                pass
            decoded = body.decode("utf-8", errors="replace")
            retry_after = inferred_retry_after(exc.code, exc, decoded)
            kind = _kind_for_policy(policy, status=exc.code, body=decoded)
            _close_urllib_error(exc)
            if _endless_kind(kind, policy):
                resets = parse_resets_in_seconds(decoded) if kind == THROTTLE_QUOTA else None
                delay = note_origin_failure(
                    url,
                    kind,
                    policy,
                    retry_after=retry_after,
                    resets_in_seconds=resets,
                )
                state_origin = get_origin_backoff(url) or OriginBackoff()
                log_throttle(
                    origin=origin_key(url),
                    kind=kind,
                    wait=delay,
                    exponent=max(0, state_origin.exponent - 1)
                    if kind == THROTTLE_RATE_LIMIT
                    else state_origin.exponent,
                    http_gate=True,
                )
                continue
            wait = state.next_wait(policy, retryable=kind == THROTTLE_TRANSPORT, retry_after=retry_after)
            if wait is None:
                if raise_on_http_error:
                    raise
                return UrllibResult(status=int(exc.code), body=body, headers=getattr(exc, "headers", {}))
            log_retry(tag, state.attempt + 1, total, status=exc.code, detail=str(exc), wait=wait)
            if wait > 0:
                throttle_sleep_sync(wait, sleep_fn)
            state.record(wait, extension=state.in_extension(policy))
        except (OSError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            # urllib discovery used to retry every transport/parse blip. Keep that
            # here so catalog fetches survive generic URLError; aiohttp stays on
            # is_retryable_exception. OAuth sets retry_json_decode=False.
            retryable = isinstance(exc, (OSError, URLError, TimeoutError))
            if isinstance(exc, json.JSONDecodeError):
                retryable = policy.retry_json_decode
            if not retryable:
                retryable = policy.is_retryable(exc=exc)
            wait = state.next_wait(policy, retryable=retryable, retry_after=None)
            if wait is None:
                raise
            log_retry(tag, state.attempt + 1, total, status=None, detail=str(exc), wait=wait)
            if wait > 0:
                throttle_sleep_sync(wait, sleep_fn)
            state.record(wait, extension=False)

from __future__ import annotations

import asyncio
import enum
import errno
import inspect
import json
import os
import re
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable
from urllib.parse import urljoin, urlparse, urlunparse

from aiohttp import ClientSession, ClientWebSocketResponse, WSMsgType, web
from aiohttp.client_exceptions import ClientConnectionResetError, ClientError

from .header_passthrough import (
    forwardable_ws_upgrade_headers,
    observe_upstream_response,
    upstream_headers_from_response,
)
from .net.emitters import WsRelayEmitter
from .net.errors import (
    THROTTLE_QUOTA,
    THROTTLE_RATE_LIMIT,
    classify_throttle,
    classify_ws_event_throttle,
    parse_resets_in_seconds,
    throttle_match_cause,
)
from .net.retry import (
    RequestThrottle,
    _env_float,
    backoff_ws_origin,
    retry_aiohttp_ws_connect,
    retry_policy_from_env,
)
from .net.sse import ClientDisconnected

CHATGPT_WS_URL = "wss://chatgpt.com/backend-api/codex/responses"
UPSTREAM_WS_HEARTBEAT = 30
_VERSIONED_BASE_RE = re.compile(r"/v\d+$")
_TERMINAL_EVENT_TYPES = frozenset({"response.completed", "response.failed", "response.incomplete", "error"})

WS_FIRST_EVENT_TIMEOUT_ENV = "CODEX_SHIM_WS_FIRST_EVENT_TIMEOUT_SEC"
WS_IDLE_TIMEOUT_ENV = "CODEX_SHIM_WS_IDLE_TIMEOUT_SEC"
DEFAULT_WS_FIRST_EVENT_TIMEOUT_SEC = 90.0
DEFAULT_WS_IDLE_TIMEOUT_SEC = 300.0
# Codex maps a response.failed with an unrecognised code to ApiError::Retryable.
WS_RELAY_STALL_ERROR_CODE = "upstream_stream_timeout"
CLIENT_GONE_POLL_SEC = 1.0


@dataclass(frozen=True)
class WsRelayTimeouts:
    """Upper bounds on upstream silence while relaying one response.create.

    aiohttp heartbeats keep a live-but-silent upstream open forever, so silence
    has to be bounded explicitly. ``None`` disables a bound (env value ``0``).
    """

    first_event: float | None
    idle: float | None


def ws_relay_timeouts_from_env() -> WsRelayTimeouts:
    def _bound(name: str, default: float) -> float | None:
        value = _env_float(name, default)
        return value if value > 0 else None

    return WsRelayTimeouts(
        first_event=_bound(WS_FIRST_EVENT_TIMEOUT_ENV, DEFAULT_WS_FIRST_EVENT_TIMEOUT_SEC),
        idle=_bound(WS_IDLE_TIMEOUT_ENV, DEFAULT_WS_IDLE_TIMEOUT_SEC),
    )


class RelayStall(enum.Enum):
    FIRST_EVENT_TIMEOUT = "first_event_timeout"
    IDLE_TIMEOUT = "idle_timeout"
    CLIENT_GONE = "client_gone"


@dataclass
class _RelayStats:
    started_at: float = field(default_factory=time.monotonic)
    last_event_at: float | None = None
    upstream_events: int = 0
    max_silence: float = 0.0
    stop: str = "exception"

    def note_event(self) -> None:
        now = time.monotonic()
        self.max_silence = max(self.max_silence, now - (self.last_event_at or self.started_at))
        self.last_event_at = now
        self.upstream_events += 1

    def log_end(self, source: str, terminal: str | None) -> None:
        now = time.monotonic()
        trailing_silence = now - (self.last_event_at or self.started_at)
        print(
            f"[stream-end] {source} "
            f"elapsed={now - self.started_at:.1f}s "
            f"upstream_events={self.upstream_events} "
            f"max_silence={max(self.max_silence, trailing_silence):.1f}s "
            f"terminal={terminal or 'NONE'} stop={self.stop}",
            flush=True,
        )


def _upstream_lane_reusable(ws: Any) -> bool:
    if getattr(ws, "closed", True):
        return False
    getter = getattr(ws, "exception", None)
    if not callable(getter):
        return True
    try:
        exc = getter()
    except Exception:
        return False
    if inspect.isawaitable(exc):
        close = getattr(exc, "close", None)
        if callable(close):
            close()
        return True
    return not isinstance(exc, BaseException)


def ws_passthrough_enabled() -> bool:
    env = os.environ.get("CODEX_SHIM_WS_PASSTHROUGH", "1")
    if env.lower() in {"0", "false", "no", "off"}:
        return False
    return True


def responses_websocket_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if _VERSIONED_BASE_RE.search(base):
        http_url = base + "/responses"
    else:
        http_url = urljoin(base + "/", "v1/responses")
    parsed = urlparse(http_url)
    scheme = "wss" if parsed.scheme == "https" else "ws"
    return urlunparse(parsed._replace(scheme=scheme))


_DEAD_TRANSPORT_ERRNOS = frozenset(
    {
        errno.ECONNRESET,
        errno.EPIPE,
        errno.ECONNABORTED,
        errno.ETIMEDOUT,
    }
)


class WsPassthroughConnectError(Exception):
    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


def _is_upstream_transport_dead(exc: BaseException) -> bool:
    if isinstance(exc, (ClientConnectionResetError, ConnectionResetError, ConnectionError)):
        return True
    if isinstance(exc, ClientError) and "closing transport" in str(exc).lower():
        return True
    if isinstance(exc, OSError) and getattr(exc, "errno", None) in _DEAD_TRANSPORT_ERRNOS:
        return True
    return "closing transport" in str(exc).lower()


async def await_ws_throttle(session: Any, url: str, exc: BaseException) -> bool:
    """Sleep this connection's backoff. True when the error is a throttle wait."""
    status = getattr(exc, "status", None)
    body = str(exc)
    kind = classify_throttle(status=status if isinstance(status, int) else None, body=body, exc=exc)
    policy = retry_policy_from_env()
    if kind not in {THROTTLE_RATE_LIMIT, THROTTLE_QUOTA} or policy.attempts <= 1:
        return False
    disconnect_fn = getattr(session, "client_disconnected", None)
    throttle = getattr(session, "throttle", None)
    if not isinstance(throttle, RequestThrottle):
        throttle = RequestThrottle()
        try:
            session.throttle = throttle
        except (AttributeError, TypeError):
            pass
    await backoff_ws_origin(
        url,
        kind,
        policy,
        throttle=throttle,
        retry_after=None,
        resets_in_seconds=parse_resets_in_seconds(body),
        disconnect_fn=disconnect_fn if callable(disconnect_fn) else None,
        ws_session=id(session),
        cause=throttle_match_cause(
            status=status if isinstance(status, int) else None,
            body=body,
            exc=exc,
        ),
    )
    return True


@dataclass
class WsPassthroughSession:
    """One inbound client WebSocket paired with upstream WS lanes keyed by URL.

    Each inbound Desktop WS connection gets its own ``WsPassthroughSession``.
    Within that session, upstream connections are keyed by upstream URL so model
    swaps that change provider/base URL open a new lane, while swapping back reuses
    an still-open lane and its native ``previous_response_id`` chain.
    """

    client_session: ClientSession
    client_ws: web.WebSocketResponse
    upstream_by_url: dict[str, ClientWebSocketResponse] = field(default_factory=dict)
    last_chained_response_id_by_url: dict[str, str] = field(default_factory=dict)
    thread_ids: set[str] = field(default_factory=set)
    throttle: RequestThrottle = field(default_factory=RequestThrottle)
    # The handler is blocked in the relay, so nobody reads client_ws; a dropped
    # client only shows on the transport. Server wires this to request.transport.
    client_transport_closed: Callable[[], bool] | None = None

    @property
    def upstream_ws(self) -> ClientWebSocketResponse | None:
        if len(self.upstream_by_url) == 1:
            return next(iter(self.upstream_by_url.values()))
        return None

    def note_thread_id(self, thread_id: str | None) -> None:
        if thread_id:
            self.thread_ids.add(thread_id)

    def matches_thread(self, thread_id: str | None) -> bool:
        if not thread_id:
            return False
        return thread_id in self.thread_ids

    def last_upstream_chained_response_id(self, upstream_url: str) -> str | None:
        return self.last_chained_response_id_by_url.get(upstream_url)

    def note_chained_response(self, upstream_url: str, response_id: str | None) -> None:
        if response_id:
            self.last_chained_response_id_by_url[upstream_url] = response_id
        else:
            self.last_chained_response_id_by_url.pop(upstream_url, None)

    def invalidate_native_chain(self, upstream_url: str | None = None) -> None:
        if upstream_url is None:
            self.last_chained_response_id_by_url.clear()
            return
        self.last_chained_response_id_by_url.pop(upstream_url, None)

    def client_disconnected(self) -> bool:
        ws = self.client_ws
        if ws is None or bool(getattr(ws, "closed", False)):
            return True
        return self.client_transport_closed is not None and self.client_transport_closed()

    async def _wait_client_gone(self) -> None:
        while not self.client_disconnected():
            await asyncio.sleep(CLIENT_GONE_POLL_SEC)

    @staticmethod
    async def _next_upstream_message(
        messages: AsyncIterator[Any],
        client_gone: asyncio.Future[None],
        bound_sec: float | None,
        stall: RelayStall,
    ) -> Any | RelayStall | None:
        """Next upstream frame, ``None`` at end of stream, or why we stopped waiting."""
        next_msg = asyncio.ensure_future(messages.__anext__())
        try:
            done, _ = await asyncio.wait(
                {next_msg, client_gone},
                timeout=bound_sec,
                return_when=asyncio.FIRST_COMPLETED,
            )
        finally:
            if not next_msg.done():
                next_msg.cancel()
                await asyncio.gather(next_msg, return_exceptions=True)
        if next_msg in done:
            try:
                return next_msg.result()
            except StopAsyncIteration:
                return None
        if client_gone in done:
            return RelayStall.CLIENT_GONE
        return stall

    async def _drop_lane(self, upstream_url: str) -> None:
        try:
            await self.close_upstream(upstream_url)
        except Exception:
            self.upstream_by_url.pop(upstream_url, None)
            self.last_chained_response_id_by_url.pop(upstream_url, None)

    async def connect_upstream(self, url: str, headers: dict[str, str]) -> tuple[dict[str, str], bool]:
        existing = self.upstream_by_url.get(url)
        if existing is not None and _upstream_lane_reusable(existing):
            return {}, True
        if existing is not None:
            await self.close_upstream(url)
        try:
            upstream_ws = await retry_aiohttp_ws_connect(
                self.client_session,
                url,
                headers=headers,
                heartbeat=UPSTREAM_WS_HEARTBEAT,
                policy=retry_policy_from_env(),
                label=f"ws-connect:{url}",
                disconnect_fn=self.client_disconnected,
                ws_session=id(self),
                throttle=self.throttle,
            )
        except ClientDisconnected as exc:
            raise WsPassthroughConnectError("client disconnected") from exc
        except Exception as exc:
            status = getattr(exc, "status", None)
            raise WsPassthroughConnectError(
                str(exc),
                status=status if isinstance(status, int) else None,
            ) from exc
        self.upstream_by_url[url] = upstream_ws
        self.last_chained_response_id_by_url.pop(url, None)
        upgrade_headers = forwardable_ws_upgrade_headers(upstream_headers_from_response(upstream_ws))
        print(f"[ws-passthrough] connected upstream url={url}", flush=True)
        return upgrade_headers, False

    async def send_response_create(self, body: dict[str, Any], *, upstream_url: str) -> None:
        upstream_ws = self.upstream_by_url.get(upstream_url)
        if upstream_ws is None or upstream_ws.closed:
            raise WsPassthroughConnectError("upstream websocket is not connected")
        payload = {"type": "response.create", **body}
        try:
            await upstream_ws.send_str(json.dumps(payload, separators=(",", ":")))
        except Exception as exc:
            await self.close_upstream(upstream_url)
            raise WsPassthroughConnectError(str(exc)) from exc

    async def relay_until_terminal(
        self,
        *,
        source: str,
        upstream_url: str,
        model_override: str | None = None,
        on_event: Callable[[dict[str, Any]], None] | None = None,
        rewrite_model: Callable[[Any, str | None], None] | None = None,
        write_event: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
        forward_terminal: Callable[[dict[str, Any]], bool] | None = None,
    ) -> dict[str, Any] | None:
        upstream_ws = self.upstream_by_url.get(upstream_url)
        if upstream_ws is None or upstream_ws.closed:
            raise WsPassthroughConnectError("upstream websocket is not connected")

        stats = _RelayStats()

        async def _write_event(event: dict[str, Any]) -> None:
            if write_event is not None:
                await write_event(event)
            else:
                await self.client_ws.send_str(json.dumps(event, separators=(",", ":")))

        emitter = WsRelayEmitter(_write_event, model=model_override or source)
        client_gone = asyncio.ensure_future(self._wait_client_gone())

        try:
            return await self._relay_loop(
                upstream_ws.__aiter__(),
                client_gone,
                upstream_ws=upstream_ws,
                source=source,
                upstream_url=upstream_url,
                model_override=model_override,
                on_event=on_event,
                rewrite_model=rewrite_model,
                write_event=_write_event,
                forward_terminal=forward_terminal,
                emitter=emitter,
                timeouts=ws_relay_timeouts_from_env(),
                stats=stats,
            )
        except asyncio.CancelledError:
            # DownstreamPinger cancels the handler once the client stops answering pings.
            stats.stop = "cancelled"
            await self._drop_lane(upstream_url)
            raise
        finally:
            client_gone.cancel()
            await asyncio.gather(client_gone, return_exceptions=True)
            stats.log_end(source, emitter.terminal_event)

    async def _relay_loop(
        self,
        messages: AsyncIterator[Any],
        client_gone: asyncio.Future[None],
        *,
        upstream_ws: Any,
        source: str,
        upstream_url: str,
        model_override: str | None,
        on_event: Callable[[dict[str, Any]], None] | None,
        rewrite_model: Callable[[Any, str | None], None] | None,
        write_event: Callable[[dict[str, Any]], Awaitable[None]],
        forward_terminal: Callable[[dict[str, Any]], bool] | None,
        emitter: WsRelayEmitter,
        timeouts: WsRelayTimeouts,
        stats: _RelayStats,
    ) -> dict[str, Any] | None:
        terminal_event: dict[str, Any] | None = None
        content_forwarded = False
        try:
            while True:
                if stats.upstream_events == 0:
                    wait, stall = timeouts.first_event, RelayStall.FIRST_EVENT_TIMEOUT
                else:
                    wait, stall = timeouts.idle, RelayStall.IDLE_TIMEOUT
                msg = await self._next_upstream_message(messages, client_gone, wait, stall)
                if msg is None:
                    stats.stop = "upstream_eof"
                    break
                if isinstance(msg, RelayStall):
                    stats.stop = msg.value
                    return await self._abort_stalled_relay(
                        msg,
                        wait=wait,
                        source=source,
                        upstream_url=upstream_url,
                        emitter=emitter,
                    )
                stats.note_event()
                if msg.type == WSMsgType.TEXT:
                    try:
                        event = json.loads(msg.data)
                    except json.JSONDecodeError:
                        await _write_error(
                            self.client_ws,
                            502,
                            "upstream_protocol_error",
                            "upstream emitted non-JSON websocket data",
                        )
                        continue
                    if not isinstance(event, dict):
                        continue
                    if not content_forwarded:
                        kind = classify_ws_event_throttle(event)
                        if kind in {THROTTLE_RATE_LIMIT, THROTTLE_QUOTA}:
                            await self.close_upstream(upstream_url)
                            status = event.get("status")
                            if not isinstance(status, int):
                                err = event.get("error")
                                nested = err.get("status") if isinstance(err, dict) else None
                                status = nested if isinstance(nested, int) else None
                            cause = throttle_match_cause(
                                status=status if isinstance(status, int) else None,
                                body=json.dumps(event),
                            )
                            print(
                                f"[ws-passthrough] upstream throttle kind={kind} "
                                f"cause={cause} event_type={event.get('type')!r} "
                                f"status={status!r} error={event.get('error')!r}",
                                flush=True,
                            )
                            raise WsPassthroughConnectError(
                                json.dumps(event),
                                status=status if isinstance(status, int) else None,
                            )
                    if on_event is not None:
                        on_event(event)
                    if rewrite_model is not None and model_override:
                        rewrite_model(event, model_override)
                    emitter.observe(event)
                    if event.get("type") == "response.completed":
                        response_obj = event.get("response")
                        usage = response_obj.get("usage") if isinstance(response_obj, dict) else None
                        observe_upstream_response(
                            source,
                            upstream_ws,
                            usage=usage if isinstance(usage, dict) else None,
                        )
                    if event.get("type") in _TERMINAL_EVENT_TYPES:
                        stats.stop = "terminal"
                        terminal_event = event
                        if forward_terminal is None or forward_terminal(event):
                            await write_event(event)
                            content_forwarded = True
                        break
                    await write_event(event)
                    content_forwarded = True
                elif msg.type in {WSMsgType.CLOSE, WSMsgType.CLOSED, WSMsgType.ERROR}:
                    stats.stop = "upstream_closed"
                    break
        except Exception as exc:
            if not _is_upstream_transport_dead(exc):
                raise
            stats.stop = "upstream_reset"
            await self._drop_lane(upstream_url)
            if content_forwarded:
                if not emitter.saw_terminal:
                    await emitter.complete()
                    terminal_event = emitter.last_emitted
                return terminal_event
            raise WsPassthroughConnectError(str(exc)) from exc
        if terminal_event is None:
            await self.close_upstream(upstream_url)
            if not emitter.saw_terminal:
                await emitter.complete()
                terminal_event = emitter.last_emitted
        return terminal_event

    async def _abort_stalled_relay(
        self,
        stall: RelayStall,
        *,
        wait: float | None,
        source: str,
        upstream_url: str,
        emitter: WsRelayEmitter,
    ) -> dict[str, Any] | None:
        """Close the lane (and its chain) and end the turn so the client is never left hanging."""
        if stall is RelayStall.CLIENT_GONE:
            print(
                f"[ws-passthrough] relay stopped reason={stall.value} source={source} "
                f"url={upstream_url}; closing lane",
                flush=True,
            )
            await self._drop_lane(upstream_url)
            raise ClientDisconnected("client websocket closed during upstream relay")
        assert wait is not None, "timeout stalls only fire with a bound"
        print(
            f"[ws-passthrough] relay stopped reason={stall.value} after={wait:g}s "
            f"source={source} url={upstream_url}; closing lane",
            flush=True,
        )
        await self._drop_lane(upstream_url)
        what = "no upstream event" if stall is RelayStall.FIRST_EVENT_TIMEOUT else "upstream went silent"
        await emitter.fail(None, f"{what} for {wait:g}s; retry the turn", code=WS_RELAY_STALL_ERROR_CODE)
        return emitter.last_emitted

    async def close_upstream(self, upstream_url: str | None = None) -> None:
        if upstream_url is None:
            for url in list(self.upstream_by_url):
                await self.close_upstream(url)
            return
        upstream_ws = self.upstream_by_url.pop(upstream_url, None)
        self.last_chained_response_id_by_url.pop(upstream_url, None)
        if upstream_ws is not None and not upstream_ws.closed:
            await upstream_ws.close()


async def _write_error(ws: web.WebSocketResponse, status: int, code: str, message: str) -> None:
    await ws.send_str(
        json.dumps(
            {
                "type": "error",
                "status": status,
                "error": {"type": code, "code": code, "message": message},
            },
            separators=(",", ":"),
        )
    )

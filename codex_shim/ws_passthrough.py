from __future__ import annotations

import errno
import inspect
import json
import os
import re
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

CHATGPT_WS_URL = "wss://chatgpt.com/backend-api/codex/responses"
UPSTREAM_WS_HEARTBEAT = 30
_VERSIONED_BASE_RE = re.compile(r"/v\d+$")
_TERMINAL_EVENT_TYPES = frozenset({"response.completed", "response.failed", "error"})


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

    async def connect_upstream(self, url: str, headers: dict[str, str]) -> tuple[dict[str, str], bool]:
        existing = self.upstream_by_url.get(url)
        if existing is not None and _upstream_lane_reusable(existing):
            return {}, True
        if existing is not None:
            await self.close_upstream(url)
        try:
            upstream_ws = await self.client_session.ws_connect(
                url,
                headers=headers,
                heartbeat=UPSTREAM_WS_HEARTBEAT,
            )
        except Exception as exc:
            raise WsPassthroughConnectError(str(exc)) from exc
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

        terminal_event: dict[str, Any] | None = None

        async def _write_event(event: dict[str, Any]) -> None:
            if write_event is not None:
                await write_event(event)
            else:
                await self.client_ws.send_str(json.dumps(event, separators=(",", ":")))

        try:
            async for msg in upstream_ws:
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
                    if on_event is not None:
                        on_event(event)
                    if rewrite_model is not None and model_override:
                        rewrite_model(event, model_override)
                    if event.get("type") == "response.completed":
                        response_obj = event.get("response")
                        usage = response_obj.get("usage") if isinstance(response_obj, dict) else None
                        observe_upstream_response(
                            source,
                            upstream_ws,
                            usage=usage if isinstance(usage, dict) else None,
                        )
                    if event.get("type") in _TERMINAL_EVENT_TYPES:
                        terminal_event = event
                        if forward_terminal is None or forward_terminal(event):
                            await _write_event(event)
                        break
                    await _write_event(event)
                elif msg.type in {WSMsgType.CLOSE, WSMsgType.CLOSED, WSMsgType.ERROR}:
                    break
        except Exception as exc:
            if not _is_upstream_transport_dead(exc):
                raise
            try:
                await self.close_upstream(upstream_url)
            except Exception:
                self.upstream_by_url.pop(upstream_url, None)
                self.last_chained_response_id_by_url.pop(upstream_url, None)
            raise WsPassthroughConnectError(str(exc)) from exc
        if terminal_event is None:
            await self.close_upstream(upstream_url)
        return terminal_event

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

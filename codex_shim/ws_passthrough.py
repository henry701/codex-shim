from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Awaitable, Callable
from urllib.parse import urljoin, urlparse, urlunparse

from aiohttp import ClientSession, ClientWebSocketResponse, WSMsgType, web

from .header_passthrough import (
    forwardable_ws_upgrade_headers,
    observe_upstream_response,
    upstream_headers_from_response,
)

CHATGPT_WS_URL = "wss://chatgpt.com/backend-api/codex/responses"
_VERSIONED_BASE_RE = re.compile(r"/v\d+$")
_TERMINAL_EVENT_TYPES = frozenset({"response.completed", "response.failed", "error"})


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


class WsPassthroughConnectError(Exception):
    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


@dataclass
class WsPassthroughSession:
    client_session: ClientSession
    client_ws: web.WebSocketResponse
    upstream_ws: ClientWebSocketResponse | None = None

    async def connect_upstream(self, url: str, headers: dict[str, str]) -> dict[str, str]:
        if self.upstream_ws is not None and not self.upstream_ws.closed:
            return {}
        try:
            self.upstream_ws = await self.client_session.ws_connect(url, headers=headers)
        except Exception as exc:
            status = getattr(exc, "status", None)
            raise WsPassthroughConnectError(str(exc), status=status) from exc
        upgrade_headers = forwardable_ws_upgrade_headers(upstream_headers_from_response(self.upstream_ws))
        print(f"[ws-passthrough] connected upstream url={url}", flush=True)
        return upgrade_headers

    async def send_response_create(self, body: dict[str, Any]) -> None:
        if self.upstream_ws is None or self.upstream_ws.closed:
            raise WsPassthroughConnectError("upstream websocket is not connected")
        payload = {"type": "response.create", **body}
        await self.upstream_ws.send_str(json.dumps(payload, separators=(",", ":")))

    async def relay_until_terminal(
        self,
        *,
        source: str,
        model_override: str | None = None,
        on_event: Callable[[dict[str, Any]], None] | None = None,
        rewrite_model: Callable[[Any, str | None], None] | None = None,
        write_event: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
    ) -> None:
        if self.upstream_ws is None or self.upstream_ws.closed:
            raise WsPassthroughConnectError("upstream websocket is not connected")

        async def _write_event(event: dict[str, Any]) -> None:
            if write_event is not None:
                await write_event(event)
            else:
                await self.client_ws.send_str(json.dumps(event, separators=(",", ":")))

        async for msg in self.upstream_ws:
            if msg.type == WSMsgType.TEXT:
                try:
                    event = json.loads(msg.data)
                except json.JSONDecodeError:
                    await _write_error(self.client_ws, 502, "upstream_protocol_error", "upstream emitted non-JSON websocket data")
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
                        self.upstream_ws,
                        usage=usage if isinstance(usage, dict) else None,
                    )
                await _write_event(event)
                if event.get("type") in _TERMINAL_EVENT_TYPES:
                    break
            elif msg.type in {WSMsgType.CLOSE, WSMsgType.CLOSED, WSMsgType.ERROR}:
                break

    async def close_upstream(self) -> None:
        if self.upstream_ws is not None and not self.upstream_ws.closed:
            await self.upstream_ws.close()
        self.upstream_ws = None


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

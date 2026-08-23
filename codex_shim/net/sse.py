from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import AsyncIterator
from contextvars import ContextVar
from typing import Any

SSE_KEEPALIVE_INTERVAL = 15.0
PING_BYTES = b": ping\n\n"
_DISCONNECT_NAMES = frozenset(
    {
        "ClientConnectionResetError",
        "ClientConnectionError",
        "ClientPayloadError",
    }
)

_current_writer: ContextVar[DownstreamWriter | None] = ContextVar("codex_shim_downstream_writer", default=None)


class ClientDisconnected(Exception):
    """Raised when the downstream Codex client closes the SSE connection."""


def keepalive_interval(override: float | None = None) -> float:
    if override is not None:
        return max(0.05, float(override))
    raw = os.environ.get("CODEX_SHIM_SSE_KEEPALIVE_INTERVAL", "").strip()
    if raw:
        try:
            return max(0.05, float(raw))
        except ValueError:
            pass
    return SSE_KEEPALIVE_INTERVAL


def request_disconnected(request: Any | None) -> bool:
    if request is None:
        return False
    transport = getattr(request, "transport", None)
    if transport is not None and transport.is_closing():
        return True
    protocol = getattr(request, "protocol", None)
    if protocol is not None:
        proto_transport = getattr(protocol, "transport", None)
        if proto_transport is not None and proto_transport.is_closing():
            return True
    return False


async def close_upstream(upstream: Any) -> None:
    if upstream is None:
        return
    try:
        upstream.close()
    except Exception:
        pass
    try:
        upstream.release()
    except Exception:
        pass


def _is_disconnect(exc: BaseException) -> bool:
    if isinstance(exc, (ConnectionResetError, ConnectionError, ClientDisconnected)):
        return True
    return exc.__class__.__name__ in _DISCONNECT_NAMES


async def write_bytes(response: Any, data: bytes, *, content: bool = True) -> None:
    """Write downstream bytes, serializing through the active StreamGuard writer if any."""
    writer = _current_writer.get()
    if writer is not None and writer.matches(response):
        await writer.write(data, content=content)
        return
    await _raw_write(response, data)


async def _raw_write(response: Any, data: bytes) -> None:
    try:
        await response.write(data)
    except ClientDisconnected:
        raise
    except Exception as exc:
        if _is_disconnect(exc):
            raise ClientDisconnected() from exc
        raise


async def write_sse(response: Any, payload: dict[str, Any], *, content: bool = True) -> None:
    await write_bytes(
        response,
        f"data: {json.dumps(payload, separators=(',', ':'))}\n\n".encode(),
        content=content,
    )


async def write_anthropic_sse(response: Any, event: str, payload: dict[str, Any], *, content: bool = True) -> None:
    data = json.dumps(payload, separators=(",", ":"))
    await write_bytes(response, f"event: {event}\ndata: {data}\n\n".encode(), content=content)


async def iter_upstream_chunks(content: Any, request: Any | None = None) -> AsyncIterator[bytes]:
    """Read upstream bytes until EOF or client disconnect.

    With ``handler_cancellation=True`` on the aiohttp runner, client STOP
    cancels the handler task and ``readany()`` raises ``CancelledError``.
    """
    del request
    try:
        while True:
            chunk = await content.readany()
            if not chunk:
                break
            yield chunk
    except asyncio.CancelledError:
        raise


async def sse_lines(upstream: Any, request: Any | None = None) -> AsyncIterator[str]:
    buffer = b""
    content = upstream.content
    try:
        async for chunk in iter_upstream_chunks(content, request):
            buffer += chunk
            while b"\n" in buffer:
                raw, buffer = buffer.split(b"\n", 1)
                line = raw.decode("utf-8", errors="replace").strip()
                if line.startswith("data:"):
                    yield line[5:].strip()
    except asyncio.CancelledError:
        await close_upstream(upstream)
        raise
    tail = buffer.decode("utf-8", errors="replace").strip()
    if tail.startswith("data:"):
        yield tail[5:].strip()


class DownstreamWriter:
    """Serializes writes so a keepalive pinger cannot interleave into an event."""

    def __init__(self, response: Any):
        self.response = response
        self._lock = asyncio.Lock()
        self.last_write_at = time.monotonic()
        self.bytes_written = 0
        self.ping_count = 0
        self.content_written = False
        self.token: Any = None

    def matches(self, response: Any) -> bool:
        return response is self.response

    def _ready(self) -> bool:
        if self.response is None:
            return False
        if getattr(self.response, "prepared", False):
            return True
        return getattr(self.response, "_payload_writer", None) is not None

    async def write(self, data: bytes, *, content: bool = True) -> None:
        async with self._lock:
            await _raw_write(self.response, data)
            now = time.monotonic()
            self.last_write_at = now
            self.bytes_written += len(data)
            if content:
                self.content_written = True

    async def ping(self) -> bool:
        if not self._ready():
            return False
        async with self._lock:
            await _raw_write(self.response, PING_BYTES)
            self.last_write_at = time.monotonic()
            self.ping_count += 1
        return True

    def activate(self) -> None:
        self.token = _current_writer.set(self)

    def deactivate(self) -> None:
        if self.token is not None:
            _current_writer.reset(self.token)
            self.token = None


class DownstreamPinger:
    """Ping the downstream SSE whenever it has been idle, independent of upstream."""

    def __init__(self, writer: DownstreamWriter, interval: float | None = None):
        self.writer = writer
        self.interval = keepalive_interval(interval)
        self._task: asyncio.Task[None] | None = None

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="codex-shim-sse-ping")

    async def stop(self) -> None:
        task = self._task
        self._task = None
        if task is None:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    async def _run(self) -> None:
        try:
            while True:
                await asyncio.sleep(self.interval)
                idle = time.monotonic() - self.writer.last_write_at
                if idle >= self.interval:
                    try:
                        await self.writer.ping()
                    except ClientDisconnected:
                        return
        except asyncio.CancelledError:
            return

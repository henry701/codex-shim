from __future__ import annotations

import asyncio
import time
import traceback
from collections.abc import AsyncIterator
from typing import Any, Protocol

from .sse import (
    ClientDisconnected,
    DownstreamPinger,
    DownstreamWriter,
    close_upstream,
    sse_lines,
    write_bytes,
)


class TerminalEmitter(Protocol):
    already_emitted: bool

    async def complete(self, response: Any, *, upstream_saw_done: bool) -> str: ...

    async def fail(self, response: Any, message: str, *, code: str) -> str: ...


class StreamGuard:
    """Guarantee a terminal event, keepalive pings, and a [stream-end] log line.

    Wrap every streaming path. Wire-format conversion stays in the emitter.
    Retry is the caller's job: call ``abandon()`` before the first content
    byte so this guard skips the terminal event and the caller can POST again.
    """

    def __init__(
        self,
        response: Any | None,
        emitter: TerminalEmitter,
        *,
        label: str,
        keepalive: bool = True,
        interval: float | None = None,
        finish_reason: Any | None = None,
        upstream: Any | None = None,
    ):
        self.response = response
        self.emitter = emitter
        self.label = label
        self.keepalive = keepalive
        self.interval = interval
        self.finish_reason = finish_reason
        self.upstream = upstream
        self.writer = DownstreamWriter(response) if response is not None else None
        self._pinger: DownstreamPinger | None = None
        self._owner_task: asyncio.Task[Any] | None = None
        self.started_at = time.monotonic()
        self.last_event_at = self.started_at
        self.max_silence = 0.0
        self.upstream_events = 0
        self.upstream_saw_done = False
        self.terminal_event: str | None = None
        self.abandoned = False
        self._entered = False

    @property
    def can_retry(self) -> bool:
        if self.abandoned:
            return True
        if self.writer is None:
            return True
        return not self.writer.content_written

    @property
    def ping_count(self) -> int:
        return 0 if self.writer is None else self.writer.ping_count

    def abandon(self) -> None:
        """Skip terminal synthesis on exit so the caller can retry the upstream POST."""
        self.abandoned = True
        self._deactivate()

    def attach_upstream(self, upstream: Any) -> None:
        self.upstream = upstream

    def mark_upstream_done(self) -> None:
        self.upstream_saw_done = True

    def note_upstream_event(self) -> None:
        now = time.monotonic()
        self.max_silence = max(self.max_silence, now - self.last_event_at)
        self.last_event_at = now
        self.upstream_events += 1

    async def write(self, data: bytes, *, content: bool = True) -> None:
        if self.response is None:
            return
        await write_bytes(self.response, data, content=content)

    async def iter_sse(self, upstream: Any | None = None, request: Any | None = None) -> AsyncIterator[str]:
        src = self.upstream if upstream is None else upstream
        if src is None:
            return
        async for line in sse_lines(src, request):
            self.note_upstream_event()
            yield line

    async def note_upstream_disconnect(self, exc: BaseException) -> None:
        print(
            f"[stream] {self.label} upstream stream disconnected: {type(exc).__name__}: {exc}",
            flush=True,
        )
        if self.response is None or self.emitter.already_emitted:
            return
        try:
            self.terminal_event = await self.emitter.fail(
                self.response,
                f"Upstream stream disconnected: {exc}",
                code="upstream_disconnect",
            )
        except ClientDisconnected:
            raise
        except Exception:
            pass

    async def __aenter__(self) -> StreamGuard:
        self._entered = True
        self._owner_task = asyncio.current_task()
        self.started_at = time.monotonic()
        self.last_event_at = self.started_at
        if self.writer is not None:
            self.writer.activate()
            if self.keepalive:
                self._pinger = DownstreamPinger(
                    self.writer,
                    self.interval,
                    owner_task=self._owner_task,
                )
                self._pinger.start()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        swallowed = False
        try:
            if self.abandoned:
                swallowed = False
            elif exc_type is ClientDisconnected:
                print(f"[cancel] client disconnected during {self.label}", flush=True)
                swallowed = True
            elif exc_type is asyncio.CancelledError:
                print(f"[cancel] client disconnected during {self.label}", flush=True)
            elif exc_type is not None:
                traceback.print_exc()
                print(
                    f"[stream] {self.label} internal stream error: {exc_type.__name__}: {exc}",
                    flush=True,
                )
                await self._fail_terminal(f"Shim stream error: {exc_type.__name__}: {exc}", "shim_stream_error")
                swallowed = True
            else:
                await self._complete_terminal()
        except ClientDisconnected:
            print(f"[cancel] client disconnected during {self.label}", flush=True)
            swallowed = True
        except asyncio.CancelledError:
            print(f"[cancel] client disconnected during {self.label}", flush=True)
            raise
        finally:
            await self._stop_pinger()
            if self.upstream is not None:
                await close_upstream(self.upstream)
                self.upstream = None
            if self.writer is not None:
                self.max_silence = max(self.max_silence, time.monotonic() - self.last_event_at)
            try:
                self._log_end()
            except Exception:
                pass
            if not self.abandoned:
                await self._write_eof()
            self._deactivate()
        if exc_type is asyncio.CancelledError:
            return False
        return swallowed

    async def _complete_terminal(self) -> None:
        if self.response is None:
            return
        try:
            self.terminal_event = await self.emitter.complete(
                self.response,
                upstream_saw_done=self.upstream_saw_done,
            )
        except ClientDisconnected:
            raise
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            print(
                f"[stream] {self.label} terminal complete failed: {type(exc).__name__}: {exc}",
                flush=True,
            )

    async def _fail_terminal(self, message: str, code: str) -> None:
        if self.response is None:
            return
        try:
            self.terminal_event = await self.emitter.fail(self.response, message, code=code)
        except ClientDisconnected:
            raise
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            print(
                f"[stream] {self.label} terminal fail failed: {type(exc).__name__}: {exc}",
                flush=True,
            )

    async def _stop_pinger(self) -> None:
        pinger = self._pinger
        self._pinger = None
        if pinger is None:
            return
        try:
            await pinger.stop()
        except Exception as exc:
            print(
                f"[stream] {self.label} pinger stop failed: {type(exc).__name__}: {exc}",
                flush=True,
            )

    def _deactivate(self) -> None:
        if self.writer is not None:
            self.writer.deactivate()

    async def _write_eof(self) -> None:
        if self.response is None:
            return
        try:
            await self.response.write_eof()
        except Exception:
            pass

    def _log_end(self) -> None:
        reason = self.finish_reason
        if callable(reason):
            try:
                reason = reason()
            except Exception:
                reason = None
        print(
            f"[stream-end] {self.label} "
            f"elapsed={time.monotonic() - self.started_at:.1f}s "
            f"upstream_events={self.upstream_events} saw_done={self.upstream_saw_done} "
            f"max_silence={self.max_silence:.1f}s pings={self.ping_count} "
            f"terminal={self.terminal_event or getattr(self.emitter, 'terminal_event', None) or 'NONE'} "
            f"finish_reason={reason if reason is not None else 'n/a'}",
            flush=True,
        )


# Re-export close_upstream so callers can `from .net.stream_guard import close_upstream`.
__all__ = ["StreamGuard", "TerminalEmitter", "ClientDisconnected", "close_upstream"]

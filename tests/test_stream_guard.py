from __future__ import annotations

import asyncio

import pytest

from codex_shim.net.emitters import ChatgptRelayEmitter, RawChatEmitter, WsRelayEmitter
from codex_shim.net.sse import ClientDisconnected, PING_BYTES
from codex_shim.net.stream_guard import StreamGuard


class _RecordingResponse:
    prepared = True

    def __init__(self) -> None:
        self.chunks: list[bytes] = []

    async def write(self, data: bytes):
        self.chunks.append(data)

    async def write_eof(self):
        self.chunks.append(b"EOF")


class _Emitter:
    def __init__(self) -> None:
        self.already_emitted = False
        self.terminal_event: str | None = None
        self.calls: list[tuple] = []

    async def complete(self, response, *, upstream_saw_done: bool) -> str:
        self.calls.append(("complete", upstream_saw_done))
        self.already_emitted = True
        self.terminal_event = "done"
        await response.write(b"data: [DONE]\n\n")
        return "done"

    async def fail(self, response, message: str, *, code: str) -> str:
        self.calls.append(("fail", message, code))
        self.already_emitted = True
        self.terminal_event = "failed"
        await response.write(f"fail:{code}".encode())
        return "failed"


async def test_stream_guard_completes_on_clean_exit():
    response = _RecordingResponse()
    emitter = _Emitter()
    async with StreamGuard(response, emitter, label="test-clean", keepalive=False) as guard:
        guard.note_upstream_event()
        guard.mark_upstream_done()
        await guard.write(b"data: hi\n\n")
    assert emitter.calls == [("complete", True)]
    assert b"data: hi\n\n" in response.chunks
    assert b"data: [DONE]\n\n" in response.chunks
    assert b"EOF" in response.chunks


async def test_stream_guard_fails_on_internal_error_and_swallows():
    response = _RecordingResponse()
    emitter = _Emitter()
    async with StreamGuard(response, emitter, label="test-boom", keepalive=False):
        raise RuntimeError("malformed upstream event")
    assert emitter.calls[0][0] == "fail"
    assert emitter.calls[0][2] == "shim_stream_error"
    assert any(chunk.startswith(b"fail:") for chunk in response.chunks)


async def test_stream_guard_abandon_skips_terminal():
    response = _RecordingResponse()
    emitter = _Emitter()
    async with StreamGuard(response, emitter, label="test-retry", keepalive=False) as guard:
        guard.abandon()
    assert emitter.calls == []
    assert b"EOF" not in response.chunks


async def test_stream_guard_swallows_client_disconnect():
    response = _RecordingResponse()
    emitter = _Emitter()
    async with StreamGuard(response, emitter, label="test-cancel", keepalive=False):
        raise ClientDisconnected()
    assert emitter.calls == []


async def test_stream_guard_pings_while_idle(monkeypatch):
    monkeypatch.setattr("codex_shim.net.sse.SSE_KEEPALIVE_INTERVAL", 0.05)
    response = _RecordingResponse()
    emitter = _Emitter()
    async with StreamGuard(response, emitter, label="test-ping", interval=0.05):
        await asyncio.sleep(0.18)
    assert PING_BYTES in response.chunks
    assert emitter.calls[0][0] == "complete"


@pytest.mark.parametrize(
    "factory,expected",
    [
        (RawChatEmitter, "done"),
        (lambda: ChatgptRelayEmitter(model="test"), "response.incomplete"),
    ],
)
async def test_emitters_synthesize_terminal_on_complete(factory, expected):
    response = _RecordingResponse()
    emitter = factory()
    async with StreamGuard(response, emitter, label="test-emitter", keepalive=False):
        pass
    joined = b"".join(response.chunks)
    if expected == "done":
        assert b"data: [DONE]" in joined
    else:
        assert b"response.incomplete" in joined
    assert emitter.terminal_event == expected


async def test_ws_relay_emitter_synthesizes_incomplete():
    sent: list[dict] = []

    async def write_event(event: dict) -> None:
        sent.append(event)

    emitter = WsRelayEmitter(write_event, model="test")
    terminal = await emitter.complete()
    assert terminal == "response.incomplete"
    assert sent[0]["type"] == "response.incomplete"
    assert sent[0]["response"]["status"] == "incomplete"

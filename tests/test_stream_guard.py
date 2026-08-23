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


class _BoomWrite:
    prepared = True

    async def write(self, data: bytes):
        raise RuntimeError("injected write failure")

    async def write_eof(self):
        pass


async def test_chatgpt_failed_then_disconnect_emits_one_failed_and_one_done():
    response = _RecordingResponse()
    emitter = ChatgptRelayEmitter(model="test")
    async with StreamGuard(response, emitter, label="test-dup-fail", keepalive=False) as guard:
        failed = {"type": "response.failed", "response": {"id": "r1", "status": "failed"}}
        emitter.observe(failed)
        await guard.write(b'data: {"type":"response.failed"}\n\n')
        await emitter.fail(response, "Upstream stream disconnected: x", code="upstream_disconnect")
        guard.mark_upstream_done()
    joined = b"".join(response.chunks)
    assert joined.count(b"response.failed") == 1
    assert joined.count(b"data: [DONE]") == 1


async def test_responses_state_fail_does_not_mark_terminal_before_failed_event_write():
    from codex_shim.server import ResponsesStreamState

    state = ResponsesStreamState("test-model")
    with pytest.raises(RuntimeError, match="injected write failure"):
        await state.fail(_BoomWrite(), "nope")
    assert state.failed is False
    assert state.terminal_emitted is False


async def test_anthropic_state_fail_does_not_mark_terminal_before_error_write():
    from codex_shim.server import AnthropicMessagesStreamState

    state = AnthropicMessagesStreamState("test-model")
    with pytest.raises(RuntimeError, match="injected write failure"):
        await state.fail(_BoomWrite(), "nope")
    assert state.failed is False
    assert state.terminal_emitted is False


async def test_pinger_failure_still_logs_end_writes_eof_and_deactivates_writer(capsys):
    from codex_shim.net.sse import _current_writer

    class BoomPinger:
        async def stop(self):
            raise RuntimeError("pinger boom")

    response = _RecordingResponse()
    emitter = _Emitter()
    async with StreamGuard(response, emitter, label="test-pinger-boom", keepalive=False) as guard:
        guard._pinger = BoomPinger()
    captured = capsys.readouterr()
    assert "[stream-end]" in captured.out
    assert "pinger stop failed" in captured.out
    assert b"EOF" in response.chunks
    assert _current_writer.get() is None


async def test_pinger_client_disconnect_cancels_owner_and_closes_upstream():
    class HangContent:
        async def readany(self):
            await asyncio.sleep(60)
            return b""

    class HangUpstream:
        def __init__(self) -> None:
            self.content = HangContent()
            self.closed = False

        def close(self):
            self.closed = True

        def release(self):
            pass

    class DisconnectPingResponse(_RecordingResponse):
        async def write(self, data: bytes):
            if data == PING_BYTES:
                raise ClientDisconnected()
            self.chunks.append(data)

    response = DisconnectPingResponse()
    emitter = _Emitter()
    upstream = HangUpstream()

    async def run():
        async with StreamGuard(
            response,
            emitter,
            label="test-ping-dc",
            interval=0.05,
            upstream=upstream,
        ) as guard:
            async for _ in guard.iter_sse():
                pass

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(run(), timeout=2.0)
    assert upstream.closed is True


async def test_abandon_immediately_deactivates_context_writer():
    from codex_shim.net.sse import _current_writer

    response = _RecordingResponse()
    emitter = _Emitter()
    async with StreamGuard(response, emitter, label="test-abandon-cv", keepalive=False) as guard:
        assert _current_writer.get() is guard.writer
        guard.abandon()
        assert _current_writer.get() is None
    assert emitter.calls == []
    assert _current_writer.get() is None


async def test_cancelled_error_propagates_after_cleanup(capsys):
    from codex_shim.net.sse import _current_writer

    response = _RecordingResponse()
    emitter = _Emitter()
    with pytest.raises(asyncio.CancelledError):
        async with StreamGuard(response, emitter, label="test-cancel-err", keepalive=False):
            raise asyncio.CancelledError()
    captured = capsys.readouterr()
    assert "[stream-end]" in captured.out
    assert b"EOF" in response.chunks
    assert _current_writer.get() is None
    assert emitter.calls == []


async def test_terminal_emitter_failure_is_logged_and_cleanup_still_runs(capsys):
    from codex_shim.net.sse import _current_writer

    class BoomEmitter(_Emitter):
        async def complete(self, response, *, upstream_saw_done: bool) -> str:
            raise RuntimeError("emitter boom")

    response = _RecordingResponse()
    emitter = BoomEmitter()
    async with StreamGuard(response, emitter, label="test-emitter-boom", keepalive=False):
        pass
    captured = capsys.readouterr()
    assert "terminal complete failed" in captured.out
    assert "[stream-end]" in captured.out
    assert b"EOF" in response.chunks
    assert _current_writer.get() is None


async def test_ping_does_not_disable_precontent_retry(monkeypatch):
    monkeypatch.setattr("codex_shim.net.sse.SSE_KEEPALIVE_INTERVAL", 0.05)
    response = _RecordingResponse()
    emitter = _Emitter()
    async with StreamGuard(response, emitter, label="test-ping-retry", interval=0.05) as guard:
        await asyncio.sleep(0.18)
        assert PING_BYTES in response.chunks
        assert guard.can_retry is True
        assert guard.writer is not None
        assert guard.writer.content_written is False
        guard.abandon()
    assert emitter.calls == []


async def test_stream_guard_closes_attached_upstream():
    class Upstream:
        def __init__(self) -> None:
            self.closed = False

        def close(self):
            self.closed = True

        def release(self):
            pass

    upstream = Upstream()
    response = _RecordingResponse()
    emitter = _Emitter()
    async with StreamGuard(
        response,
        emitter,
        label="test-up",
        keepalive=False,
        upstream=upstream,
    ):
        pass
    assert upstream.closed is True

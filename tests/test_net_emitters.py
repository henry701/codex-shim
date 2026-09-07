from __future__ import annotations

from types import SimpleNamespace

from codex_shim.net.emitters import (
    AnthropicMessagesEmitter,
    AnthropicRelayEmitter,
    ChatgptRelayEmitter,
    RawChatEmitter,
    ResponsesEmitter,
    WsRelayEmitter,
)


class _FakeStream:
    def __init__(self) -> None:
        self.chunks: list[bytes] = []

    async def write(self, data: bytes) -> None:
        self.chunks.append(data)


async def test_raw_chat_emitter_complete_and_fail_are_idempotent():
    stream = _FakeStream()
    emitter = RawChatEmitter()
    assert await emitter.complete(stream, upstream_saw_done=True) == "done"
    assert await emitter.complete(stream, upstream_saw_done=True) == "done"
    failed = RawChatEmitter()
    assert await failed.fail(stream, "boom", code="upstream_error") == "error"
    assert await failed.fail(stream, "boom", code="upstream_error") == "error"
    joined = b"".join(stream.chunks).decode()
    assert "data: [DONE]" in joined
    assert "upstream_error" in joined


async def test_chatgpt_relay_synthesizes_incomplete_then_skips_second_complete():
    stream = _FakeStream()
    emitter = ChatgptRelayEmitter(model="codex-gpt-5-6-luna")
    emitter.observe({"type": "response.output_text.delta", "response": {"id": "resp_1"}})
    assert await emitter.complete(stream, upstream_saw_done=True) == "response.incomplete"
    assert await emitter.complete(stream, upstream_saw_done=True) == "response.incomplete"
    text = b"".join(stream.chunks).decode()
    assert "response.incomplete" in text
    assert "data: [DONE]" in text

    terminal = ChatgptRelayEmitter(model="codex-gpt-5-6-luna")
    terminal.observe({"type": "response.completed", "response": {"id": "resp_2"}})
    assert await terminal.complete(stream, upstream_saw_done=True) == "response.completed"
    assert await terminal.fail(stream, "late", code="x") == "response.completed"


async def test_chatgpt_relay_fail_writes_failed_event():
    stream = _FakeStream()
    emitter = ChatgptRelayEmitter(model="codex-gpt-5-6-luna")
    assert await emitter.fail(stream, "quota", code="rate_limit") == "response.failed"
    text = b"".join(stream.chunks).decode()
    assert "response.failed" in text
    assert "quota" in text


async def test_anthropic_relay_and_messages_emitter():
    stream = _FakeStream()
    relay = AnthropicRelayEmitter(model="claude")
    assert await relay.complete(stream, upstream_saw_done=True) == "message_stop"
    relay.observe({"type": "message_stop"})
    assert await relay.complete(stream, upstream_saw_done=True) == "message_stop"
    assert await relay.fail(stream, "late", code="x") == "message_stop"

    failing = AnthropicRelayEmitter(model="claude")
    assert await failing.fail(stream, "boom", code="api_error") == "error"
    assert await failing.fail(stream, "boom", code="api_error") == "error"

    class State:
        def __init__(self) -> None:
            self.failed = False
            self.terminal_emitted = False
            self.calls: list[str] = []

        async def finish(self, response):
            del response
            self.terminal_emitted = True
            self.calls.append("finish")

        async def fail(self, response, message, *, code):
            del response, message, code
            self.failed = True
            self.calls.append("fail")

    state = State()
    wrapped = AnthropicMessagesEmitter(state)
    assert await wrapped.complete(stream, upstream_saw_done=True) == "message_stop"
    assert wrapped.already_emitted is True
    assert await wrapped.complete(stream, upstream_saw_done=True) == "message_stop"
    state2 = State()
    wrapped2 = AnthropicMessagesEmitter(state2)
    assert await wrapped2.fail(stream, "x", code="y") == "error"
    assert await wrapped2.fail(stream, "x", code="y") == "error"


async def test_responses_emitter_and_ws_relay():
    stream = _FakeStream()

    class State:
        def __init__(self) -> None:
            self.failed = False
            self.terminal_emitted = False
            self.terminal_event = None
            self.upstream_finish_reason = "stop"

        async def finish(self, response, *, upstream_saw_done):
            del response, upstream_saw_done
            self.terminal_emitted = True
            self.terminal_event = "response.completed"

        async def fail(self, response, message, *, code):
            del response, message, code
            self.failed = True
            self.terminal_event = "response.failed"

    state = State()
    emitter = ResponsesEmitter(state)
    assert emitter.finish_reason() == "stop"
    assert await emitter.complete(stream, upstream_saw_done=True) == "response.completed"
    assert await emitter.complete(stream, upstream_saw_done=True) == "response.completed"
    failed_state = State()
    failed = ResponsesEmitter(failed_state)
    assert await failed.fail(stream, "x", code="y") == "response.failed"
    assert await failed.fail(stream, "x", code="y") == "response.failed"

    events: list[dict] = []

    async def write_event(event):
        events.append(event)

    ws = WsRelayEmitter(write_event, model="codex-gpt-5-6-luna")
    assert await ws.complete() == "response.incomplete"
    assert await ws.complete() == "response.incomplete"
    fresh = WsRelayEmitter(write_event, model="codex-gpt-5-6-luna")
    assert await fresh.fail(None, "nope", code="unauthorized") == "response.failed"
    assert await fresh.fail(None, "nope", code="unauthorized") == "response.failed"
    assert events[0]["type"] == "response.incomplete"
    assert any(event.get("type") == "response.failed" for event in events)


def test_responses_emitter_terminal_passthrough():
    state = SimpleNamespace(failed=False, terminal_emitted=False, terminal_event="kept")
    assert ResponsesEmitter(state).terminal_event == "kept"

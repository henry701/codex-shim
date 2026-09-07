from __future__ import annotations

from types import SimpleNamespace

from aiohttp import web
from aiohttp.test_utils import make_mocked_request

from codex_shim.compaction.local import serialize_conversation, verbatim_user_quotes
from codex_shim.net.sse import (
    ClientDisconnected,
    close_upstream,
    keepalive_interval,
    request_disconnected,
)
from codex_shim.net.stream_guard import StreamGuard
from codex_shim.server import ResponsesStreamState, ShimServer, _web_search_stream_item


class _FakeStream:
    def __init__(self) -> None:
        self.chunks: list[bytes] = []

    async def write(self, data: bytes) -> None:
        self.chunks.append(data)


def _settings(tmp_path):
    path = tmp_path / "models.json"
    path.write_text("{}")
    return path


async def test_write_anthropic_remaining_event_types():
    downstream = _FakeStream()
    state = ResponsesStreamState("claude-real")
    await state.write_anthropic_delta(
        downstream,
        {"type": "message_start", "message": {"usage": {"input_tokens": 3, "output_tokens": 1}}},
    )
    await state.write_anthropic_delta(
        downstream,
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "thinking", "thinking": "plan", "signature": "sig0"},
        },
    )
    await state.write_anthropic_delta(
        downstream,
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {"type": "redacted_thinking", "data": "hidden"},
        },
    )
    await state.write_anthropic_delta(
        downstream,
        {
            "type": "content_block_start",
            "index": 2,
            "content_block": {"type": "tool_use", "id": "toolu_2", "name": "lookup"},
        },
    )
    await state.write_anthropic_delta(
        downstream,
        {"type": "content_block_delta", "index": 2, "delta": {"type": "input_json_delta", "partial_json": ""}},
    )
    await state.write_anthropic_delta(
        downstream,
        {
            "type": "message_delta",
            "usage": {"output_tokens": 9, "input_tokens": 3},
        },
    )
    await state.write_anthropic_delta(
        downstream,
        {"type": "message_delta", "usage": {"output_tokens": 11}},
    )
    await state.write_anthropic_delta(downstream, {"type": "content_block_stop", "index": 2})
    await state.write_anthropic_delta(downstream, {"type": "content_block_stop", "index": 0})
    assert state.usage is not None
    assert state.usage.get("output_tokens") == 11


def test_serialize_conversation_and_verbatim_quotes():
    text = serialize_conversation(
        [
            "skip",
            {"type": "compaction", "encrypted_content": "old"},
            {"type": "message", "role": "user", "content": "please inspect src/app.py carefully now"},
            {"type": "function_call", "name": "shell", "call_id": "c1", "arguments": {"cmd": "ls"}},
            {"type": "function_call_output", "call_id": "c1", "output": "ok" * 2000},
            {"type": "unknown", "text": "leftover"},
        ]
    )
    assert "[user]:" in text or "[previous-summary]" in text or "shell" in text
    assert verbatim_user_quotes([]) == ""
    quotes = verbatim_user_quotes(
        [
            {"type": "message", "role": "user", "content": "first user prompt here is long enough"},
            {"type": "message", "role": "user", "content": "second user prompt is also long enough"},
        ],
        max_prompts=1,
    )
    assert "user-messages-verbatim" in quotes


async def test_sse_keepalive_disconnect_and_close(monkeypatch):
    assert keepalive_interval(0.01) == 0.05
    monkeypatch.setenv("CODEX_SHIM_SSE_KEEPALIVE_INTERVAL", "2.5")
    assert keepalive_interval() == 2.5
    monkeypatch.setenv("CODEX_SHIM_SSE_KEEPALIVE_INTERVAL", "nope")
    monkeypatch.setenv("CODEX_SHIM_KEEPALIVE_MIN", "5")
    monkeypatch.setenv("CODEX_SHIM_KEEPALIVE_MAX", "3")
    value = keepalive_interval()
    assert value >= 0.05
    monkeypatch.delenv("CODEX_SHIM_SSE_KEEPALIVE_INTERVAL")
    monkeypatch.setenv("CODEX_SHIM_KEEPALIVE_MIN", "nope")
    monkeypatch.setenv("CODEX_SHIM_KEEPALIVE_MAX", "6")

    assert request_disconnected(None) is False
    assert request_disconnected(SimpleNamespace(transport=None, protocol=None)) is False
    assert request_disconnected(SimpleNamespace(transport=SimpleNamespace(is_closing=lambda: True), protocol=None)) is True
    assert request_disconnected(
        SimpleNamespace(
            transport=SimpleNamespace(is_closing=lambda: False),
            protocol=SimpleNamespace(transport=SimpleNamespace(is_closing=lambda: True)),
        )
    ) is True

    class Boom:
        def close(self):
            raise RuntimeError("close")

        def release(self):
            raise RuntimeError("release")

    await close_upstream(None)
    await close_upstream(Boom())

    item = _web_search_stream_item({"id": "i", "call_id": "c", "arguments": "{nope"}, "completed")
    assert item["action"]["query"] == "{nope"


async def test_stream_guard_swallows_client_disconnect():
    class Emitter:
        already_emitted = False

        async def complete(self, response, *, upstream_saw_done):
            del response, upstream_saw_done
            return "complete"

        async def fail(self, response, message, *, code):
            del response, message, code
            return "fail"

    stream = _FakeStream()
    async with StreamGuard(stream, Emitter(), label="test", keepalive=False) as guard:
        raise ClientDisconnected()
    assert guard.terminal_event is None or guard.abandoned is False


async def test_responses_compaction_v2_error_and_success(monkeypatch, tmp_path):
    shim = ShimServer(_settings(tmp_path))
    request = make_mocked_request("POST", "/v1/responses")

    async def resolve_err(*a, **k):
        return {}, "local", None, web.Response(text='{"error":{"message":"nope"}}', status=400)

    monkeypatch.setattr(ShimServer, "_resolve_compaction_v2_result", resolve_err)
    error = await shim._responses_compaction_v2(request, {"model": "local", "input": []}, [])
    assert error.status == 400

    async def resolve_ok(*a, **k):
        return ({"id": "cmp", "encrypted_content": "x"}, "local", {"input_tokens": 1}, None)

    async def fake_stream(*a, **k):
        return web.json_response({"ok": True})

    monkeypatch.setattr("codex_shim.server._stream_compaction_v2_sse", fake_stream)
    monkeypatch.setattr(ShimServer, "_resolve_compaction_v2_result", resolve_ok)
    ok = await shim._responses_compaction_v2(request, {"model": "local", "input": []}, [])
    assert ok.status == 200


async def test_resolve_ws_passthrough_target_missing_auth(monkeypatch, tmp_path):
    auth = tmp_path / "auth.json"
    monkeypatch.setattr("codex_shim.server.DEFAULT_CODEX_AUTH", auth)
    monkeypatch.setattr("codex_shim.server.is_chatgpt_passthrough_slug", lambda model: True)
    shim = ShimServer(_settings(tmp_path))
    assert await shim._resolve_ws_passthrough_target({"model": "codex-gpt-5-6-luna"}) is None
    auth.write_text("{")
    assert await shim._resolve_ws_passthrough_target({"model": "codex-gpt-5-6-luna"}) is None
    auth.write_text('{"tokens":{}}')
    assert await shim._resolve_ws_passthrough_target({"model": "codex-gpt-5-6-luna"}) is None
    auth.write_text('{"tokens":{"access_token":"tok","account_id":"acct"}}')
    target = await shim._resolve_ws_passthrough_target({"model": "codex-gpt-5-6-luna"})
    assert target is not None
    assert target.kind == "chatgpt"

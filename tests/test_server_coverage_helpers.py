from __future__ import annotations

import json
from pathlib import Path

from aiohttp import web

from codex_shim.compaction import encode_shim_compaction_summary
from codex_shim.compaction.input_audit import summarize_compaction_input_item_types
from codex_shim.server import (
    ResponsesStreamState,
    ShimServer,
    _encode_thinking_payload,
    _join_url,
    _log_stream_event,
    _summarize_compaction_input_items,
    _write_ws_error,
)
from codex_shim.translate import _decode_thinking_blob
from codex_shim import server as server_module


def _settings(tmp_path: Path) -> Path:
    settings = tmp_path / "models.json"
    settings.write_text("{}")
    return settings


class _FakeStream:
    def __init__(self) -> None:
        self.chunks: list[bytes] = []

    async def write(self, data: bytes) -> None:
        self.chunks.append(data)


class _FakeWs:
    def __init__(self) -> None:
        self.sent: list[str] = []

    async def send_str(self, payload: str) -> None:
        self.sent.append(payload)


def _sse_events(text: str) -> list[dict]:
    events = []
    for block in text.split("\n\n"):
        if not block.startswith("data:"):
            continue
        data = block.removeprefix("data:").strip()
        if data and data != "[DONE]":
            events.append(json.loads(data))
    return events


def test_summarize_compaction_input_items_labels_tool_and_message_rows():
    assert summarize_compaction_input_item_types("not-a-list") == []
    assert _summarize_compaction_input_items is summarize_compaction_input_item_types
    assert _summarize_compaction_input_items("not-a-list") == []
    assert _summarize_compaction_input_items([1, "x"]) == ["?", "?"]
    labeled = _summarize_compaction_input_items(
        [
            {"type": "function_call", "name": "shell"},
            {"type": "function_call_output", "call_id": "call_abcdefghijklmnopqrstuvwxyz"},
            {"type": "message", "role": "user"},
            {"role": "assistant"},
        ]
    )
    assert labeled[0] == "function_call name='shell'"
    assert labeled[1].startswith("function_call_output call_id='call_abcdefghijklmnop")
    assert labeled[2] == "message role='user'"
    assert labeled[3] == "assistant"


def test_log_stream_event_includes_namespaced_and_search_details(monkeypatch, capsys):
    monkeypatch.delenv("CODEX_SHIM_STREAM_LOG", raising=False)
    _log_stream_event({"type": "response.output_item.added", "item": {"type": "function_call", "name": "shell"}})
    assert capsys.readouterr().out == ""

    monkeypatch.setenv("CODEX_SHIM_STREAM_LOG", "1")
    _log_stream_event(
        {
            "type": "response.output_item.added",
            "item": {"type": "function_call", "namespace": "mcp__exa"},
        }
    )
    _log_stream_event(
        {
            "type": "response.output_item.done",
            "item": {"type": "web_search_call", "action": {"query": "ukraine"}},
        }
    )
    _log_stream_event({"type": "response.reasoning_summary_text.delta", "delta": "abcd"})
    out = capsys.readouterr().out
    assert "mcp__exa/" in out
    assert "query=ukraine" in out or "name=ukraine" in out
    assert "len=4" in out


def test_summarization_result_from_chatgpt_response_handles_bad_payloads(tmp_path):
    server = ShimServer(_settings(tmp_path))
    compact_body = {"model": "codex-gpt-5-6-luna"}

    stream = web.StreamResponse()
    failed = server._summarization_result_from_chatgpt_response(compact_body, stream)
    assert failed.error_response is stream
    assert failed.summary == ""

    bad_json = web.Response(text="{not-json", status=200)
    assert server._summarization_result_from_chatgpt_response(compact_body, bad_json).error_response is bad_json

    not_object = web.Response(text="[]", status=200)
    assert server._summarization_result_from_chatgpt_response(compact_body, not_object).error_response is not_object

    empty = web.Response(text=json.dumps({"output": []}), status=200)
    assert server._summarization_result_from_chatgpt_response(compact_body, empty).error_response is empty

    ok = web.Response(
        text=json.dumps(
            {
                "output": [{"type": "message", "content": [{"type": "output_text", "text": "keep going"}]}],
                "usage": {
                    "input_tokens": 9,
                    "output_tokens": 2,
                    "input_tokens_details": {"cached_tokens": 4},
                },
            }
        ),
        status=200,
    )
    result = server._summarization_result_from_chatgpt_response(compact_body, ok)
    assert result.summary == "keep going"
    assert result.usage["input_tokens_details"]["cached_tokens"] == 4

    encoded = encode_shim_compaction_summary("blob summary")
    compacted = web.Response(
        text=json.dumps({"output": [{"type": "compaction", "encrypted_content": encoded}]}),
        status=200,
    )
    assert server._summarization_result_from_chatgpt_response(compact_body, compacted).summary == "blob summary"


def test_chatgpt_compaction_lock_is_reused(tmp_path):
    server = ShimServer(_settings(tmp_path))
    first = server._chatgpt_compaction_lock("sess-a")
    second = server._chatgpt_compaction_lock("sess-a")
    third = server._chatgpt_compaction_lock("sess-b")
    assert first is second
    assert third is not first


async def test_compaction_acquire_chatgpt_lock_uses_session_key(tmp_path):
    server = ShimServer(_settings(tmp_path))
    request = type("Req", (), {"session_key": "thread-9"})()
    lock = await server._compaction_acquire_chatgpt_lock(request)
    assert lock is server._chatgpt_compaction_lock("thread-9")


async def test_anthropic_content_block_deltas_emit_text_json_and_thinking():
    downstream = _FakeStream()
    state = ResponsesStreamState("claude-real")
    await state.write_anthropic_delta(
        downstream,
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": "Hi"}},
    )
    await state.write_anthropic_delta(
        downstream,
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {"type": "tool_use", "id": "toolu_1", "name": "lookup"},
        },
    )
    await state.write_anthropic_delta(
        downstream,
        {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "input_json_delta", "partial_json": '{"q":'},
        },
    )
    await state.write_anthropic_delta(
        downstream,
        {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "input_json_delta", "partial_json": '"repo"}'},
        },
    )
    await state.write_anthropic_delta(
        downstream,
        {
            "type": "content_block_delta",
            "index": 2,
            "delta": {"type": "thinking_delta", "thinking": "plan "},
        },
    )
    await state.write_anthropic_delta(
        downstream,
        {
            "type": "content_block_delta",
            "index": 2,
            "delta": {"type": "signature_delta", "signature": "sig"},
        },
    )
    await state.write_anthropic_delta(
        downstream,
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": " there"}},
    )
    events = _sse_events(b"".join(downstream.chunks).decode())
    types = [event.get("type") for event in events]
    assert "response.output_text.delta" in types
    assert "response.function_call_arguments.delta" in types
    assert "response.reasoning_summary_text.delta" in types
    thinking = state.reasoning_blocks[("anthropic_thinking", 2)]
    assert thinking["text"] == "plan "
    assert thinking["signature"] == "sig"


def test_stream_state_reset_for_next_turn_clears_tool_maps():
    state = ResponsesStreamState("local-llama")
    state.tool_calls[1] = {"name": "shell"}
    state.message_text = "hello"
    state.snapshot_turn()
    state.reset_for_next_turn()
    assert state.tool_calls == {}
    assert state.mcp_tool_calls == {}
    assert state.message_text == ""
    assert len(state.completed_turns) == 1


def test_thinking_payload_roundtrips_through_translate_decoder():
    encoded = _encode_thinking_payload({"type": "thinking", "thinking": "plan", "signature": "sig"})
    assert encoded.startswith("anthropic-thinking-v1:")
    assert _decode_thinking_blob(encoded)["thinking"] == "plan"
    assert _decode_thinking_blob("nope") is None
    assert _decode_thinking_blob("anthropic-thinking-v1:%%%") is None
    assert _decode_thinking_blob("anthropic-thinking-v1:W10=") is None


def test_join_url_keeps_versioned_bases():
    assert _join_url("https://api.example/v1", "/chat/completions") == "https://api.example/v1/chat/completions"
    assert _join_url("https://api.example", "/messages") == "https://api.example/v1/messages"
    assert _join_url("https://api.example", "/chat/completions").endswith("/v1/chat/completions")


async def test_handle_response_create_websocket_chatgpt_auth_errors(tmp_path, monkeypatch):
    server = ShimServer(_settings(tmp_path))
    monkeypatch.setattr(server_module, "ws_passthrough_enabled", lambda: False)

    async def no_target(self, payload):
        del payload
        return None

    monkeypatch.setattr(ShimServer, "_resolve_ws_passthrough_target", no_target)
    monkeypatch.setattr(server_module, "is_chatgpt_passthrough_slug", lambda slug: True)

    missing = tmp_path / "missing-auth.json"
    monkeypatch.setattr(server_module, "DEFAULT_CODEX_AUTH", missing)
    ws = _FakeWs()
    await server._handle_response_create_websocket(object(), ws, {"model": "codex-gpt-5-6-luna"})
    assert "auth.json not found" in ws.sent[0]

    invalid = tmp_path / "bad-auth.json"
    invalid.write_text("{")
    monkeypatch.setattr(server_module, "DEFAULT_CODEX_AUTH", invalid)
    ws = _FakeWs()
    await server._handle_response_create_websocket(object(), ws, {"model": "codex-gpt-5-6-luna"})
    assert "not valid JSON" in ws.sent[0]

    empty = tmp_path / "empty-auth.json"
    empty.write_text(json.dumps({"tokens": {}}))
    monkeypatch.setattr(server_module, "DEFAULT_CODEX_AUTH", empty)
    ws = _FakeWs()
    await server._handle_response_create_websocket(object(), ws, {"model": "codex-gpt-5-6-luna"})
    assert "no access_token" in ws.sent[0]


async def test_write_ws_error_payload_shape():
    ws = _FakeWs()
    await _write_ws_error(ws, 401, "unauthorized", "nope")
    payload = json.loads(ws.sent[0])
    assert payload["type"] == "error"
    assert payload["status"] == 401
    assert payload["error"]["message"] == "nope"

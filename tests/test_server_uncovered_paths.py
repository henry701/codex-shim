from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer, make_mocked_request

from codex_shim.compaction.pipeline import PreparedInput
from codex_shim.compaction.types import CompactionRequest
from codex_shim.mcp_search import CODEX_TOOL_SEARCH_NAME, MCP_TOOL_SEARCH_NAME
from codex_shim.server import (
    ResponsesStreamState,
    ShimServer,
    _anthropic_error_response,
    _log_chatgpt_passthrough_trace,
    _log_upstream_io_detail,
    _stream_responses_error_from_body,
    _summarize_input_items,
)
from codex_shim.settings import ShimModel


@pytest.fixture
def auth_present(monkeypatch, tmp_path):
    auth = tmp_path / "auth.json"
    auth.write_text(json.dumps({"tokens": {"access_token": "stub", "account_id": "acct"}}))
    monkeypatch.setattr("codex_shim.settings.DEFAULT_CODEX_AUTH", auth)
    monkeypatch.setattr("codex_shim.server.DEFAULT_CODEX_AUTH", auth)
    return auth


class _FakeSseContent:
    def __init__(self, chunks: list[bytes]):
        self._chunks = list(chunks)

    async def readany(self):
        await asyncio.sleep(0)
        if not self._chunks:
            return b""
        return self._chunks.pop(0)


class _FakeStream:
    def __init__(self) -> None:
        self.chunks: list[bytes] = []

    async def write(self, data: bytes) -> None:
        self.chunks.append(data)


def _sse_chunk(payload: dict[str, Any]) -> bytes:
    return f"data: {json.dumps(payload)}\n\n".encode()


def _settings(tmp_path: Path) -> Path:
    settings = tmp_path / "models.json"
    settings.write_text("{}")
    return settings


def _shim(tmp_path: Path) -> ShimServer:
    return ShimServer(_settings(tmp_path))


def _anthropic_route() -> ShimModel:
    return ShimModel(
        slug="claude-local",
        model="claude-3",
        display_name="Claude",
        provider="anthropic",
        base_url="http://example.invalid/v1",
        api_key="secret",
    )


def _openai_chat_route() -> ShimModel:
    return ShimModel(
        slug="local-llama",
        model="llama",
        display_name="Llama",
        provider="openai",
        base_url="http://example.invalid/v1",
        api_key="secret",
    )


def _openai_responses_route() -> ShimModel:
    return ShimModel(
        slug="console-model",
        model="gpt-x",
        display_name="Console",
        provider="openai-responses",
        base_url="http://example.invalid/v1",
        api_key="secret",
    )


def _prepared() -> PreparedInput:
    return PreparedInput(
        native_input=[{"type": "message", "role": "user", "content": "hi"}],
        summarization_input=[{"type": "message", "role": "assistant", "content": "ok"}],
        previous_summary="prior",
    )


def test_summarize_input_items_covers_tool_and_message_shapes():
    assert _summarize_input_items("nope") == (0, [])
    count, labels = _summarize_input_items(
        [
            "bare",
            {"type": "function_call", "name": "shell"},
            {"type": "function_call_output", "call_id": "call_abcdefghijklmnopqrstuvwxyz"},
            {"type": "web_search_call", "action": {"query": "find the repo readme please"}},
            {"type": "mcp_tool_call", "server": "exa", "tool": "search"},
            {"type": "message", "role": "user"},
            {"type": "agent_message", "author": "a", "recipient": "b"},
            {"type": "agent_message"},
            {"role": "assistant"},
        ],
        tail=20,
    )
    assert count == 9
    assert labels[0] == "?"
    assert labels[1] == "function_call(shell)"
    assert "call_id=" in labels[2]
    assert "query=" in labels[3]
    assert labels[4] == "mcp_tool_call(exa/search)"
    assert labels[5] == "message(role=user)"
    assert "from=a" in labels[6]
    assert labels[7] == "agent_message(self)"
    assert labels[8] == "assistant"


def test_log_passthrough_trace_and_io_detail(monkeypatch, capsys):
    monkeypatch.delenv("CODEX_SHIM_PASSTHROUGH_TRACE", raising=False)
    monkeypatch.delenv("CODEX_SHIM_REQUEST_LOG", raising=False)
    monkeypatch.delenv("CODEX_SHIM_STREAM_LOG", raising=False)
    request = make_mocked_request(
        "POST",
        "/v1/responses",
        headers={"Authorization": "Bearer secret", "session_id": "s1"},
    )
    _log_chatgpt_passthrough_trace(request, {"input": []}, {"Authorization": "x"}, phase="pre")
    _log_upstream_io_detail(surface="byok-compact", phase="pre-request", url="local:openai")
    assert capsys.readouterr().out == ""

    monkeypatch.setenv("CODEX_SHIM_PASSTHROUGH_TRACE", "1")
    _log_chatgpt_passthrough_trace(
        request,
        {
            "input": [{"type": "message", "role": "user"}],
            "previous_response_id": "resp_1",
            "reasoning": {"effort": "medium"},
        },
        {"Authorization": "Bearer secret", "Accept": "application/json"},
        phase="pre-upstream",
    )
    _log_upstream_io_detail(
        surface="byok-compact",
        phase="pre-request",
        url="claude:anthropic",
        forwarded={"model": "claude", "stream": False, "input": [{"type": "message"}], "instructions": "x"},
        status=200,
        response_text="ok" * 2000,
    )
    out = capsys.readouterr().out
    assert "[chatgpt-trace]" in out
    assert "<redacted>" in out
    assert "[io]" in out
    assert "byok-compact" in out


async def test_anthropic_error_response_maps_status_type_and_request_id():
    class Upstream:
        status = 401
        headers = {"request-id": "req_9", "x-request-id": "ignored"}

        async def text(self):
            return json.dumps({"error": {"message": "bad key", "type": "ignored_type"}})

        def release(self):
            pass

    response = await _anthropic_error_response(Upstream())
    assert response.status == 401
    payload = json.loads(response.text)
    assert payload["error"]["type"] == "authentication_error"
    assert payload["error"]["message"] == "bad key"
    assert payload["request_id"] == "req_9"


async def test_anthropic_error_response_uses_top_level_message_when_error_missing():
    class Upstream:
        status = 418
        headers = {}

        async def text(self):
            return json.dumps({"message": "teapot"})

        def release(self):
            pass

    response = await _anthropic_error_response(Upstream())
    payload = json.loads(response.text)
    assert payload["error"]["message"] == "teapot"
    assert payload["error"]["type"] == "api_error"


async def test_stream_responses_error_from_body_emits_failed_event():
    async def handler(request):
        return await _stream_responses_error_from_body(
            request,
            "local-llama",
            502,
            '{"error":{"message":"boom"}}',
            slug="local-llama",
        )

    app = web.Application()
    app.router.add_post("/err", handler)
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        resp = await client.post("/err")
        assert resp.status == 200
        text = await resp.text()
        assert "response.failed" in text
        assert "boom" in text
    finally:
        await client.close()


async def test_chatgpt_passthrough_collect_stream_returns_completed_json(
    monkeypatch, tmp_path, auth_present
):
    class FakeUpstream:
        status = 200
        content_type = "text/event-stream"
        headers = {"Content-Type": "text/event-stream"}
        content = _FakeSseContent(
            [
                _sse_chunk({"type": "response.created", "response": {"id": "resp_s"}}),
                b"data: not-json\n\n",
                _sse_chunk(
                    {
                        "type": "response.completed",
                        "response": {
                            "id": "resp_s",
                            "output": [
                                {
                                    "type": "message",
                                    "role": "assistant",
                                    "content": [{"type": "output_text", "text": "compacted history"}],
                                }
                            ],
                            "usage": {"input_tokens": 3, "output_tokens": 4},
                        },
                    }
                ),
                b"data: [DONE]\n\n",
            ]
        )

        def release(self):
            pass

    async def fake_post(self, url, json=None, headers=None):
        del self, url, json, headers
        return FakeUpstream()

    monkeypatch.setattr("codex_shim.server.ClientSession.post", fake_post)
    shim = _shim(tmp_path)
    request = make_mocked_request("POST", "/v1/responses", headers={"session-id": "s1"})
    response = await shim._chatgpt_passthrough(
        request,
        {"model": "codex-gpt-5-5", "input": "hi"},
        collect_stream=True,
        response_model_override="codex-gpt-5-5",
        upstream_model="gpt-5.5",
    )
    assert isinstance(response, web.Response)
    assert response.status == 200
    payload = json.loads(response.text)
    assert payload["id"] == "resp_s"
    assert payload["output"][0]["content"][0]["text"] == "compacted history"


async def test_chatgpt_passthrough_collect_stream_failed_event_returns_502(
    monkeypatch, tmp_path, auth_present
):
    class FakeUpstream:
        status = 200
        content_type = "text/event-stream"
        headers = {"Content-Type": "text/event-stream"}
        content = _FakeSseContent(
            [
                _sse_chunk(
                    {
                        "type": "response.failed",
                        "response": {"error": {"message": "quota exceeded"}},
                    }
                ),
                b"data: [DONE]\n\n",
            ]
        )

        def release(self):
            pass

    async def fake_post(self, url, json=None, headers=None):
        del self, url, json, headers
        return FakeUpstream()

    monkeypatch.setattr("codex_shim.server.ClientSession.post", fake_post)
    shim = _shim(tmp_path)
    request = make_mocked_request("POST", "/v1/responses", headers={"session-id": "s1"})
    response = await shim._chatgpt_passthrough(
        request,
        {"model": "codex-gpt-5-5", "input": "hi"},
        collect_stream=True,
        upstream_model="gpt-5.5",
    )
    assert response.status == 502
    assert "quota exceeded" in response.text


async def test_compaction_native_chatgpt_json_error_paths(monkeypatch, tmp_path, auth_present):
    shim = _shim(tmp_path)
    request = CompactionRequest(
        http_request=make_mocked_request("POST", "/v1/responses/compact", headers={"session-id": "s1"}),
        body={"model": "codex-gpt-5-5", "input": [{"type": "message", "role": "user", "content": "hi"}]},
        stripped_input=[{"type": "message", "role": "user", "content": "hi"}],
        requested_slug="codex-gpt-5-5",
        provider="chatgpt",
        session_key="s1",
    )

    async def bad_json(self, *args, **kwargs):
        del self, args, kwargs
        return web.Response(text="{not-json", status=200)

    monkeypatch.setattr(ShimServer, "_post_chatgpt_native_compact", bad_json)
    result = await shim._compaction_native_chatgpt(request, _prepared())
    assert result.item is None
    assert result.error_response.status == 200

    async def not_dict(self, *args, **kwargs):
        del self, args, kwargs
        return web.Response(text="[]", status=200)

    monkeypatch.setattr(ShimServer, "_post_chatgpt_native_compact", not_dict)
    result = await shim._compaction_native_chatgpt(request, _prepared())
    assert result.item is None

    async def empty_payload(self, *args, **kwargs):
        del self, args, kwargs
        return web.json_response({"output": [], "usage": {"input_tokens": 1, "output_tokens": 0}})

    monkeypatch.setattr(ShimServer, "_post_chatgpt_native_compact", empty_payload)
    result = await shim._compaction_native_chatgpt(request, _prepared())
    assert result.item is not None
    assert result.usage == {"input_tokens": 1, "output_tokens": 0}


async def test_compaction_native_and_summarization_cursor_parse_errors(monkeypatch, tmp_path):
    shim = _shim(tmp_path)
    request = CompactionRequest(
        http_request=make_mocked_request("POST", "/v1/responses/compact", headers={"session-id": "s1"}),
        body={"model": "cursor-composer-2-5", "input": [{"type": "message", "role": "user", "content": "hi"}]},
        stripped_input=[{"type": "message", "role": "user", "content": "hi"}],
        requested_slug="cursor-composer-2-5",
        provider="cursor",
        session_key="s1",
    )

    async def native_bad(self, *args, **kwargs):
        del self, args, kwargs
        return web.Response(text="{nope", status=200)

    monkeypatch.setattr(ShimServer, "_cursor_passthrough", native_bad)
    native = await shim._compaction_native_cursor(request, _prepared())
    assert native.item is None

    async def native_list(self, *args, **kwargs):
        del self, args, kwargs
        return web.Response(text="[]", status=200)

    monkeypatch.setattr(ShimServer, "_cursor_passthrough", native_list)
    native = await shim._compaction_native_cursor(request, _prepared())
    assert native.item is None

    async def native_http_error(self, *args, **kwargs):
        del self, args, kwargs
        return web.Response(text='{"error":{"message":"cursor down"}}', status=502)

    monkeypatch.setattr(ShimServer, "_cursor_passthrough", native_http_error)
    native = await shim._compaction_native_cursor(request, _prepared())
    assert native.native_status == 502
    assert "cursor down" in native.native_message

    async def sum_bad(self, *args, **kwargs):
        del self, args, kwargs
        return web.Response(text="{nope", status=200)

    monkeypatch.setattr(ShimServer, "_cursor_passthrough", sum_bad)
    summarized = await shim._compaction_summarization_cursor(request, _prepared(), "native failed")
    assert summarized.summary == ""
    assert summarized.error_response.status == 200

    async def sum_empty(self, *args, **kwargs):
        del self, args, kwargs
        return web.json_response({"output": []})

    monkeypatch.setattr(ShimServer, "_cursor_passthrough", sum_empty)
    summarized = await shim._compaction_summarization_cursor(request, _prepared(), "native failed")
    assert summarized.summary == ""

    async def sum_ok(self, *args, **kwargs):
        del self, args, kwargs
        return web.json_response(
            {
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "cursor summary"}],
                    }
                ],
                "usage": {"input_tokens": 2, "output_tokens": 2},
            }
        )

    monkeypatch.setattr(ShimServer, "_cursor_passthrough", sum_ok)
    summarized = await shim._compaction_summarization_cursor(request, _prepared(), "native failed")
    assert summarized.summary == "cursor summary"
    assert summarized.usage["output_tokens"] == 2


async def test_fetch_byok_compact_summary_provider_branches(monkeypatch, tmp_path):
    shim = _shim(tmp_path)
    monkeypatch.setattr(ShimServer, "_apply_responses_input_pipeline", lambda self, body, request, **kwargs: body)
    request = make_mocked_request("POST", "/v1/responses/compact", headers={"session-id": "s1"})
    compact_body = {"model": "x", "input": []}

    async def json_error(self, *args, **kwargs):
        del self, args, kwargs
        return web.Response(text="{nope", status=200)

    async def list_payload(self, *args, **kwargs):
        del self, args, kwargs
        return web.Response(text="[]", status=200)

    async def http_error(self, *args, **kwargs):
        del self, args, kwargs
        return web.Response(text="fail", status=502)

    async def ok_payload(self, *args, **kwargs):
        del self, args, kwargs
        return web.json_response(
            {
                "output": [
                    {"type": "message", "content": [{"type": "output_text", "text": "byok summary"}]}
                ],
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }
        )

    monkeypatch.setattr(ShimServer, "_post_openai_responses", json_error)
    summary, usage, err = await shim._fetch_byok_compact_summary(
        request, _openai_responses_route(), compact_body, tool_types={}, tool_resolve={}
    )
    assert summary == ""
    assert err is not None

    monkeypatch.setattr(ShimServer, "_post_openai_responses", list_payload)
    summary, usage, err = await shim._fetch_byok_compact_summary(
        request, _openai_responses_route(), compact_body, tool_types={}, tool_resolve={}
    )
    assert err is not None

    monkeypatch.setattr(ShimServer, "_post_openai_responses", http_error)
    summary, usage, err = await shim._fetch_byok_compact_summary(
        request, _openai_responses_route(), compact_body, tool_types={}, tool_resolve={}
    )
    assert err.status == 502

    monkeypatch.setattr(ShimServer, "_post_openai_responses", ok_payload)
    summary, usage, err = await shim._fetch_byok_compact_summary(
        request, _openai_responses_route(), compact_body, tool_types={}, tool_resolve={}
    )
    assert summary == "byok summary"
    assert err is None
    assert usage == {"input_tokens": 1, "output_tokens": 1}

    monkeypatch.setattr(ShimServer, "_post_openai_chat", json_error)
    summary, usage, err = await shim._fetch_byok_compact_summary(
        request, _openai_chat_route(), compact_body, tool_types={}, tool_resolve={}
    )
    assert err is not None

    monkeypatch.setattr(ShimServer, "_post_openai_chat", ok_payload)
    summary, usage, err = await shim._fetch_byok_compact_summary(
        request, _openai_chat_route(), compact_body, tool_types={}, tool_resolve={}
    )
    assert summary == "byok summary"

    monkeypatch.setattr(ShimServer, "_post_anthropic", http_error)
    summary, usage, err = await shim._fetch_byok_compact_summary(
        request, _anthropic_route(), compact_body, tool_types={}, tool_resolve={}
    )
    assert err.status == 502

    monkeypatch.setattr(ShimServer, "_post_anthropic", json_error)
    summary, usage, err = await shim._fetch_byok_compact_summary(
        request, _anthropic_route(), compact_body, tool_types={}, tool_resolve={}
    )
    assert err is not None

    monkeypatch.setattr(ShimServer, "_post_anthropic", list_payload)
    summary, usage, err = await shim._fetch_byok_compact_summary(
        request, _anthropic_route(), compact_body, tool_types={}, tool_resolve={}
    )
    assert err is not None

    monkeypatch.setattr(ShimServer, "_post_anthropic", ok_payload)
    summary, usage, err = await shim._fetch_byok_compact_summary(
        request, _anthropic_route(), compact_body, tool_types={}, tool_resolve={}
    )
    assert summary == "byok summary"
    assert usage["output_tokens"] == 1

    mystery = ShimModel(
        slug="mystery",
        model="x",
        display_name="X",
        provider="mystery",
        base_url="http://example.invalid",
        api_key="k",
    )
    summary, usage, err = await shim._fetch_byok_compact_summary(
        request, mystery, compact_body, tool_types={}, tool_resolve={}
    )
    assert summary == ""
    assert err.status == 502


async def test_finalize_pending_tool_search_mcp_and_namespaced():
    stream = _FakeStream()
    state = ResponsesStreamState("local-llama", tool_resolve={"ns.shell": ("ns", "shell")})
    await state.start(stream)
    await state._open_message(stream)

    state.tool_calls[0] = {"name": CODEX_TOOL_SEARCH_NAME, "call_id": "c0", "id": "fc_0", "arguments": "{}"}
    await state._finalize_pending_tool(stream, 0)
    assert 0 in state.tool_search_calls
    assert 0 not in state.tool_calls

    state.tool_calls[1] = {
        "name": "mcp__exa.web_search",
        "call_id": "c1",
        "id": "fc_1",
        "arguments": "{}",
    }
    await state._finalize_pending_tool(stream, 1)
    assert 1 in state.mcp_tool_calls

    state.tool_calls[2] = {"name": "mcp_stub", "call_id": "c2", "id": "fc_2", "arguments": "{}"}
    await state._finalize_pending_tool(stream, 2)
    assert state.tool_calls[2].get("emitted") is not True

    state.tool_calls[3] = {
        "name": "ns.shell",
        "call_id": "c3",
        "id": "fc_3",
        "arguments": "{}",
    }
    await state._finalize_pending_tool(stream, 3)
    assert state.tool_calls[3]["emitted"] is True
    assert state.tool_calls[3]["namespace"] == "ns"
    assert state.tool_calls[3]["name"] == "shell"
    joined = b"".join(stream.chunks).decode()
    assert "response.output_item.added" in joined
    assert "function_call" in joined

    await state._finalize_pending_tool(stream, 99)
    state.tool_calls[4] = {"name": MCP_TOOL_SEARCH_NAME, "emitted": True}
    await state._finalize_pending_tool(stream, 4)


async def test_cursor_bridge_delivery_stream_and_json(tmp_path):
    shim = _shim(tmp_path)
    app = web.Application()

    async def stream_handler(request):
        return await shim._cursor_bridge_delivery_response(
            request,
            {
                "model": "cursor-composer-2-5",
                "stream": True,
                "input": [{"type": "message", "role": "user", "content": "hi"}],
            },
            "cursor-composer-2-5",
            message="leftover assistant text",
        )

    async def json_handler(request):
        return await shim._cursor_bridge_delivery_response(
            request,
            {
                "model": "cursor-composer-2-5",
                "stream": False,
                "input": [{"type": "message", "role": "user", "content": "hi"}],
            },
            "cursor-composer-2-5",
            message="leftover assistant text",
        )

    app.router.add_post("/stream", stream_handler)
    app.router.add_post("/json", json_handler)
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        streamed = await client.post("/stream", headers={"session-id": "s1"})
        assert streamed.status == 200
        text = await streamed.text()
        assert "leftover assistant text" in text
        assert "response.completed" in text or "response.incomplete" in text

        bundled = await client.post("/json", headers={"session-id": "s1"})
        assert bundled.status == 200
        payload = await bundled.json()
        assert payload["output"][0]["content"][0]["text"] == "leftover assistant text"
    finally:
        await client.close()


async def test_anthropic_responses_stream_text_and_error_event(tmp_path):
    async def messages(request):
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        await response.write(b"data: not-json\n\n")
        await response.write(
            b'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hello"}}\n\n'
        )
        await response.write(b'data: {"error":{"message":"upstream boom"}}\n\n')
        await response.write(b"data: [DONE]\n\n")
        await response.write_eof()
        return response

    upstream = web.Application()
    upstream.router.add_post("/v1/messages", messages)
    upstream_client = TestClient(TestServer(upstream))
    await upstream_client.start_server()
    try:
        settings = tmp_path / "settings.json"
        settings.write_text(
            json.dumps(
                {
                    "customModels": [
                        {
                            "model": "claude-upstream",
                            "displayName": "Claude Upstream",
                            "provider": "anthropic",
                            "baseUrl": str(upstream_client.make_url("")),
                            "apiKey": "secret",
                        }
                    ]
                }
            )
        )
        shim_client = TestClient(TestServer(ShimServer(settings).app()))
        await shim_client.start_server()
        try:
            resp = await shim_client.post(
                "/v1/responses",
                json={"model": "claude-upstream", "input": "hi", "stream": True},
            )
            assert resp.status == 200
            text = await resp.text()
            assert "hello" in text
            assert "upstream boom" in text or "response.failed" in text
        finally:
            await shim_client.close()
    finally:
        await upstream_client.close()

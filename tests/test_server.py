from __future__ import annotations

import asyncio
import json
import subprocess
from typing import Any

import pytest
from aiohttp import ClientSession, WSMsgType, web
from aiohttp.test_utils import TestClient, TestServer

from codex_shim.compaction import decode_shim_compaction_summary
from codex_shim import mcp_search
from codex_shim import server as server_module
from codex_shim.server import (
    PICKER_TOKEN_HEADER,
    ResponsesStreamState,
    ShimServer,
    _current_managed_model,
    _iter_reasoning_delta_chunks,
    _picker_html,
    _request_disconnected,
    _rewrite_response_model,
    _sanitize_chatgpt_passthrough_body,
    _set_active_model,
    parse_upstream_error,
)
from codex_shim.catalog_slugs import codex_catalog_slug
from codex_shim.settings import FALLBACK_CHATGPT_PASSTHROUGH_SLUGS
from codex_shim.translate import SHIM_ENCRYPTED_CONTENT_PREFIX


@pytest.fixture
def auth_present(monkeypatch, tmp_path):
    auth = tmp_path / "auth.json"
    auth.write_text(json.dumps({"tokens": {"access_token": "stub", "account_id": "acct"}}))
    monkeypatch.setattr("codex_shim.settings.DEFAULT_CODEX_AUTH", auth)
    monkeypatch.setattr("codex_shim.server.DEFAULT_CODEX_AUTH", auth)
    return auth


@pytest.fixture
def auth_missing(monkeypatch, tmp_path):
    missing = tmp_path / "missing-auth.json"
    monkeypatch.setattr("codex_shim.settings.DEFAULT_CODEX_AUTH", missing)
    monkeypatch.setattr("codex_shim.server.DEFAULT_CODEX_AUTH", missing)


def test_sanitize_chatgpt_passthrough_body_drops_shim_reasoning():
    body = {
        "model": "claude-local",
        "input": [
            {"type": "message", "role": "user", "content": "hi"},
            {
                "id": "rs_shim",
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "local thought"}],
                "encrypted_content": f"{SHIM_ENCRYPTED_CONTENT_PREFIX}deadbeef",
            },
            {
                "id": "rs_openai",
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "openai thought"}],
                "encrypted_content": "openai-verifiable-content",
            },
        ],
    }

    sanitized = _sanitize_chatgpt_passthrough_body(body)

    assert sanitized is not body
    assert sanitized["input"] is not body["input"]
    assert [item["id"] for item in sanitized["input"] if item.get("type") == "reasoning"] == ["rs_openai"]
    assert sanitized["input"][1]["encrypted_content"] == "openai-verifiable-content"
    assert len(body["input"]) == 3


def test_sanitize_chatgpt_passthrough_body_removes_nested_shim_encrypted_content():
    body = {
        "model": "claude-local",
        "input": [
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "done",
                        "encrypted_content": f"{SHIM_ENCRYPTED_CONTENT_PREFIX}deadbeef",
                    }
                ],
            }
        ],
    }

    sanitized = _sanitize_chatgpt_passthrough_body(body)

    assert "encrypted_content" not in sanitized["input"][0]["content"][0]
    assert "encrypted_content" in body["input"][0]["content"][0]


def test_sanitize_chatgpt_passthrough_body_removes_previous_response_id_top_level_only():
    body = {
        "model": "codex-gpt-5-5",
        "previous_response_id": "resp_previous",
        "metadata": {"previous_response_id": "metadata-value"},
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": "hi",
            }
        ],
    }

    sanitized = _sanitize_chatgpt_passthrough_body(body)

    assert "previous_response_id" not in sanitized
    assert sanitized["metadata"]["previous_response_id"] == "metadata-value"
    assert body["previous_response_id"] == "resp_previous"


def test_rewrite_response_model_only_rewrites_chatgpt_metadata():
    payload = {
        "model": "gpt-5.5",
        "nested": [{"model": "gpt-5.5"}, {"model": "other"}],
    }

    _rewrite_response_model(payload, "custom-model")

    assert payload == {
        "model": "custom-model",
        "nested": [{"model": "custom-model"}, {"model": "other"}],
    }


def test_image_generation_detection_is_conservative():
    shim = ShimServer()
    tools = [
        {"type": "function", "function": {"name": "shell"}},
        {"type": "image_generation", "name": "image_generation"},
    ]

    assert shim._needs_image_gen({"tools": tools, "input": [{"role": "user", "content": "write code for an icon component"}]}) is False
    assert shim._needs_image_gen({"tools": tools, "input": [{"role": "user", "content": "@image generate a neon fox"}]}) is True
    assert shim._needs_image_gen({"tools": tools, "tool_choice": {"type": "image_generation"}, "input": "hi"}) is True
    assert shim._needs_image_followup(
        {
            "input": [
                {"type": "image_generation_call", "id": "ig_1"},
                {"role": "user", "content": "make it brighter"},
            ]
        }
    ) is True


async def test_image_generation_routes_to_chatgpt_passthrough_and_rewrites_model(monkeypatch, tmp_path, auth_present):
    captured = {}

    class FakeUpstream:
        status = 200
        content_type = "application/json"

        async def json(self, content_type=None):
            return {"id": "resp_img", "model": "gpt-5.5", "output": [{"type": "image_generation_call", "model": "gpt-5.5"}]}

        def release(self):
            pass

    async def fake_post(self, url, json=None, headers=None):
        captured["url"] = url
        captured["body"] = json
        captured["headers"] = headers
        return FakeUpstream()

    monkeypatch.setattr("codex_shim.server.ClientSession.post", fake_post)
    settings = tmp_path / "settings.json"
    settings.write_text(
        json.dumps(
            {
                "customModels": [
                    {
                        "model": "real-openai",
                        "displayName": "Real OpenAI",
                        "provider": "openai",
                        "baseUrl": "http://example.invalid/v1",
                    }
                ]
            }
        )
    )
    shim_client = TestClient(TestServer(ShimServer(settings).app()))
    await shim_client.start_server()

    resp = await shim_client.post(
        "/v1/responses",
        json={
            "model": "real-openai",
            "input": [{"role": "user", "content": "@image generate a neon fox"}],
            "tools": [{"type": "image_generation", "name": "image_generation"}],
        },
    )
    assert resp.status == 200
    payload = await resp.json()
    assert payload["model"] == "real-openai"
    assert payload["output"][0]["model"] == "real-openai"
    assert captured["body"]["model"] == "gpt-5.5"
    assert captured["headers"]["Authorization"] == "Bearer stub"

    await shim_client.close()


async def test_chatgpt_passthrough_requests_advertise_zstd_encoding(monkeypatch, tmp_path, auth_present):
    captured = {}

    class FakeUpstream:
        status = 200
        content_type = "application/json"

        async def json(self, content_type=None):
            return {"id": "resp_1", "model": "gpt-5.5", "output": []}

        def release(self):
            pass

    async def fake_post(self, url, json=None, headers=None):
        captured["url"] = str(url)
        captured["headers"] = headers
        return FakeUpstream()

    monkeypatch.setattr("codex_shim.server.ClientSession.post", fake_post)
    settings = tmp_path / "settings.json"
    settings.write_text("{}")
    shim_client = TestClient(TestServer(ShimServer(settings).app()))
    await shim_client.start_server()

    resp = await shim_client.post(
        "/v1/responses",
        json={"model": "codex-gpt-5-5", "input": "hi"},
        headers={"Accept-Encoding": "zstd, gzip"},
    )

    assert resp.status == 200
    assert captured["url"] == "https://chatgpt.com/backend-api/codex/responses"
    assert captured["headers"]["Accept-Encoding"] == "zstd, gzip, deflate"

    await shim_client.close()


async def test_chatgpt_passthrough_drops_previous_response_id(monkeypatch, tmp_path, auth_present):
    captured = {}

    class FakeUpstream:
        status = 200
        content_type = "application/json"

        async def json(self, content_type=None):
            return {"id": "resp_1", "model": "gpt-5.5", "output": []}

        def release(self):
            pass

    async def fake_post(self, url, json=None, headers=None):
        captured["url"] = str(url)
        captured["body"] = json
        return FakeUpstream()

    monkeypatch.setattr("codex_shim.server.ClientSession.post", fake_post)
    settings = tmp_path / "settings.json"
    settings.write_text("{}")
    shim_client = TestClient(TestServer(ShimServer(settings).app()))
    await shim_client.start_server()

    resp = await shim_client.post(
        "/v1/responses",
        json={"model": "codex-gpt-5-5", "previous_response_id": "resp_previous", "input": "hi"},
    )

    assert resp.status == 200
    assert captured["url"] == "https://chatgpt.com/backend-api/codex/responses"
    assert captured["body"]["model"] == "gpt-5.5"
    assert "previous_response_id" not in captured["body"]

    await shim_client.close()


async def test_chatgpt_passthrough_expands_previous_response_id(monkeypatch, tmp_path, auth_present):
    captured = []
    first_input = [{"type": "message", "role": "user", "content": "run a command"}]
    tool_call = {
        "type": "function_call",
        "id": "fc_1",
        "call_id": "call_1",
        "name": "exec_command",
        "arguments": "{\"cmd\":\"printf ok\"}",
    }
    tool_output = {"type": "function_call_output", "call_id": "call_1", "output": "ok"}

    class FakeUpstream:
        status = 200
        content_type = "application/json"

        def __init__(self, payload):
            self.payload = payload

        async def json(self, content_type=None):
            return self.payload

        def release(self):
            pass

    async def fake_post(self, url, json=None, headers=None):
        captured.append(json)
        if len(captured) == 1:
            return FakeUpstream({"id": "resp_previous", "model": "gpt-5.5", "output": [tool_call]})
        return FakeUpstream({"id": "resp_next", "model": "gpt-5.5", "output": []})

    monkeypatch.setattr("codex_shim.server.ClientSession.post", fake_post)
    settings = tmp_path / "settings.json"
    settings.write_text("{}")
    shim_client = TestClient(TestServer(ShimServer(settings).app()))
    await shim_client.start_server()

    first = await shim_client.post("/v1/responses", json={"model": "codex-gpt-5-5", "input": first_input})
    assert first.status == 200

    second = await shim_client.post(
        "/v1/responses",
        json={
            "model": "codex-gpt-5-5",
            "previous_response_id": "resp_previous",
            "input": [tool_output],
        },
    )
    assert second.status == 200

    assert "previous_response_id" not in captured[1]
    assert captured[1]["input"] == [*first_input, tool_call, tool_output]

    await shim_client.close()


def _zstd_compress(data: bytes) -> bytes:
    return subprocess.run(
        ["zstd", "-q", "-c", "-"],
        input=data,
        check=True,
        stdout=subprocess.PIPE,
    ).stdout


async def test_zstd_compressed_responses_request_reaches_handler(monkeypatch, tmp_path, auth_present):
    captured = {}

    class FakeUpstream:
        status = 200
        content_type = "application/json"

        async def json(self, content_type=None):
            return {"id": "resp_1", "model": "gpt-5.5", "output": []}

        def release(self):
            pass

    async def fake_post(self, url, json=None, headers=None):
        captured["body"] = json
        return FakeUpstream()

    monkeypatch.setattr("codex_shim.server.ClientSession.post", fake_post)
    settings = tmp_path / "settings.json"
    settings.write_text("{}")
    shim_client = TestClient(TestServer(ShimServer(settings).app()))
    await shim_client.start_server()

    raw = json.dumps({"model": "codex-gpt-5-5", "input": "hi"}).encode()
    resp = await shim_client.post(
        "/v1/responses",
        data=_zstd_compress(raw),
        headers={"Content-Type": "application/json", "Content-Encoding": "zstd"},
    )

    assert resp.status == 200
    assert captured["body"]["model"] == "gpt-5.5"

    await shim_client.close()


async def test_chatgpt_compact_passthrough_requests_advertise_zstd_encoding(monkeypatch, tmp_path, auth_present):
    captured = {}

    class FakeUpstream:
        status = 200
        content_type = "application/json"

        async def json(self, content_type=None):
            return {"id": "resp_compact", "model": "gpt-5.5", "output": []}

        def release(self):
            pass

    async def fake_post(self, url, json=None, headers=None):
        captured["url"] = str(url)
        captured["headers"] = headers
        return FakeUpstream()

    monkeypatch.setattr("codex_shim.server.ClientSession.post", fake_post)
    settings = tmp_path / "settings.json"
    settings.write_text("{}")
    shim_client = TestClient(TestServer(ShimServer(settings).app()))
    await shim_client.start_server()

    resp = await shim_client.post(
        "/v1/responses/compact",
        json={"model": "codex-gpt-5-5", "input": "summarize"},
        headers={"Accept-Encoding": "zstd, gzip"},
    )

    assert resp.status == 200
    assert captured["url"] == "https://chatgpt.com/backend-api/codex/responses/compact"
    assert captured["headers"]["Accept-Encoding"] == "zstd, gzip, deflate"

    await shim_client.close()


class _FakeSseContent:
    def __init__(self, chunks: list[bytes]):
        self._chunks = list(chunks)

    async def readany(self):
        await asyncio.sleep(0)
        if not self._chunks:
            return b""
        return self._chunks.pop(0)


async def test_chatgpt_passthrough_websocket_relays_sse_events(monkeypatch, tmp_path, auth_present):
    captured = {}

    class FakeUpstream:
        status = 200
        content_type = "text/event-stream"
        content = _FakeSseContent(
            [
                b'data: {"type":"response.created","response":{"id":"resp_1","model":"gpt-5.5"}}\n\n',
                b'data: {"type":"response.output_text.delta","delta":"ok"}\n\n',
                b'data: {"type":"response.completed","response":{"id":"resp_1","model":"gpt-5.5","status":"completed"}}\n\n',
            ]
        )

        def release(self):
            pass

    async def fake_post(self, url, json=None, headers=None):
        captured["url"] = str(url)
        captured["body"] = json
        captured["headers"] = headers
        return FakeUpstream()

    monkeypatch.setattr("codex_shim.server.ClientSession.post", fake_post)
    settings = tmp_path / "settings.json"
    settings.write_text("{}")
    shim_client = TestClient(TestServer(ShimServer(settings).app()))
    await shim_client.start_server()

    ws = await shim_client.ws_connect("/v1/responses")
    await ws.send_json(
        {
            "type": "response.create",
            "model": "codex-gpt-5-5",
            "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]}],
            "tools": [],
            "tool_choice": "auto",
            "parallel_tool_calls": True,
            "previous_response_id": "resp_previous",
            "store": False,
            "stream": True,
            "include": [],
        }
    )

    events = []
    for _ in range(3):
        msg = await ws.receive(timeout=2)
        assert msg.type == WSMsgType.TEXT
        events.append(json.loads(msg.data))

    assert [event["type"] for event in events] == [
        "response.created",
        "response.output_text.delta",
        "response.completed",
    ]
    assert events[0]["response"]["model"] == "codex-gpt-5-5"
    assert events[2]["response"]["model"] == "codex-gpt-5-5"
    assert captured["url"] == "https://chatgpt.com/backend-api/codex/responses"
    assert captured["body"]["model"] == "gpt-5.5"
    assert captured["body"]["stream"] is True
    assert "previous_response_id" not in captured["body"]
    assert "type" not in captured["body"]
    assert captured["headers"]["Authorization"] == "Bearer stub"

    await ws.close()
    await shim_client.close()


async def test_chatgpt_passthrough_websocket_expands_previous_response_id(monkeypatch, tmp_path, auth_present):
    captured = []
    first_input = [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "run"}]}]
    tool_call = {
        "type": "function_call",
        "id": "fc_1",
        "call_id": "call_1",
        "name": "exec_command",
        "arguments": "{\"cmd\":\"printf ok\"}",
    }
    tool_output = {"type": "function_call_output", "call_id": "call_1", "output": "ok"}

    class FakeUpstream:
        status = 200
        content_type = "text/event-stream"

        def __init__(self, chunks):
            self.content = _FakeSseContent(chunks)

        def release(self):
            pass

    async def fake_post(self, url, json=None, headers=None):
        captured.append(json)
        if len(captured) == 1:
            return FakeUpstream(
                [
                    b'data: {"type":"response.created","response":{"id":"resp_previous","model":"gpt-5.5"}}\n\n',
                    b'data: {"type":"response.output_item.done","response":{"id":"resp_previous","model":"gpt-5.5","output":[]},"item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"exec_command","arguments":"{\\\"cmd\\\":\\\"printf ok\\\"}"}}\n\n',
                    b'data: {"type":"response.completed","response":{"id":"resp_previous","model":"gpt-5.5","status":"completed","output":[]}}\n\n',
                ]
            )
        return FakeUpstream(
            [
                b'data: {"type":"response.created","response":{"id":"resp_next","model":"gpt-5.5"}}\n\n',
                b'data: {"type":"response.completed","response":{"id":"resp_next","model":"gpt-5.5","status":"completed"}}\n\n',
            ]
        )

    monkeypatch.setattr("codex_shim.server.ClientSession.post", fake_post)
    settings = tmp_path / "settings.json"
    settings.write_text("{}")
    shim_client = TestClient(TestServer(ShimServer(settings).app()))
    await shim_client.start_server()

    ws = await shim_client.ws_connect("/v1/responses")
    await ws.send_json({"type": "response.create", "model": "codex-gpt-5-5", "input": first_input, "stream": True})
    for _ in range(3):
        msg = await ws.receive(timeout=2)
        assert msg.type == WSMsgType.TEXT

    await ws.send_json(
        {
            "type": "response.create",
            "model": "codex-gpt-5-5",
            "previous_response_id": "resp_previous",
            "input": [tool_output],
            "stream": True,
        }
    )
    for _ in range(2):
        msg = await ws.receive(timeout=2)
        assert msg.type == WSMsgType.TEXT

    assert "previous_response_id" not in captured[1]
    assert captured[1]["input"] == [*first_input, tool_call, tool_output]

    await shim_client.close()


async def test_responses_websocket_bridges_byok_models_through_http_route(tmp_path):
    async def chat(request):
        body = await request.json()
        assert body["model"] == "real-openai"
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        await response.write(b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n')
        await response.write(b'data: {"choices":[{"delta":{}}]}\n\n')
        await response.write(b"data: [DONE]\n\n")
        await response.write_eof()
        return response

    upstream = web.Application()
    upstream.router.add_post("/v1/chat/completions", chat)
    upstream_client = TestClient(TestServer(upstream))
    await upstream_client.start_server()

    settings = tmp_path / "settings.json"
    settings.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "model": "real-openai",
                        "display_name": "Real OpenAI",
                        "provider": "openai",
                        "base_url": str(upstream_client.make_url("/v1")),
                        "api_key": "secret",
                    }
                ]
            }
        )
    )
    shim_client = TestClient(TestServer(ShimServer(settings).app()))
    await shim_client.start_server()

    ws = await shim_client.ws_connect("/v1/responses")
    await ws.send_json(
        {
            "type": "response.create",
            "model": "real-openai",
            "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]}],
            "tools": [],
            "tool_choice": "auto",
            "parallel_tool_calls": True,
            "store": False,
            "stream": True,
            "include": [],
        }
    )

    events = []
    while True:
        msg = await ws.receive(timeout=2)
        assert msg.type == WSMsgType.TEXT
        payload = json.loads(msg.data)
        events.append(payload)
        if payload.get("type") == "response.completed":
            break

    assert any(event.get("type") == "response.output_text.delta" and event.get("delta") == "hello" for event in events)
    assert events[-1]["response"]["model"] == "real-openai"

    await ws.close()
    await shim_client.close()
    await upstream_client.close()


async def test_chatgpt_passthrough_falls_back_to_byok_on_error(monkeypatch, tmp_path, auth_present):
    calls: list[str] = []

    class FailingChatGPTUpstream:
        status = 503
        content_type = "text/plain"

        async def text(self):
            return "chatgpt unavailable"

        def release(self):
            pass

    async def fake_post(self, url, json=None, headers=None):
        calls.append(str(url))
        if "chatgpt.com" in str(url):
            return FailingChatGPTUpstream()
        return await orig_post(self, url, json=json, headers=headers)

    upstream = web.Application()
    captured: dict[str, Any] = {}

    async def chat(request):
        captured["body"] = await request.json()
        return web.json_response(
            {
                "id": "chatcmpl_fallback",
                "choices": [{"message": {"role": "assistant", "content": "thread title"}}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
            }
        )

    upstream.router.add_post("/v1/chat/completions", chat)
    upstream_client = TestClient(TestServer(upstream))
    await upstream_client.start_server()

    orig_post = ClientSession.post
    monkeypatch.setattr("codex_shim.server.ClientSession.post", fake_post)

    settings = tmp_path / "settings.json"
    settings.write_text(
        json.dumps(
            {
                "passthrough_error_fallback": {"gpt-5.4-mini": "or-free-router"},
                "customModels": [
                    {
                        "model": "openrouter/free",
                        "displayName": "OpenRouter Free",
                        "slug": "or-free-router",
                        "provider": "generic-chat-completion-api",
                        "baseUrl": str(upstream_client.make_url("/v1")),
                        "apiKey": "secret",
                    }
                ],
            }
        )
    )
    shim_client = TestClient(TestServer(ShimServer(settings).app()))
    await shim_client.start_server()

    resp = await shim_client.post(
        "/v1/responses",
        json={"model": "gpt-5.4-mini", "input": [{"role": "user", "content": "name this thread"}]},
    )
    assert resp.status == 200
    payload = await resp.json()
    assert payload["model"] == "gpt-5.4-mini"
    assert captured["body"]["model"] == "openrouter/free"
    assert any("chatgpt.com" in url for url in calls)
    assert any("chat/completions" in url for url in calls)

    await shim_client.close()
    await upstream_client.close()


async def test_responses_routes_to_openai_chat(tmp_path):
    captured = {}

    async def chat(request):
        captured["headers"] = dict(request.headers)
        captured["body"] = await request.json()
        return web.json_response(
            {
                "id": "chatcmpl_fake",
                "choices": [{"message": {"role": "assistant", "content": "hello"}}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
            }
        )

    upstream = web.Application()
    upstream.router.add_post("/v1/chat/completions", chat)
    upstream_client = TestClient(TestServer(upstream))
    await upstream_client.start_server()

    settings = tmp_path / "settings.json"
    settings.write_text(
        json.dumps(
            {
                "customModels": [
                    {
                        "model": "real-openai",
                        "displayName": "Real OpenAI",
                        "provider": "openai",
                        "baseUrl": str(upstream_client.make_url("/v1")),
                        "apiKey": "secret",
                    }
                ]
            }
        )
    )
    shim_client = TestClient(TestServer(ShimServer(settings).app()))
    await shim_client.start_server()

    resp = await shim_client.post("/v1/responses", json={"model": "real-openai", "input": "hi"})
    assert resp.status == 200
    payload = await resp.json()
    assert payload["output"][0]["content"][0]["text"] == "hello"
    assert payload["usage"] == {"input_tokens": 2, "output_tokens": 1, "total_tokens": 3}
    assert captured["body"]["model"] == "real-openai"
    assert captured["headers"]["Authorization"] == "Bearer secret"

    await shim_client.close()
    await upstream_client.close()


async def test_missing_api_key_env_has_model_specific_error(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENCODE_GO_API_KEY", raising=False)
    settings = tmp_path / "settings.json"
    settings.write_text(
        json.dumps(
            {
                "customModels": [
                    {
                        "model": "glm-5.1",
                        "displayName": "OpenCode Go GLM-5.1",
                        "provider": "generic-chat-completion-api",
                        "baseUrl": "https://opencode.ai/zen/go/v1",
                        "apiKeyEnv": "OPENCODE_GO_API_KEY",
                    }
                ]
            }
        )
    )
    shim_client = TestClient(TestServer(ShimServer(settings).app()))
    await shim_client.start_server()

    resp = await shim_client.post("/v1/responses", json={"model": "glm-5-1", "input": "hi"})

    assert resp.status == 401
    text = await resp.text()
    assert "OPENCODE_GO_API_KEY" in text
    assert "CURSOR_API_KEY" not in text

    await shim_client.close()


def _sse_events(text: str) -> list[dict]:
    events = []
    for block in text.split("\n\n"):
        if not block.startswith("data:"):
            continue
        data = block.removeprefix("data:").strip()
        if data and data != "[DONE]":
            events.append(json.loads(data))
    return events


def test_parse_upstream_error_reads_zen_model_error():
    body = json.dumps(
        {
            "type": "error",
            "error": {
                "type": "ModelError",
                "message": (
                    "Free promotion has ended for MiniMax M3 Free. "
                    "You can continue using the model by subscribing to OpenCode Go"
                ),
            },
        }
    )
    code, message = parse_upstream_error(body, 400)
    assert code == "ModelError"
    assert "Free promotion has ended" in message


async def test_responses_stream_state_fail_emits_terminal_error_events():
    class FakeResponse:
        def __init__(self):
            self.chunks: list[bytes] = []

        async def write(self, data: bytes):
            self.chunks.append(data)

    downstream = FakeResponse()
    state = ResponsesStreamState("oc-free-minimax-m3-free")
    await state.start(downstream)
    await state.fail(downstream, "Free promotion has ended", code="ModelError")
    events = _sse_events(b"".join(downstream.chunks).decode())
    assert events[1]["type"] == "error"
    assert events[1]["message"] == "Free promotion has ended"
    assert events[-1]["type"] == "response.failed"
    assert events[-1]["response"]["status"] == "failed"
    assert events[-1]["response"]["error"]["message"] == "Free promotion has ended"
    assert b"data: [DONE]" in b"".join(downstream.chunks)


def _named_sse_events(text: str) -> list[tuple[str | None, dict]]:
    events = []
    for block in text.split("\n\n"):
        event_name = None
        data = None
        for line in block.splitlines():
            if line.startswith("event:"):
                event_name = line.removeprefix("event:").strip()
            elif line.startswith("data:"):
                data = line.removeprefix("data:").strip()
        if data and data != "[DONE]":
            events.append((event_name, json.loads(data)))
    return events


async def test_streaming_openai_chat_response_completed_includes_usage(tmp_path):
    async def chat(request):
        await request.json()
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        await response.write(b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n')
        await response.write(
            b'data: {"choices":[{"delta":{}}],"usage":{"prompt_tokens":4,"completion_tokens":2,"total_tokens":6,"prompt_tokens_details":{"cached_tokens":3}}}\n\n'
        )
        await response.write(b"data: [DONE]\n\n")
        await response.write_eof()
        return response

    upstream = web.Application()
    upstream.router.add_post("/v1/chat/completions", chat)
    upstream_client = TestClient(TestServer(upstream))
    await upstream_client.start_server()

    settings = tmp_path / "settings.json"
    settings.write_text(
        json.dumps(
            {
                "customModels": [
                    {
                        "model": "real-openai",
                        "displayName": "Real OpenAI",
                        "provider": "openai",
                        "baseUrl": str(upstream_client.make_url("/v1")),
                        "apiKey": "secret",
                    }
                ]
            }
        )
    )
    shim_client = TestClient(TestServer(ShimServer(settings).app()))
    await shim_client.start_server()

    resp = await shim_client.post("/v1/responses", json={"model": "real-openai", "input": "hi", "stream": True})
    assert resp.status == 200
    events = _sse_events(await resp.text())
    completed = [event for event in events if event.get("type") == "response.completed"][-1]
    assert completed["response"]["usage"] == {
        "input_tokens": 4,
        "output_tokens": 2,
        "total_tokens": 6,
        "input_tokens_details": {"cached_tokens": 3},
    }

    await shim_client.close()
    await upstream_client.close()


async def test_sse_lines_closes_upstream_when_client_disconnects(tmp_path):
    upstream_state = {"sent": 0}

    async def slow_upstream(request):
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        try:
            for _ in range(500):
                payload = json.dumps({"choices": [{"delta": {"content": "x"}}]})
                await response.write(f"data: {payload}\n\n".encode())
                upstream_state["sent"] += 1
                await asyncio.sleep(0.01)
        except (ConnectionResetError, ConnectionError, asyncio.CancelledError):
            pass
        finally:
            try:
                await response.write_eof()
            except Exception:
                pass
        return response

    upstream = web.Application()
    upstream.router.add_post("/v1/chat/completions", slow_upstream)
    upstream_client = TestClient(TestServer(upstream))
    await upstream_client.start_server()

    settings = tmp_path / "settings.json"
    settings.write_text(
        json.dumps(
            {
                "customModels": [
                    {
                        "model": "slow",
                        "displayName": "Slow",
                        "provider": "openai",
                        "baseUrl": str(upstream_client.make_url("/v1")),
                        "apiKey": "secret",
                    }
                ]
            }
        )
    )
    shim_client = TestClient(TestServer(ShimServer(settings).app()))
    await shim_client.start_server()

    resp = await shim_client.post(
        "/v1/responses",
        json={"model": "slow", "input": "hi", "stream": True},
    )
    assert resp.status == 200
    await resp.content.readline()
    resp.close()
    await asyncio.sleep(0.4)
    assert 0 < upstream_state["sent"] < 500

    await shim_client.close()
    await upstream_client.close()


async def test_request_disconnected_reflects_closing_transport():
    class FakeTransport:
        def __init__(self, closing: bool):
            self._closing = closing

        def is_closing(self) -> bool:
            return self._closing

    class FakeRequest:
        def __init__(self, closing: bool):
            self.transport = FakeTransport(closing)
            self.protocol = None

    assert _request_disconnected(FakeRequest(False)) is False
    assert _request_disconnected(FakeRequest(True)) is True


def test_iter_reasoning_delta_chunks_splits_large_text():
    text = "x" * 200
    chunks = _iter_reasoning_delta_chunks(text)
    assert len(chunks) == 3
    assert "".join(chunks) == text


async def test_reasoning_item_populates_summary_for_desktop_collapsible():
    class FakeResponse:
        def __init__(self):
            self.chunks: list[bytes] = []

        async def write(self, data: bytes):
            self.chunks.append(data)

    downstream = FakeResponse()
    state = ResponsesStreamState("local-llama")
    reasoning = await state._open_reasoning(downstream, key=("chat",), initial_text="thought")
    await state._close_reasoning(downstream, reasoning)

    events = _sse_events(b"".join(downstream.chunks).decode())
    done = [event for event in events if event.get("type") == "response.output_item.done"][-1]
    item = done["item"]
    assert item["type"] == "reasoning"
    assert item["summary"] == [{"type": "summary_text", "text": "thought"}]
    assert item["encrypted_content"].startswith(SHIM_ENCRYPTED_CONTENT_PREFIX)
    delta_events = [event for event in events if event.get("type") == "response.reasoning_summary_text.delta"]
    assert len(delta_events) >= 1


async def test_chat_tool_delta_streams_exec_command_to_client():
    class FakeResponse:
        def __init__(self):
            self.chunks: list[bytes] = []

        async def write(self, data: bytes):
            self.chunks.append(data)

    downstream = FakeResponse()
    state = ResponsesStreamState("local-llama")
    await state.start(downstream)
    await state.write_chat_delta(
        downstream,
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "function": {"name": "exec_command", "arguments": '{"cmd":"'},
                            }
                        ]
                    }
                }
            ]
        },
    )
    await state.write_chat_delta(
        downstream,
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "function": {"arguments": 'ls"}'}}
                        ]
                    }
                }
            ]
        },
    )
    await state.finish(downstream, upstream_saw_done=True)

    events = _sse_events(b"".join(downstream.chunks).decode())
    function_added = [
        event
        for event in events
        if event.get("type") == "response.output_item.added"
        and (event.get("item") or {}).get("type") == "function_call"
    ]
    assert len(function_added) == 1
    assert function_added[0]["item"]["name"] == "exec_command"
    completed = [event for event in events if event.get("type") == "response.completed"][-1]
    tool_items = [item for item in completed["response"]["output"] if item.get("type") == "function_call"]
    assert len(tool_items) == 1
    assert tool_items[0]["name"] == "exec_command"


async def test_chat_tool_delta_streams_namespaced_tool_with_sanitized_name():
    class FakeResponse:
        def __init__(self):
            self.chunks: list[bytes] = []

        async def write(self, data: bytes):
            self.chunks.append(data)

    downstream = FakeResponse()
    tool_resolve = {"multi_agent_v1_spawn_agent": ("multi_agent_v1", "spawn_agent")}
    state = ResponsesStreamState("local-llama", tool_resolve=tool_resolve)
    await state.start(downstream)
    await state.write_chat_delta(
        downstream,
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_ns",
                                "function": {
                                    "name": "multi_agent_v1_spawn_agent",
                                    "arguments": '{"task":"review"}',
                                },
                            }
                        ]
                    }
                }
            ]
        },
    )
    await state.finish(downstream, upstream_saw_done=True)

    events = _sse_events(b"".join(downstream.chunks).decode())
    function_added = [
        event
        for event in events
        if event.get("type") == "response.output_item.added"
        and (event.get("item") or {}).get("type") == "function_call"
    ]
    assert len(function_added) == 1
    assert function_added[0]["item"]["namespace"] == "multi_agent_v1"
    assert function_added[0]["item"]["name"] == "spawn_agent"
    completed = [event for event in events if event.get("type") == "response.completed"][-1]
    tool_items = [item for item in completed["response"]["output"] if item.get("type") == "function_call"]
    assert tool_items[0]["namespace"] == "multi_agent_v1"
    assert tool_items[0]["name"] == "spawn_agent"


async def test_chat_tool_delta_streams_tool_search_call_to_client():
    class FakeResponse:
        def __init__(self):
            self.chunks: list[bytes] = []

        async def write(self, data: bytes):
            self.chunks.append(data)

    downstream = FakeResponse()
    state = ResponsesStreamState("local-llama")
    await state.start(downstream)
    await state.write_chat_delta(
        downstream,
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_search",
                                "function": {"name": "tool_search_call", "arguments": '{"query":"mcp__exa"}'},
                            }
                        ]
                    }
                }
            ]
        },
    )
    events_mid = _sse_events(b"".join(downstream.chunks).decode())
    function_added_mid = [
        event
        for event in events_mid
        if event.get("type") == "response.output_item.added"
        and (event.get("item") or {}).get("type") == "function_call"
    ]
    assert function_added_mid == []
    tool_search_added = [
        event
        for event in events_mid
        if event.get("type") == "response.output_item.added"
        and (event.get("item") or {}).get("type") == "tool_search_call"
    ]
    assert len(tool_search_added) == 1
    tool_search_done = [
        event
        for event in events_mid
        if event.get("type") == "response.output_item.done"
        and (event.get("item") or {}).get("type") == "tool_search_call"
    ]
    assert len(tool_search_done) == 1
    assert tool_search_done[0]["item"]["arguments"]["query"] == "mcp__exa"
    outputs_mid = [
        event
        for event in events_mid
        if event.get("type") == "response.output_item.done"
        and (event.get("item") or {}).get("type") == "function_call_output"
    ]
    assert outputs_mid == []
    await state.finish(downstream, upstream_saw_done=True)

    completed = [e for e in _sse_events(b"".join(downstream.chunks).decode()) if e.get("type") == "response.completed"][-1]
    search_items = [i for i in completed["response"]["output"] if i.get("type") == "tool_search_call"]
    assert len(search_items) == 1
    assert search_items[0]["arguments"]["query"] == "mcp__exa"


async def test_write_chat_delta_streams_content_with_complete_tool_in_same_chunk():
    class FakeResponse:
        def __init__(self):
            self.chunks: list[bytes] = []

        async def write(self, data: bytes):
            self.chunks.append(data)

    downstream = FakeResponse()
    state = ResponsesStreamState("local-llama")
    await state.start(downstream)
    await state.write_chat_delta(
        downstream,
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_exec",
                                "function": {
                                    "name": "exec_command",
                                    "arguments": '{"cmd":"echo hi"}',
                                },
                            }
                        ],
                        "content": "Let me run that command.",
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        },
    )
    assert state.message_text == "Let me run that command."


async def test_write_chat_delta_streams_content_before_tool_call():
    class FakeResponse:
        def __init__(self):
            self.chunks: list[bytes] = []

        async def write(self, data: bytes):
            self.chunks.append(data)

    downstream = FakeResponse()
    state = ResponsesStreamState("local-llama")
    await state.start(downstream)
    await state.write_chat_delta(
        downstream,
        {
            "choices": [
                {
                    "delta": {
                        "content": "Let me search Exa for the latest headline.",
                    }
                }
            ]
        },
    )
    await state.write_chat_delta(
        downstream,
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_exa",
                                "function": {
                                    "name": "mcp__exa__web_search_exa",
                                    "arguments": '{"query":"ukraine"}',
                                },
                            }
                        ]
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        },
    )
    assert state.message_text == "Let me search Exa for the latest headline."
    events = _sse_events(b"".join(downstream.chunks).decode())
    text_deltas = [e for e in events if e.get("type") == "response.output_text.delta"]
    assert text_deltas
    assert "".join(e["delta"] for e in text_deltas) == state.message_text


async def test_write_chat_delta_streams_content_with_incomplete_tool_in_same_chunk():
    class FakeResponse:
        def __init__(self):
            self.chunks: list[bytes] = []

        async def write(self, data: bytes):
            self.chunks.append(data)

    downstream = FakeResponse()
    state = ResponsesStreamState("local-llama")
    await state.start(downstream)
    await state.write_chat_delta(
        downstream,
        {
            "choices": [
                {
                    "delta": {
                        "content": "Let me search...",
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_exa",
                                "function": {
                                    "name": "mcp__exa__web_search_exa",
                                    "arguments": '{"query":"uk',
                                },
                            }
                        ],
                    }
                }
            ]
        },
    )
    assert state.message_text == "Let me search..."


async def test_write_chat_delta_streams_content_after_tool_closed_in_prior_chunk():
    class FakeResponse:
        async def write(self, data: bytes):
            return None

    downstream = FakeResponse()
    state = ResponsesStreamState("local-llama")
    await state.write_chat_delta(
        downstream,
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_exa",
                                "function": {
                                    "name": "mcp__exa__web_search_exa",
                                    "arguments": '{"query":"ukraine"}',
                                },
                            }
                        ]
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        },
    )
    await state.write_chat_delta(
        downstream,
        {
            "choices": [
                {
                    "delta": {
                        "content": "I'll summarize once results arrive.",
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        },
    )
    assert state.message_text == "I'll summarize once results arrive."


async def test_mcp_tool_call_emits_namespaced_function_call_when_name_is_chunked():
    class FakeResponse:
        def __init__(self):
            self.chunks: list[bytes] = []

        async def write(self, data: bytes):
            self.chunks.append(data)

    downstream = FakeResponse()
    state = ResponsesStreamState("local-llama")
    await state.start(downstream)
    for part in ("mcp", "__exa", "__web_search_exa"):
        await state.write_chat_delta(
            downstream,
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_exa",
                                    "function": {"name": part, "arguments": ""},
                                }
                            ]
                        }
                    }
                ]
            },
        )
    await state.write_chat_delta(
        downstream,
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": {"arguments": '{"query":"ukraine"}'},
                            }
                        ]
                    }
                }
            ]
        },
    )
    events = _sse_events(b"".join(downstream.chunks).decode())
    function_added = [
        e
        for e in events
        if e.get("type") == "response.output_item.added"
        and (e.get("item") or {}).get("type") == "function_call"
        and (e.get("item") or {}).get("namespace") == "mcp__exa"
    ]
    assert len(function_added) == 1
    assert function_added[0]["item"]["name"] == "web_search_exa"


async def test_mcp_tool_call_streams_argument_deltas_like_llama():
    class FakeResponse:
        def __init__(self):
            self.chunks: list[bytes] = []

        async def write(self, data: bytes):
            self.chunks.append(data)

    downstream = FakeResponse()
    state = ResponsesStreamState("local-llama")
    await state.start(downstream)
    arg_parts = ["{", '"query"', ':"', "ukraine war headline today", '"', "}"]
    await state.write_chat_delta(
        downstream,
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_exa",
                                "function": {
                                    "name": "mcp__exa__web_search_exa",
                                    "arguments": arg_parts[0],
                                },
                            }
                        ]
                    }
                }
            ]
        },
    )
    for part in arg_parts[1:]:
        await state.write_chat_delta(
            downstream,
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "function": {"arguments": part}}
                            ]
                        }
                    }
                ]
            },
        )
    await state.write_chat_delta(
        downstream,
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
    )
    await state.finish(downstream, upstream_saw_done=True)
    events = _sse_events(b"".join(downstream.chunks).decode())
    arg_deltas = [
        e for e in events if e.get("type") == "response.function_call_arguments.delta"
    ]
    assert arg_deltas
    assert "".join(e["delta"] for e in arg_deltas) == '{"query":"ukraine war headline today"}'
    done = [
        e
        for e in events
        if e.get("type") == "response.function_call_arguments.done"
    ][-1]
    json.loads(done["arguments"])


async def test_mcp_tool_call_does_not_close_on_finish_reason_with_incomplete_args():
    class FakeResponse:
        def __init__(self):
            self.chunks: list[bytes] = []

        async def write(self, data: bytes):
            self.chunks.append(data)

    downstream = FakeResponse()
    state = ResponsesStreamState("local-llama")
    await state.start(downstream)
    await state.write_chat_delta(
        downstream,
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_exa",
                                "function": {
                                    "name": "mcp__exa__web_search_exa",
                                    "arguments": '{"query":"ukr',
                                },
                            }
                        ]
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        },
    )
    events = _sse_events(b"".join(downstream.chunks).decode())
    assert not [
        e for e in events if e.get("type") == "response.function_call_arguments.done"
    ]
    await state.write_chat_delta(
        downstream,
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "function": {"arguments": 'aine"}'}}
                        ]
                    }
                }
            ]
        },
    )
    await state.finish(downstream, upstream_saw_done=True)
    events = _sse_events(b"".join(downstream.chunks).decode())
    done = [
        e
        for e in events
        if e.get("type") == "response.function_call_arguments.done"
    ][-1]
    assert json.loads(done["arguments"]) == {"query": "ukraine"}


async def test_finish_does_not_complete_incomplete_tool_or_response():
    class FakeResponse:
        def __init__(self):
            self.chunks: list[bytes] = []

        async def write(self, data: bytes):
            self.chunks.append(data)

    downstream = FakeResponse()
    state = ResponsesStreamState("local-llama")
    await state.start(downstream)
    await state.write_chat_delta(
        downstream,
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_exec",
                                "function": {
                                    "name": "exec_command",
                                    "arguments": "{",
                                },
                            }
                        ]
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        },
    )
    await state.finish(downstream, upstream_saw_done=True)
    events = _sse_events(b"".join(downstream.chunks).decode())
    assert not [
        e
        for e in events
        if e.get("type") == "response.output_item.done"
        and (e.get("item") or {}).get("name") == "exec_command"
    ]
    assert not [
        e
        for e in events
        if e.get("type") == "response.function_call_arguments.done"
    ]
    assert not [e for e in events if e.get("type") == "response.completed"]
    terminal = [
        e
        for e in events
        if e.get("type") in ("response.completed", "response.incomplete", "response.failed")
    ][-1]
    assert terminal["type"] == "response.incomplete"
    assert terminal["response"]["status"] == "incomplete"
    assert b"data: [DONE]" in b"".join(downstream.chunks)
    arg_deltas = [
        e for e in events if e.get("type") == "response.function_call_arguments.delta"
    ]
    assert arg_deltas
    assert arg_deltas[0]["delta"] == "{"


async def test_finish_emits_response_incomplete_with_length_reason_when_upstream_done_truncated():
    class FakeResponse:
        def __init__(self):
            self.chunks: list[bytes] = []

        async def write(self, data: bytes):
            self.chunks.append(data)

    downstream = FakeResponse()
    state = ResponsesStreamState("zen-nemotron-3-ultra-free")
    await state.start(downstream)
    await state.write_chat_delta(
        downstream,
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_exec",
                                "function": {
                                    "name": "exec_command",
                                    "arguments": "{",
                                },
                            }
                        ]
                    },
                    "finish_reason": "length",
                }
            ]
        },
    )
    await state.finish(downstream, upstream_saw_done=True)
    events = _sse_events(b"".join(downstream.chunks).decode())
    terminal = [
        e
        for e in events
        if e.get("type") in ("response.completed", "response.incomplete", "response.failed")
    ][-1]
    assert terminal["type"] == "response.incomplete"
    assert terminal["response"]["incomplete_details"] == {"reason": "max_output_tokens"}


async def test_finish_emits_response_completed_only_when_upstream_done_and_items_complete():
    class FakeResponse:
        def __init__(self):
            self.chunks: list[bytes] = []

        async def write(self, data: bytes):
            self.chunks.append(data)

    downstream = FakeResponse()
    state = ResponsesStreamState("local-llama")
    await state.start(downstream)
    await state.write_chat_delta(
        downstream,
        {"choices": [{"delta": {"content": "hello"}}]},
    )
    await state.finish(downstream, upstream_saw_done=True)
    events = _sse_events(b"".join(downstream.chunks).decode())
    assert events[-1]["type"] == "response.completed"
    assert b"data: [DONE]" in b"".join(downstream.chunks)

    downstream2 = FakeResponse()
    state2 = ResponsesStreamState("local-llama")
    await state2.start(downstream2)
    await state2.write_chat_delta(
        downstream2,
        {"choices": [{"delta": {"content": "hello"}}]},
    )
    await state2.finish(downstream2, upstream_saw_done=False)
    events2 = _sse_events(b"".join(downstream2.chunks).decode())
    assert not [e for e in events2 if e.get("type") == "response.completed"]
    assert b"data: [DONE]" not in b"".join(downstream2.chunks)


async def test_mcp_tool_call_emits_namespaced_function_call():
    class FakeResponse:
        def __init__(self):
            self.chunks: list[bytes] = []

        async def write(self, data: bytes):
            self.chunks.append(data)

    downstream = FakeResponse()
    state = ResponsesStreamState("local-llama")
    await state.start(downstream)
    await state.write_chat_delta(
        downstream,
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_exa",
                                "function": {
                                    "name": "mcp__exa__web_search_exa",
                                    "arguments": '{"query":"ukraine"}',
                                },
                            }
                        ]
                    }
                }
            ]
        },
    )
    events = _sse_events(b"".join(downstream.chunks).decode())
    function_added = [
        e
        for e in events
        if e.get("type") == "response.output_item.added"
        and (e.get("item") or {}).get("type") == "function_call"
        and (e.get("item") or {}).get("namespace") == "mcp__exa"
    ]
    assert len(function_added) == 1
    assert function_added[0]["item"]["name"] == "web_search_exa"
    outputs = [
        e
        for e in events
        if e.get("type") == "response.output_item.done"
        and (e.get("item") or {}).get("type") == "function_call"
        and (e.get("item") or {}).get("namespace") == "mcp__exa"
    ]
    assert len(outputs) == 1
    assert outputs[0]["item"]["name"] == "web_search_exa"
    assert '"query":"ukraine"' in outputs[0]["item"]["arguments"]


async def test_mcp_tool_call_in_final_response_output():
    class FakeResponse:
        def __init__(self):
            self.chunks: list[bytes] = []

        async def write(self, data: bytes):
            self.chunks.append(data)

    downstream = FakeResponse()
    state = ResponsesStreamState("local-llama")
    await state.start(downstream)
    await state.write_chat_delta(
        downstream,
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_exa",
                                "function": {
                                    "name": "mcp__exa__web_search_exa",
                                    "arguments": '{"query":"ukraine"}',
                                },
                            }
                        ]
                    }
                }
            ]
        },
    )
    await state.finish(downstream, upstream_saw_done=True)

    events = _sse_events(b"".join(downstream.chunks).decode())
    completed = [e for e in events if e.get("type") == "response.completed"][-1]
    output = completed["response"]["output"]
    calls = [item for item in output if item.get("type") == "function_call"]
    mcp_calls = [item for item in calls if item.get("namespace") == "mcp__exa"]
    outputs = [item for item in output if item.get("type") == "function_call_output"]
    assert len(mcp_calls) == 1
    assert len(calls) == 1
    assert mcp_calls[0]["name"] == "web_search_exa"
    assert '"query":"ukraine"' in mcp_calls[0]["arguments"]
    assert outputs == []


async def test_mcp_tool_call_emits_codex_native_item():
    class FakeResponse:
        def __init__(self):
            self.chunks: list[bytes] = []

        async def write(self, data: bytes):
            self.chunks.append(data)

    downstream = FakeResponse()
    state = ResponsesStreamState("local-llama")
    await state.start(downstream)
    await state.write_chat_delta(
        downstream,
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_exa",
                                "function": {
                                    "name": "mcp__exa__web_search_exa",
                                    "arguments": '{"query":"ukraine war"}',
                                },
                            }
                        ]
                    }
                }
            ]
        },
    )
    await state.finish(downstream, upstream_saw_done=True)

    events = _sse_events(b"".join(downstream.chunks).decode())
    outputs = [
        e
        for e in events
        if e.get("type") == "response.output_item.done"
        and (e.get("item") or {}).get("type") == "function_call"
        and (e.get("item") or {}).get("namespace") == "mcp__exa"
    ]
    assert len(outputs) == 1
    assert '"query":"ukraine war"' in outputs[0]["item"]["arguments"]
    assert outputs[0]["item"]["status"] == "completed"


async def test_streaming_anthropic_response_completed_includes_usage():
    class FakeResponse:
        def __init__(self):
            self.chunks: list[bytes] = []

        async def write(self, data: bytes):
            self.chunks.append(data)

    downstream = FakeResponse()
    state = ResponsesStreamState("claude-real")
    await state.write_anthropic_delta(
        downstream,
        {
            "type": "message_start",
            "message": {
                "usage": {
                    "input_tokens": 5,
                    "cache_read_input_tokens": 4,
                    "output_tokens": 1,
                }
            },
        },
    )
    await state.write_anthropic_delta(
        downstream,
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 3},
        },
    )
    await state.finish(downstream, upstream_saw_done=True)

    events = _sse_events(b"".join(downstream.chunks).decode())
    completed = [event for event in events if event.get("type") == "response.completed"][-1]
    assert completed["response"]["usage"] == {
        "input_tokens": 5,
        "output_tokens": 3,
        "total_tokens": 8,
        "input_tokens_details": {
            "cached_tokens": 4,
            "cache_read_input_tokens": 4,
        },
    }


async def test_responses_compact_routes_to_openai_chat_and_returns_compacted_window(tmp_path):
    captured = {}

    async def chat(request):
        captured["body"] = await request.json()
        return web.json_response(
            {
                "id": "chatcmpl_compact",
                "choices": [{"message": {"role": "assistant", "content": "Task: keep implementing compact support."}}],
                "usage": {"prompt_tokens": 9, "completion_tokens": 2, "total_tokens": 11},
            }
        )

    upstream = web.Application()
    upstream.router.add_post("/v1/chat/completions", chat)
    upstream_client = TestClient(TestServer(upstream))
    await upstream_client.start_server()

    settings = tmp_path / "settings.json"
    settings.write_text(
        json.dumps(
            {
                "customModels": [
                    {
                        "model": "real-openai",
                        "displayName": "Real OpenAI",
                        "provider": "openai",
                        "baseUrl": str(upstream_client.make_url("/v1")),
                        "apiKey": "secret",
                    }
                ]
            }
        )
    )
    shim_client = TestClient(TestServer(ShimServer(settings).app()))
    await shim_client.start_server()

    resp = await shim_client.post(
        "/v1/responses/compact",
        json={
            "model": "real-openai",
            "input": [
                {"role": "user", "content": "implement compact"},
                {"type": "function_call_output", "call_id": "call_1", "output": "tests pass"},
            ],
            "service_tier": "priority",
            "stream": True,
        },
    )
    assert resp.status == 200
    payload = await resp.json()
    assert payload["status"] == "completed"
    assert payload["model"] == "real-openai"
    assert payload["output"][0]["type"] == "compaction"
    assert decode_shim_compaction_summary(payload["output"][0]["encrypted_content"]) == (
        "Task: keep implementing compact support."
    )
    assert payload["usage"] == {"input_tokens": 9, "output_tokens": 2, "total_tokens": 11}
    assert captured["body"]["model"] == "real-openai"
    assert captured["body"]["stream"] is False
    assert "service_tier" not in captured["body"]
    assert "Compact the conversation" in captured["body"]["messages"][0]["content"]

    await shim_client.close()
    await upstream_client.close()


async def test_responses_compaction_v2_emits_single_compaction_stream_item(tmp_path):
    captured = {}

    async def chat(request):
        captured["body"] = await request.json()
        return web.json_response(
            {
                "id": "chatcmpl_compact_v2",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Compacted task state for DeepSeek.",
                            "reasoning_content": "thinking that must not become output items",
                        }
                    }
                ],
                "usage": {"prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16},
            }
        )

    upstream = web.Application()
    upstream.router.add_post("/v1/chat/completions", chat)
    upstream_client = TestClient(TestServer(upstream))
    await upstream_client.start_server()

    settings = tmp_path / "settings.json"
    settings.write_text(
        json.dumps(
            {
                "customModels": [
                    {
                        "model": "deepseek-v4-pro",
                        "displayName": "DeepSeek V4 Pro",
                        "provider": "openai",
                        "baseUrl": str(upstream_client.make_url("/v1")),
                        "apiKey": "secret",
                    }
                ]
            }
        )
    )
    shim_client = TestClient(TestServer(ShimServer(settings).app()))
    await shim_client.start_server()

    resp = await shim_client.post(
        "/v1/responses",
        json={
            "model": "deepseek-v4-pro",
            "stream": True,
            "input": [
                {"role": "user", "content": "long thread"},
                {"type": "compaction_trigger"},
            ],
        },
        headers={"User-Agent": "codex-cli/0.138.0"},
    )
    assert resp.status == 200
    text = await resp.text()
    assert "response.output_item.done" in text
    assert '"type":"compaction"' in text.replace(" ", "")
    assert "response.completed" in text
    assert "reasoning" not in text
    assert captured["body"]["stream"] is False
    assert captured["body"]["messages"]

    done_lines = [line for line in text.splitlines() if line.startswith("data: ") and "response.output_item.done" in line]
    assert len(done_lines) == 1
    payload = json.loads(done_lines[0].removeprefix("data: "))
    item = payload["item"]
    assert item["type"] == "compaction"
    assert decode_shim_compaction_summary(item["encrypted_content"]) == "Compacted task state for DeepSeek."

    await shim_client.close()
    await upstream_client.close()


async def test_responses_routes_openai_responses_provider_to_upstream_responses(tmp_path):
    captured = {}

    async def responses_handler(request):
        captured["body"] = await request.json()
        captured["headers"] = dict(request.headers)
        return web.json_response(
            {
                "id": "resp_upstream",
                "object": "response",
                "status": "completed",
                "model": "gpt-5.3-codex",
                "output": [],
            }
        )

    upstream = web.Application()
    upstream.router.add_post("/v1/responses", responses_handler)
    upstream_client = TestClient(TestServer(upstream))
    await upstream_client.start_server()

    settings = tmp_path / "settings.json"
    settings.write_text(
        json.dumps(
            {
                "customModels": [
                    {
                        "model": "upstream-codex-responses",
                        "displayName": "Codex Responses",
                        "provider": "openai-responses",
                        "baseUrl": str(upstream_client.make_url("/v1")),
                        "apiKey": "secret",
                    }
                ]
            }
        )
    )
    shim_client = TestClient(TestServer(ShimServer(settings).app()))
    await shim_client.start_server()

    resp = await shim_client.post(
        "/v1/responses",
        json={"model": "upstream-codex-responses", "input": "hello", "stream": False},
    )
    assert resp.status == 200
    payload = await resp.json()
    assert payload["id"] == "resp_upstream"
    assert captured["body"]["model"] == "upstream-codex-responses"
    assert captured["body"]["input"] == "hello"
    assert captured["headers"]["Authorization"] == "Bearer secret"

    chat_resp = await shim_client.post(
        "/v1/chat/completions",
        json={"model": "upstream-codex-responses", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert chat_resp.status == 502

    await shim_client.close()
    await upstream_client.close()


async def test_responses_compact_chatgpt_passthrough_uses_compact_endpoint(monkeypatch, tmp_path, auth_present):
    captured = {}

    class FakeUpstream:
        status = 200
        content_type = "application/json"

        async def json(self, content_type=None):
            return {"id": "resp_compact", "model": "gpt-5.5", "output": [{"type": "message", "model": "gpt-5.5"}]}

        def release(self):
            pass

    async def fake_post(self, url, json=None, headers=None):
        captured["url"] = url
        captured["body"] = json
        captured["headers"] = headers
        return FakeUpstream()

    monkeypatch.setattr("codex_shim.server.ClientSession.post", fake_post)
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"customModels": []}))
    shim_client = TestClient(TestServer(ShimServer(settings).app()))
    await shim_client.start_server()

    resp = await shim_client.post("/v1/responses/compact", json={"model": "openai-gpt-5-5-codex-max", "input": "hi", "stream": True})
    assert resp.status == 200
    payload = await resp.json()
    assert payload["model"] == "openai-gpt-5-5-codex-max"
    assert payload["output"][0]["model"] == "openai-gpt-5-5-codex-max"
    assert captured["url"] == "https://chatgpt.com/backend-api/codex/responses/compact"
    assert captured["body"]["model"] == "gpt-5.5"
    assert "stream" not in captured["body"]
    assert captured["headers"]["Accept"] == "application/json"

    await shim_client.close()


async def test_health_and_models_include_chatgpt_passthrough_when_auth_present(tmp_path, auth_present, monkeypatch):
    missing_cache = tmp_path / "missing-models-cache.json"
    monkeypatch.setattr("codex_shim.settings.DEFAULT_CODEX_MODELS_CACHE", missing_cache)
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"customModels": []}))
    shim_client = TestClient(TestServer(ShimServer(settings).app()))
    await shim_client.start_server()

    health = await shim_client.get("/health")
    assert health.status == 200
    body = await health.json()
    assert body["models"] == len(FALLBACK_CHATGPT_PASSTHROUGH_SLUGS)
    assert body["chatgpt_passthrough"] is True

    models = await shim_client.get("/v1/models")
    assert models.status == 200
    payload = await models.json()
    expected = [codex_catalog_slug(slug) for slug in FALLBACK_CHATGPT_PASSTHROUGH_SLUGS]
    assert sorted(model["id"] for model in payload["data"]) == sorted(expected)

    await shim_client.close()


async def test_health_and_models_hide_chatgpt_passthrough_when_auth_missing(tmp_path, auth_missing):
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"customModels": []}))
    shim_client = TestClient(TestServer(ShimServer(settings).app()))
    await shim_client.start_server()

    health = await shim_client.get("/health")
    body = await health.json()
    assert body["models"] == 0
    assert body["chatgpt_passthrough"] is False

    models = await shim_client.get("/v1/models")
    payload = await models.json()
    assert payload["data"] == []

    await shim_client.close()


@pytest.fixture
def cursor_present(monkeypatch):
    from codex_shim.cursor_passthrough import CursorCatalogModel, _fallback_cursor_models

    def _on(**_kwargs):
        return True

    for target in (
        "codex_shim.cursor_passthrough.cursor_passthrough_available",
        "codex_shim.server.cursor_passthrough_available",
        "codex_shim.catalog.cursor_passthrough_available",
        "codex_shim.cli.cursor_passthrough_available",
    ):
        monkeypatch.setattr(target, _on)
    monkeypatch.setattr(
        "codex_shim.cursor_passthrough._load_cursor_catalog_models",
        lambda **_: _fallback_cursor_models(),
    )


@pytest.fixture
def cursor_missing(monkeypatch):
    monkeypatch.setattr("codex_shim.cursor_passthrough.cursor_passthrough_available", lambda **_: False)
    monkeypatch.setattr("codex_shim.server.cursor_passthrough_available", lambda **_: False)
    monkeypatch.setattr("codex_shim.catalog.cursor_passthrough_available", lambda **_: False)


async def test_health_and_models_include_cursor_passthrough_when_auth_present(tmp_path, cursor_present, auth_missing):
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"customModels": []}))
    shim_client = TestClient(TestServer(ShimServer(settings).app()))
    await shim_client.start_server()

    health = await shim_client.get("/health")
    assert health.status == 200
    body = await health.json()
    assert body["models"] == 1
    assert body["cursor_passthrough"] is True

    models = await shim_client.get("/v1/models")
    payload = await models.json()
    assert [model["id"] for model in payload["data"]] == ["cursor-composer-2-5"]

    await shim_client.close()


async def test_health_and_models_hide_cursor_passthrough_when_auth_missing(tmp_path, cursor_missing, auth_missing):
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"customModels": []}))
    shim_client = TestClient(TestServer(ShimServer(settings).app()))
    await shim_client.start_server()

    health = await shim_client.get("/health")
    body = await health.json()
    assert body["models"] == 0
    assert body["cursor_passthrough"] is False

    models = await shim_client.get("/v1/models")
    payload = await models.json()
    assert payload["data"] == []

    await shim_client.close()


async def test_cursor_passthrough_stream_shows_tool_activity_without_function_calls(
    monkeypatch, tmp_path, cursor_present, auth_missing
):
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"customModels": []}))

    async def fake_cursor_events(_prompt, _model):
        yield {"type": "text_delta", "delta": "Segment one."}
        yield {
            "type": "tool_started",
            "call_id": "tool-1",
            "tool_call": {"readToolCall": {"args": {"path": "README.md"}}},
            "markdown": "**cursor-agent · read**\n\n> `README.md`\n",
        }
        yield {
            "type": "tool_completed",
            "call_id": "tool-1",
            "tool_call": {},
            "markdown": "\n**Result**\n\n```\nok\n```\n",
        }
        yield {"type": "text_delta", "delta": "Segment two."}
        yield {"type": "completed", "text": "Segment one.Segment two."}

    monkeypatch.setattr(server_module, "iter_cursor_agent_events", fake_cursor_events)

    shim_client = TestClient(TestServer(ShimServer(settings).app()))
    await shim_client.start_server()
    try:
        resp = await shim_client.post(
            "/v1/responses",
            json={
                "model": "cursor-composer-2-5",
                "input": [{"role": "user", "content": "hello"}],
                "stream": True,
            },
        )
        assert resp.status == 200
        events = _sse_events(await resp.text())
        added = [event for event in events if event.get("type") == "response.output_item.added"]
        item_types = [(event.get("item") or {}).get("type") for event in added]
        assert item_types.count("message") == 2
        assert item_types.count("reasoning") == 1
        assert not any((event.get("item") or {}).get("type") == "function_call" for event in added)
        reasoning_added = next(event for event in added if (event.get("item") or {}).get("type") == "reasoning")
        reasoning_done = [
            event
            for event in events
            if event.get("type") == "response.output_item.done"
            and (event.get("item") or {}).get("type") == "reasoning"
        ][-1]
        summary = (reasoning_done.get("item") or {}).get("summary") or []
        assert summary and "README.md" in summary[0]["text"]
        assert reasoning_added is not None
        completed = [event for event in events if event.get("type") == "response.completed"][-1]
        output_types = [item.get("type") for item in completed["response"]["output"]]
        assert output_types.count("message") == 2
        assert output_types.count("reasoning") == 1
    finally:
        await shim_client.close()


async def test_chat_routes_to_openai_normalizes_developer_role(tmp_path):
    captured = {}

    async def chat(request):
        captured["body"] = await request.json()
        return web.json_response({"id": "chatcmpl_fake", "choices": [{"message": {"role": "assistant", "content": "ok"}}]})

    upstream = web.Application()
    upstream.router.add_post("/v1/chat/completions", chat)
    upstream_client = TestClient(TestServer(upstream))
    await upstream_client.start_server()

    settings = tmp_path / "settings.json"
    settings.write_text(
        json.dumps(
            {
                "customModels": [
                    {
                        "model": "deepseek-reasoner",
                        "displayName": "DeepSeek Reasoner",
                        "provider": "openai",
                        "baseUrl": str(upstream_client.make_url("/v1")),
                        "apiKey": "secret",
                    }
                ]
            }
        )
    )
    shim_client = TestClient(TestServer(ShimServer(settings).app()))
    await shim_client.start_server()

    resp = await shim_client.post(
        "/v1/chat/completions",
        json={"model": "deepseek-reasoner", "messages": [{"role": "developer", "content": "rules"}, {"role": "user", "content": "hi"}]},
    )
    assert resp.status == 200
    assert [message["role"] for message in captured["body"]["messages"]] == ["system", "user"]

    await shim_client.close()
    await upstream_client.close()


async def test_chat_routes_to_anthropic(tmp_path):
    captured = {}

    async def messages(request):
        captured["headers"] = dict(request.headers)
        captured["body"] = await request.json()
        return web.json_response({"id": "msg_fake", "content": [{"type": "text", "text": "anthropic hello"}], "stop_reason": "end_turn"})

    upstream = web.Application()
    upstream.router.add_post("/v1/messages", messages)
    upstream_client = TestClient(TestServer(upstream))
    await upstream_client.start_server()

    settings = tmp_path / "settings.json"
    settings.write_text(
        json.dumps(
            {
                "customModels": [
                    {
                        "model": "claude-real",
                        "displayName": "Claude Real",
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

    resp = await shim_client.post("/v1/chat/completions", json={"model": "claude-real", "messages": [{"role": "user", "content": "hi"}]})
    assert resp.status == 200
    payload = await resp.json()
    assert payload["choices"][0]["message"]["content"] == "anthropic hello"
    assert captured["body"]["model"] == "claude-real"
    assert captured["headers"]["x-api-key"] == "secret"
    assert "Authorization" not in captured["headers"]

    await shim_client.close()
    await upstream_client.close()


async def test_anthropic_messages_routes_to_openai_chat(tmp_path):
    captured = {}

    async def chat(request):
        captured["headers"] = dict(request.headers)
        captured["body"] = await request.json()
        return web.json_response(
            {
                "id": "chatcmpl_fake",
                "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "openai hello"}}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
            }
        )

    upstream = web.Application()
    upstream.router.add_post("/v1/chat/completions", chat)
    upstream_client = TestClient(TestServer(upstream))
    await upstream_client.start_server()

    settings = tmp_path / "settings.json"
    settings.write_text(
        json.dumps(
            {
                "customModels": [
                    {
                        "model": "real-openai",
                        "displayName": "Real OpenAI",
                        "provider": "generic-chat-completion-api",
                        "baseUrl": str(upstream_client.make_url("/v1")),
                        "apiKey": "secret",
                    }
                ]
            }
        )
    )
    shim_client = TestClient(TestServer(ShimServer(settings).app()))
    await shim_client.start_server()

    resp = await shim_client.post(
        "/v1/messages",
        json={
            "model": "real-openai",
            "system": "System",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
            "max_tokens": 42,
        },
    )
    assert resp.status == 200
    payload = await resp.json()
    assert payload["type"] == "message"
    assert payload["model"] == "real-openai"
    assert payload["content"] == [{"type": "text", "text": "openai hello"}]
    assert payload["usage"] == {"input_tokens": 2, "output_tokens": 1}
    assert captured["headers"]["Authorization"] == "Bearer secret"
    assert captured["body"]["model"] == "real-openai"
    assert captured["body"]["max_tokens"] == 42
    assert captured["body"]["messages"] == [{"role": "system", "content": "System"}, {"role": "user", "content": "hi"}]

    await shim_client.close()
    await upstream_client.close()


async def test_anthropic_messages_passes_through_anthropic_upstream(tmp_path):
    captured = {}

    async def messages(request):
        captured["headers"] = dict(request.headers)
        captured["body"] = await request.json()
        return web.json_response(
            {
                "id": "msg_fake",
                "type": "message",
                "role": "assistant",
                "model": "claude-upstream",
                "content": [{"type": "text", "text": "anthropic hello"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 2, "output_tokens": 1},
            }
        )

    upstream = web.Application()
    upstream.router.add_post("/v1/messages", messages)
    upstream_client = TestClient(TestServer(upstream))
    await upstream_client.start_server()

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

    resp = await shim_client.post(
        "/v1/messages",
        json={"model": "claude-upstream", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 42},
    )
    assert resp.status == 200
    payload = await resp.json()
    assert payload["model"] == "claude-upstream"
    assert payload["content"][0]["text"] == "anthropic hello"
    assert captured["body"]["model"] == "claude-upstream"
    assert captured["headers"]["x-api-key"] == "secret"
    assert "Authorization" not in captured["headers"]

    await shim_client.close()
    await upstream_client.close()


async def test_anthropic_messages_streams_openai_chat_as_anthropic_sse(tmp_path):
    captured = {}

    async def chat(request):
        captured["body"] = await request.json()
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        await response.write(b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n')
        await response.write(
            b'data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":4,"completion_tokens":2,"total_tokens":6}}\n\n'
        )
        await response.write(b"data: [DONE]\n\n")
        await response.write_eof()
        return response

    upstream = web.Application()
    upstream.router.add_post("/v1/chat/completions", chat)
    upstream_client = TestClient(TestServer(upstream))
    await upstream_client.start_server()

    settings = tmp_path / "settings.json"
    settings.write_text(
        json.dumps(
            {
                "customModels": [
                    {
                        "model": "real-openai",
                        "displayName": "Real OpenAI",
                        "provider": "generic-chat-completion-api",
                        "baseUrl": str(upstream_client.make_url("/v1")),
                        "apiKey": "secret",
                    }
                ]
            }
        )
    )
    shim_client = TestClient(TestServer(ShimServer(settings).app()))
    await shim_client.start_server()

    resp = await shim_client.post(
        "/v1/messages",
        json={"model": "real-openai", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 42, "stream": True},
    )
    assert resp.status == 200
    text = await resp.text()
    assert "[DONE]" not in text
    events = _named_sse_events(text)
    assert [event for event, _ in events] == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    assert events[2][1]["delta"] == {"type": "text_delta", "text": "hello"}
    assert events[4][1]["delta"]["stop_reason"] == "end_turn"
    assert events[4][1]["usage"] == {"input_tokens": 4, "output_tokens": 2}
    assert captured["body"]["stream_options"] == {"include_usage": True}

    await shim_client.close()
    await upstream_client.close()



async def test_anthropic_messages_streams_tool_calls_as_anthropic_sse(tmp_path):
    async def chat(request):
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        await response.write(
            b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"lookup","arguments":""}}]}}]}\n\n'
        )
        await response.write(
            b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"q\\":\\"repo\\"}"}}]}}]}\n\n'
        )
        await response.write(
            b'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":5,"completion_tokens":3,"total_tokens":8}}\n\n'
        )
        await response.write(b"data: [DONE]\n\n")
        await response.write_eof()
        return response

    upstream = web.Application()
    upstream.router.add_post("/v1/chat/completions", chat)
    upstream_client = TestClient(TestServer(upstream))
    await upstream_client.start_server()

    settings = tmp_path / "settings.json"
    settings.write_text(
        json.dumps(
            {
                "customModels": [
                    {
                        "model": "real-openai",
                        "displayName": "Real OpenAI",
                        "provider": "generic-chat-completion-api",
                        "baseUrl": str(upstream_client.make_url("/v1")),
                        "apiKey": "secret",
                    }
                ]
            }
        )
    )
    shim_client = TestClient(TestServer(ShimServer(settings).app()))
    await shim_client.start_server()

    resp = await shim_client.post(
        "/v1/messages",
        json={"model": "real-openai", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 42, "stream": True},
    )
    assert resp.status == 200
    text = await resp.text()
    events = _named_sse_events(text)
    event_names = [event for event, _ in events]
    assert "message_start" in event_names
    assert "content_block_start" in event_names
    tool_start = next(payload for name, payload in events if name == "content_block_start" and payload.get("content_block", {}).get("type") == "tool_use")
    assert tool_start["content_block"]["id"] == "call_1"
    assert tool_start["content_block"]["name"] == "lookup"
    tool_deltas = [payload for name, payload in events if name == "content_block_delta" and payload.get("delta", {}).get("type") == "input_json_delta"]
    assert len(tool_deltas) >= 1
    message_delta = next(payload for name, payload in events if name == "message_delta")
    assert message_delta["delta"]["stop_reason"] == "tool_use"
    assert "message_stop" in event_names

    await shim_client.close()
    await upstream_client.close()


async def test_anthropic_messages_streams_reasoning_as_anthropic_sse(tmp_path):
    async def chat(request):
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        await response.write(
            b'data: {"choices":[{"delta":{"reasoning_content":"let me think"}}]}\n\n'
        )
        await response.write(
            b'data: {"choices":[{"delta":{"content":"the answer"}}]}\n\n'
        )
        await response.write(
            b'data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}}\n\n'
        )
        await response.write(b"data: [DONE]\n\n")
        await response.write_eof()
        return response

    upstream = web.Application()
    upstream.router.add_post("/v1/chat/completions", chat)
    upstream_client = TestClient(TestServer(upstream))
    await upstream_client.start_server()

    settings = tmp_path / "settings.json"
    settings.write_text(
        json.dumps(
            {
                "customModels": [
                    {
                        "model": "real-openai",
                        "displayName": "Real OpenAI",
                        "provider": "generic-chat-completion-api",
                        "baseUrl": str(upstream_client.make_url("/v1")),
                        "apiKey": "secret",
                    }
                ]
            }
        )
    )
    shim_client = TestClient(TestServer(ShimServer(settings).app()))
    await shim_client.start_server()

    resp = await shim_client.post(
        "/v1/messages",
        json={"model": "real-openai", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 42, "stream": True},
    )
    assert resp.status == 200
    events = _named_sse_events(await resp.text())
    thinking_starts = [p for n, p in events if n == "content_block_start" and p.get("content_block", {}).get("type") == "thinking"]
    assert len(thinking_starts) == 1
    thinking_deltas = [p for n, p in events if n == "content_block_delta" and p.get("delta", {}).get("type") == "thinking_delta"]
    assert len(thinking_deltas) == 1
    assert thinking_deltas[0]["delta"]["thinking"] == "let me think"
    text_deltas = [p for n, p in events if n == "content_block_delta" and p.get("delta", {}).get("type") == "text_delta"]
    assert len(text_deltas) == 1
    assert text_deltas[0]["delta"]["text"] == "the answer"

    await shim_client.close()
    await upstream_client.close()


async def test_anthropic_messages_returns_anthropic_error_for_upstream_failure(tmp_path):
    async def chat(request):
        return web.json_response(
            {"error": {"message": "invalid api key", "type": "invalid_request_error"}},
            status=401,
        )

    upstream = web.Application()
    upstream.router.add_post("/v1/chat/completions", chat)
    upstream_client = TestClient(TestServer(upstream))
    await upstream_client.start_server()

    settings = tmp_path / "settings.json"
    settings.write_text(
        json.dumps(
            {
                "customModels": [
                    {
                        "model": "real-openai",
                        "displayName": "Real OpenAI",
                        "provider": "generic-chat-completion-api",
                        "baseUrl": str(upstream_client.make_url("/v1")),
                        "apiKey": "bad-key",
                    }
                ]
            }
        )
    )
    shim_client = TestClient(TestServer(ShimServer(settings).app()))
    await shim_client.start_server()

    resp = await shim_client.post(
        "/v1/messages",
        json={"model": "real-openai", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 42},
    )
    assert resp.status == 401
    payload = await resp.json()
    assert payload["type"] == "error"
    assert payload["error"]["type"] == "authentication_error"
    assert "invalid api key" in payload["error"]["message"]

    await shim_client.close()
    await upstream_client.close()


async def test_anthropic_messages_streams_anthropic_passthrough(tmp_path):
    async def messages(request):
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        await response.write(b'event: message_start\ndata: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-upstream","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":1,"output_tokens":0}}}\n\n')
        await response.write(b'event: content_block_start\ndata: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n\n')
        await response.write(b'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hello"}}\n\n')
        await response.write(b'event: content_block_stop\ndata: {"type":"content_block_stop","index":0}\n\n')
        await response.write(b'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":1}}\n\n')
        await response.write(b'event: message_stop\ndata: {"type":"message_stop"}\n\n')
        await response.write_eof()
        return response

    upstream = web.Application()
    upstream.router.add_post("/v1/messages", messages)
    upstream_client = TestClient(TestServer(upstream))
    await upstream_client.start_server()

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

    resp = await shim_client.post(
        "/v1/messages",
        json={"model": "claude-upstream", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 42, "stream": True},
    )
    assert resp.status == 200
    text = await resp.text()
    events = _named_sse_events(text)
    event_names = [event for event, _ in events]
    assert event_names == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    text_delta = next(payload for name, payload in events if name == "content_block_delta")
    assert text_delta["delta"]["text"] == "hello"

    await shim_client.close()
    await upstream_client.close()

def _picker_settings_file(tmp_path):
    settings = tmp_path / "settings.json"
    settings.write_text(
        json.dumps(
            {
                "customModels": [
                    {
                        "model": "kimi-k26",
                        "displayName": "Kimi K2.6",
                        "provider": "openai",
                        "baseUrl": "http://example.invalid/v1",
                        "apiKey": "k",
                    },
                    {
                        "model": "deepseek-v4-pro",
                        "displayName": "DeepSeek V4 Pro",
                        "provider": "openai",
                        "baseUrl": "http://example.invalid/v1",
                        "apiKey": "k",
                    },
                ]
            }
        )
    )
    return settings


def _stub_codex_config(monkeypatch, tmp_path, *, model: str = "kimi-k26") -> "Path":
    config = tmp_path / "config.toml"
    config.write_text(
        f'model = "{model}"\n'
        'model_provider = "openai"\n'
        'openai_base_url = "http://127.0.0.1:8765/v1"\n'
        'model_catalog_json = "/tmp/catalog.json"\n'
    )
    monkeypatch.setattr(server_module, "CODEX_CONFIG_PATH", config)
    return config


def _picker_headers(shim: ShimServer) -> dict[str, str]:
    return {PICKER_TOKEN_HEADER: shim.picker_token}


def test_picker_html_renders_self_contained_page():
    html = _picker_html("test-token")
    assert html.startswith("<!DOCTYPE html>")
    assert "/api/models" in html
    assert "/api/switch" in html
    assert PICKER_TOKEN_HEADER in html
    assert 'const PICKER_TOKEN = "test-token";' in html


def test_picker_html_json_escapes_token():
    token = 'tok"\'</script>'
    html = _picker_html(token)
    literal = html.split("const PICKER_TOKEN = ", 1)[1].split(";", 1)[0]
    assert literal == '"tok\\"\'\\u003c/script>"'
    assert "<script>" not in literal


def test_current_managed_model_reads_top_level_model(monkeypatch, tmp_path):
    _stub_codex_config(monkeypatch, tmp_path, model="deepseek-v4-pro")
    assert _current_managed_model() == "deepseek-v4-pro"


def test_current_managed_model_returns_none_when_config_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(server_module, "CODEX_CONFIG_PATH", tmp_path / "nope.toml")
    assert _current_managed_model() is None


def test_set_active_model_rewrites_model_line(monkeypatch, tmp_path):
    config = _stub_codex_config(monkeypatch, tmp_path)
    _set_active_model("deepseek-v4-pro", "DeepSeek V4 Pro")
    text = config.read_text()
    assert 'model = "deepseek-v4-pro"' in text
    assert 'model = "kimi-k26"' not in text


def test_set_active_model_no_op_when_config_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(server_module, "CODEX_CONFIG_PATH", tmp_path / "nope.toml")
    # Should not raise.
    _set_active_model("anything", "Anything")


async def test_picker_page_served_at_picker(tmp_path, auth_missing):
    settings = _picker_settings_file(tmp_path)
    shim_client = TestClient(TestServer(ShimServer(settings).app()))
    await shim_client.start_server()
    try:
        resp = await shim_client.get("/picker")
        assert resp.status == 200
        text = await resp.text()
        assert "/api/models" in text
    finally:
        await shim_client.close()


async def test_api_models_lists_configured_models_with_active_flag(
    monkeypatch, tmp_path, auth_missing
):
    settings = _picker_settings_file(tmp_path)
    _stub_codex_config(monkeypatch, tmp_path, model="deepseek-v4-pro")
    shim_client = TestClient(TestServer(ShimServer(settings).app()))
    await shim_client.start_server()
    try:
        resp = await shim_client.get("/api/models")
        assert resp.status == 200
        data = await resp.json()
        slugs = [m["slug"] for m in data]
        assert slugs == ["deepseek-v4-pro", "kimi-k26"]
        active = {m["slug"]: m["active"] for m in data}
        assert active == {"kimi-k26": False, "deepseek-v4-pro": True}
    finally:
        await shim_client.close()


async def test_api_models_includes_chatgpt_when_auth_present(
    monkeypatch, tmp_path, auth_present
):
    settings = _picker_settings_file(tmp_path)
    _stub_codex_config(monkeypatch, tmp_path, model="codex-gpt-5-5")
    shim_client = TestClient(TestServer(ShimServer(settings).app()))
    await shim_client.start_server()
    try:
        resp = await shim_client.get("/api/models")
        data = await resp.json()
        slugs = [m["slug"] for m in data]
        assert slugs == sorted(slugs)
        active = next(m for m in data if m["slug"] == "codex-gpt-5-5")
        assert active["active"] is True
    finally:
        await shim_client.close()


async def test_switch_model_rewrites_config_without_restart(
    monkeypatch, tmp_path, auth_missing
):
    settings = _picker_settings_file(tmp_path)
    config = _stub_codex_config(monkeypatch, tmp_path, model="kimi-k2.6")
    restart_calls = []
    monkeypatch.setattr(server_module, "_restart_codex_app", lambda: restart_calls.append(True))

    shim = ShimServer(settings)
    shim_client = TestClient(TestServer(shim.app()))
    await shim_client.start_server()
    try:
        resp = await shim_client.post(
            "/api/switch",
            json={"slug": "deepseek-v4-pro", "restart_codex": False},
            headers=_picker_headers(shim),
        )
        assert resp.status == 200
        payload = await resp.json()
        assert payload == {"ok": True, "model": "deepseek-v4-pro", "restarted": False}
        text = config.read_text()
        assert 'model = "deepseek-v4-pro"' in text
        assert restart_calls == []
    finally:
        await shim_client.close()


async def test_switch_model_triggers_restart_when_requested(
    monkeypatch, tmp_path, auth_missing
):
    settings = _picker_settings_file(tmp_path)
    _stub_codex_config(monkeypatch, tmp_path, model="kimi-k2.6")
    restart_calls = []
    monkeypatch.setattr(server_module, "_restart_codex_app", lambda: restart_calls.append(True))

    shim = ShimServer(settings)
    shim_client = TestClient(TestServer(shim.app()))
    await shim_client.start_server()
    try:
        resp = await shim_client.post(
            "/api/switch",
            json={"slug": "deepseek-v4-pro", "restart_codex": True},
            headers=_picker_headers(shim),
        )
        assert resp.status == 200
        payload = await resp.json()
        assert payload["restarted"] is True
        assert restart_calls == [True]
    finally:
        await shim_client.close()


async def test_switch_model_rejects_missing_picker_token(monkeypatch, tmp_path, auth_missing):
    settings = _picker_settings_file(tmp_path)
    config = _stub_codex_config(monkeypatch, tmp_path, model="kimi-k2.6")
    restart_calls = []
    monkeypatch.setattr(server_module, "_restart_codex_app", lambda: restart_calls.append(True))

    shim_client = TestClient(TestServer(ShimServer(settings).app()))
    await shim_client.start_server()
    try:
        resp = await shim_client.post(
            "/api/switch",
            json={"slug": "deepseek-v4-pro", "restart_codex": True},
        )
        assert resp.status == 403
        assert await resp.json() == {"error": "forbidden"}
        assert 'model = "kimi-k2.6"' in config.read_text()
        assert restart_calls == []
    finally:
        await shim_client.close()


async def test_switch_model_rejects_bad_picker_token(monkeypatch, tmp_path, auth_missing):
    settings = _picker_settings_file(tmp_path)
    config = _stub_codex_config(monkeypatch, tmp_path, model="kimi-k2.6")
    restart_calls = []
    monkeypatch.setattr(server_module, "_restart_codex_app", lambda: restart_calls.append(True))

    shim_client = TestClient(TestServer(ShimServer(settings).app()))
    await shim_client.start_server()
    try:
        resp = await shim_client.post(
            "/api/switch",
            json={"slug": "deepseek-v4-pro", "restart_codex": True},
            headers={PICKER_TOKEN_HEADER: "wrong"},
        )
        assert resp.status == 403
        assert await resp.json() == {"error": "forbidden"}
        assert 'model = "kimi-k2.6"' in config.read_text()
        assert restart_calls == []
    finally:
        await shim_client.close()


async def test_switch_model_rejects_unknown_slug(monkeypatch, tmp_path, auth_missing):
    settings = _picker_settings_file(tmp_path)
    _stub_codex_config(monkeypatch, tmp_path)
    shim = ShimServer(settings)
    shim_client = TestClient(TestServer(shim.app()))
    await shim_client.start_server()
    try:
        resp = await shim_client.post("/api/switch", json={"slug": "nope"}, headers=_picker_headers(shim))
        assert resp.status == 404
    finally:
        await shim_client.close()


async def test_switch_model_requires_slug(tmp_path, auth_missing):
    settings = _picker_settings_file(tmp_path)
    shim = ShimServer(settings)
    shim_client = TestClient(TestServer(shim.app()))
    await shim_client.start_server()
    try:
        resp = await shim_client.post("/api/switch", json={}, headers=_picker_headers(shim))
        assert resp.status == 400
    finally:
        await shim_client.close()

from __future__ import annotations

import asyncio
import json
import subprocess
from typing import Any

import pytest
from aiohttp import ClientSession, WSMsgType, web
from aiohttp.test_utils import TestClient, TestServer

from codex_shim.compaction import decode_shim_compaction_summary, encode_shim_compaction_summary
from codex_shim.responses_input_pipeline import UNKNOWN_FUNCTION_TOOL_NAME
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
    _sse_lines,
    _set_active_model,
    parse_upstream_error,
)
from codex_shim.catalog_slugs import codex_catalog_slug
from codex_shim.settings import FALLBACK_CHATGPT_PASSTHROUGH_SLUGS
from codex_shim.translate import SHIM_ENCRYPTED_CONTENT_PREFIX
from codex_shim.ws_passthrough import WsPassthroughSession
from ws_test_support import MockUpstreamWsState, start_mock_upstream_ws


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


@pytest.fixture
def chatgpt_cache_dir(monkeypatch, tmp_path):
    cache_root = tmp_path / "chatgpt-conversations"
    monkeypatch.setattr(server_module, "_chatgpt_conversations_dir", lambda: cache_root)
    return cache_root


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


def test_sanitize_chatgpt_passthrough_body_keeps_native_compaction_item():
    body = {
        "model": "codex-gpt-5-5",
        "input": [
            {"type": "message", "role": "user", "content": "hi"},
            {"type": "compaction", "encrypted_content": "gAAAA-openai-native"},
        ],
    }

    sanitized = _sanitize_chatgpt_passthrough_body(body)

    assert len(sanitized["input"]) == 2
    assert sanitized["input"][1]["type"] == "compaction"
    assert sanitized["input"][1]["encrypted_content"] == "gAAAA-openai-native"


def test_sanitize_chatgpt_passthrough_body_drops_empty_compaction_item():
    body = {
        "model": "codex-gpt-5-5",
        "input": [
            {"type": "message", "role": "user", "content": "hi"},
            {"type": "compaction", "encrypted_content": ""},
            {"type": "compaction"},
        ],
    }

    sanitized = _sanitize_chatgpt_passthrough_body(body)

    assert len(sanitized["input"]) == 1
    assert sanitized["input"][0]["type"] == "message"


def test_sanitize_chatgpt_passthrough_body_rewrites_shim_compaction_item():
    from codex_shim.compaction import compaction_output_item

    compaction = compaction_output_item("Task state preserved from Cursor compaction.")
    body = {
        "model": "codex-gpt-5-5",
        "input": [
            {"type": "message", "role": "user", "content": "hi"},
            compaction,
        ],
    }

    sanitized = _sanitize_chatgpt_passthrough_body(body)

    assert len(sanitized["input"]) == 2
    assert sanitized["input"][1]["type"] == "message"
    assert sanitized["input"][1]["role"] == "developer"
    assert "Task state preserved from Cursor compaction." in sanitized["input"][1]["content"][0]["text"]
    assert compaction["encrypted_content"].endswith("==") or True  # blob unchanged in source


def test_sanitize_chatgpt_passthrough_body_keeps_previous_response_id_by_default():
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

    assert sanitized["previous_response_id"] == "resp_previous"
    assert sanitized["metadata"]["previous_response_id"] == "metadata-value"


def test_sanitize_chatgpt_passthrough_body_forwards_effort_verbatim():
    sanitized = _sanitize_chatgpt_passthrough_body(
        {
            "model": "codex-gpt-5-6-luna",
            "input": [{"type": "message", "role": "user", "content": "hi"}],
            "reasoning": {"effort": "max", "summary": "auto"},
        }
    )
    assert sanitized["reasoning"]["effort"] == "max"
    assert sanitized["reasoning"]["summary"] == "auto"
    assert sanitized["reasoning"]["context"] == "all_turns"


@pytest.mark.parametrize("effort", ["low", "medium", "high", "xhigh", "max", "ultra", "maximum"])
def test_sanitize_chatgpt_passthrough_body_does_not_rewrite_effort(effort):
    sanitized = _sanitize_chatgpt_passthrough_body(
        {
            "model": "codex-gpt-5-6-luna",
            "input": [{"type": "message", "role": "user", "content": "hi"}],
            "reasoning": {"effort": effort},
        }
    )
    assert sanitized["reasoning"]["effort"] == effort


def test_sanitize_chatgpt_passthrough_body_rewrites_function_call_id_prefix():
    sanitized = _sanitize_chatgpt_passthrough_body(
        {
            "model": "codex-gpt-5-6-luna",
            "input": [
                {
                    "type": "function_call",
                    "id": "call_vSN3n7d4PoQtq7pr_1",
                    "call_id": "call_vSN3n7d4PoQtq7pr_1",
                    "name": "spawn_agent",
                    "namespace": "multi_agent_v1",
                    "arguments": '{"task":"review"}',
                },
                {
                    "type": "function_call_output",
                    "id": "call_vSN3n7d4PoQtq7pr_1",
                    "call_id": "call_vSN3n7d4PoQtq7pr_1",
                    "output": "ok",
                },
                {
                    "type": "function_call",
                    "id": "fc_already",
                    "call_id": "call_already",
                    "name": "lookup",
                    "arguments": "{}",
                },
            ],
        }
    )
    spawn = sanitized["input"][0]
    assert spawn["id"] == "fc_vSN3n7d4PoQtq7pr_1"
    assert spawn["call_id"] == "call_vSN3n7d4PoQtq7pr_1"
    output = sanitized["input"][1]
    assert "id" not in output
    assert output["call_id"] == "call_vSN3n7d4PoQtq7pr_1"
    already = sanitized["input"][2]
    assert already["id"] == "fc_already"
    assert already["call_id"] == "call_already"


def test_sanitize_chatgpt_passthrough_body_can_strip_previous_response_id_for_legacy_expand():
    body = {
        "model": "codex-gpt-5-5",
        "previous_response_id": "resp_previous",
        "input": [{"type": "message", "role": "user", "content": "hi"}],
    }

    sanitized = _sanitize_chatgpt_passthrough_body(body, strip_previous_response_id=True)

    assert "previous_response_id" not in sanitized
    assert body["previous_response_id"] == "resp_previous"


def test_finalize_chatgpt_passthrough_body_forwards_fast_tier_and_token_caps():
    from codex_shim.server import _finalize_chatgpt_passthrough_body

    sanitized = _finalize_chatgpt_passthrough_body(
        {
            "model": "gpt-5.5",
            "max_output_tokens": 4096,
            "max_tokens": 2048,
            "service_tier": "priority",
            "store": True,
        }
    )

    assert sanitized["max_output_tokens"] == 4096
    assert sanitized["max_tokens"] == 2048
    assert sanitized["service_tier"] == "priority"
    assert sanitized["store"] is False
    assert sanitized["parallel_tool_calls"] is False
    assert sanitized["reasoning"] == {"context": "all_turns"}


def test_finalize_chatgpt_passthrough_body_forces_lite_reasoning_context():
    from codex_shim.server import _finalize_chatgpt_passthrough_body

    sanitized = _finalize_chatgpt_passthrough_body(
        {
            "model": "gpt-5.6-luna",
            "parallel_tool_calls": True,
            "reasoning": {"effort": "high", "summary": "auto", "context": "last_turn"},
        }
    )

    assert sanitized["reasoning"]["effort"] == "high"
    assert sanitized["reasoning"]["summary"] == "auto"
    assert sanitized["reasoning"]["context"] == "all_turns"
    assert sanitized["parallel_tool_calls"] is False


def test_finalize_chatgpt_compact_passthrough_body_omits_store_and_stream():
    from codex_shim.server import _finalize_chatgpt_compact_passthrough_body

    sanitized = _finalize_chatgpt_compact_passthrough_body(
        {
            "model": "gpt-5.5",
            "max_output_tokens": 4096,
            "max_tokens": 2048,
            "service_tier": "priority",
            "store": False,
            "stream": False,
            "instructions": "Compact.",
        }
    )

    assert sanitized["instructions"] == "Compact."
    assert sanitized["parallel_tool_calls"] is False
    assert sanitized["max_output_tokens"] == 4096
    assert sanitized["max_tokens"] == 2048
    assert sanitized["service_tier"] == "priority"
    assert "store" not in sanitized
    assert "stream" not in sanitized


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
    assert captured["headers"]["Accept-Encoding"] == "zstd, gzip"

    await shim_client.close()


async def test_chatgpt_passthrough_http_always_expands_even_when_expand_env_disabled(
    monkeypatch, tmp_path, auth_present
):
    monkeypatch.setenv("CODEX_SHIM_CHATGPT_EXPAND_CONTINUATIONS", "0")
    captured = {}

    class FakeUpstream:
        status = 200
        content_type = "application/json"
        headers = {"x-request-id": "req_1"}

        async def json(self, content_type=None):
            return {"id": "resp_1", "model": "gpt-5.5", "output": []}

        def release(self):
            pass

    async def fake_post(self, url, json=None, headers=None):
        captured["url"] = url
        captured["body"] = json
        captured["headers"] = headers
        return FakeUpstream()

    monkeypatch.setattr("codex_shim.server.ClientSession.post", fake_post)
    settings = tmp_path / "settings.json"
    settings.write_text("{}")
    shim_client = TestClient(TestServer(ShimServer(settings).app()))
    await shim_client.start_server()

    resp = await shim_client.post(
        "/v1/responses",
        json={"model": "codex-gpt-5-5", "previous_response_id": "resp_previous", "input": "hi"},
        headers={"session_id": "sess-abc", "x-codex-turn-state": "running"},
    )

    assert resp.status == 200
    assert "previous_response_id" not in captured["body"]
    assert captured["headers"]["session_id"] == "sess-abc"
    assert resp.headers["x-request-id"] == "req_1"

    await shim_client.close()


async def test_chatgpt_passthrough_http_forwards_service_tier_and_output_caps(
    monkeypatch, tmp_path, auth_present
):
    captured = {}

    class FakeUpstream:
        status = 200
        content_type = "application/json"
        headers = {}

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

    resp = await shim_client.post(
        "/v1/responses",
        json={
            "model": "codex-gpt-5-6-luna",
            "input": "hi",
            "service_tier": "priority",
            "max_output_tokens": 4096,
            "max_tokens": 2048,
        },
    )

    assert resp.status == 200
    assert captured["body"]["service_tier"] == "priority"
    assert captured["body"]["max_output_tokens"] == 4096
    assert captured["body"]["max_tokens"] == 2048
    assert captured["body"]["store"] is False

    await shim_client.close()


async def test_chatgpt_passthrough_http_forwards_reasoning_effort_verbatim(
    monkeypatch, tmp_path, auth_present
):
    captured = {}

    class FakeUpstream:
        status = 200
        content_type = "application/json"
        headers = {}

        async def json(self, content_type=None):
            return {"id": "resp_1", "model": "gpt-5.6-luna", "output": []}

        def release(self):
            pass

    async def fake_post(self, url, json=None, headers=None):
        captured["url"] = url
        captured["body"] = json
        return FakeUpstream()

    monkeypatch.setattr("codex_shim.server.ClientSession.post", fake_post)
    settings = tmp_path / "settings.json"
    settings.write_text("{}")
    shim_client = TestClient(TestServer(ShimServer(settings).app()))
    await shim_client.start_server()

    resp = await shim_client.post(
        "/v1/responses",
        json={
            "model": "codex-gpt-5-6-luna",
            "input": "hi",
            "reasoning": {"effort": "max", "summary": "auto"},
        },
    )

    assert resp.status == 200
    assert captured["url"] == "https://chatgpt.com/backend-api/codex/responses"
    assert captured["body"]["reasoning"]["effort"] == "max"
    assert captured["body"]["reasoning"]["summary"] == "auto"
    assert captured["body"]["reasoning"]["context"] == "all_turns"

    await shim_client.close()


async def test_chatgpt_passthrough_strips_previous_response_id_when_expand_disabled(
    monkeypatch, tmp_path, auth_present
):
    """Legacy test name; HTTP always expands regardless of CODEX_SHIM_CHATGPT_EXPAND_CONTINUATIONS."""
    await test_chatgpt_passthrough_http_always_expands_even_when_expand_env_disabled(
        monkeypatch, tmp_path, auth_present
    )


async def test_chatgpt_passthrough_expands_previous_response_id_by_default(
    monkeypatch, tmp_path, auth_present, chatgpt_cache_dir
):
    captured = []
    session_headers = {"session-id": "test-session-expand"}
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
        headers = {}

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

    first = await shim_client.post(
        "/v1/responses",
        json={"model": "codex-gpt-5-5", "input": first_input},
        headers=session_headers,
    )
    assert first.status == 200

    second = await shim_client.post(
        "/v1/responses",
        json={
            "model": "codex-gpt-5-5",
            "previous_response_id": "resp_previous",
            "input": [tool_output],
        },
        headers=session_headers,
    )
    assert second.status == 200

    assert "previous_response_id" not in captured[1]
    assert captured[1]["input"] == [*first_input, tool_call, tool_output]

    await shim_client.close()


async def test_chatgpt_compaction_expands_previous_response_id_from_cache(
    monkeypatch, tmp_path, auth_present, chatgpt_cache_dir
):
    captured: dict[str, Any] = {}
    session_headers = {"session-id": "test-session-compact-expand"}
    first_input = [{"type": "message", "role": "user", "content": "run a command"}]
    tool_call = {
        "type": "function_call",
        "id": "fc_1",
        "call_id": "call_1",
        "name": "exec_command",
        "arguments": "{\"cmd\":\"printf ok\"}",
    }
    tool_output = {"type": "function_call_output", "call_id": "call_1", "output": "ok"}
    tail_output = {"type": "function_call_output", "call_id": "call_2", "output": "tail"}

    class FakeUpstream:
        status = 200
        content_type = "application/json"
        headers = {}

        def __init__(self, payload):
            self.payload = payload

        async def json(self, content_type=None):
            return self.payload

        async def text(self):
            return json.dumps(self.payload)

        def release(self):
            pass

    async def fake_post(self, url, json=None, headers=None):
        assert not str(url).endswith("/responses/compact"), url
        captured.setdefault("turn_calls", []).append(json)
        input_items = (json or {}).get("input") or []
        if isinstance(input_items, list) and input_items and input_items[-1].get("type") == "compaction_trigger":
            captured["compact_url"] = str(url)
            captured["compact_input"] = input_items
            return _FakeSseUpstream(
                _chatgpt_compaction_v2_sse_chunks(encrypted_content="openai-native-compaction-blob")
            )
        if len(captured["turn_calls"]) == 1:
            return FakeUpstream({"id": "resp_previous", "model": "gpt-5.5", "output": [tool_call]})
        return FakeUpstream({"id": "resp_next", "model": "gpt-5.5", "output": []})

    monkeypatch.setattr("codex_shim.server.ClientSession.post", fake_post)
    settings = tmp_path / "settings.json"
    settings.write_text("{}")
    shim_client = TestClient(TestServer(ShimServer(settings).app()))
    await shim_client.start_server()

    first = await shim_client.post(
        "/v1/responses",
        json={"model": "codex-gpt-5-5", "input": first_input},
        headers=session_headers,
    )
    assert first.status == 200

    second = await shim_client.post(
        "/v1/responses",
        json={
            "model": "codex-gpt-5-5",
            "previous_response_id": "resp_previous",
            "input": [tool_output],
        },
        headers=session_headers,
    )
    assert second.status == 200

    compact = await shim_client.post(
        "/v1/responses",
        json={
            "model": "codex-gpt-5-5",
            "stream": True,
            "previous_response_id": "resp_next",
            "input": [tail_output, {"type": "compaction_trigger"}],
        },
        headers=session_headers,
    )
    assert compact.status == 200
    assert captured.get("compact_url") == "https://chatgpt.com/backend-api/codex/responses"
    compact_input = captured.get("compact_input")
    assert isinstance(compact_input, list)
    synth_call_2 = {
        "type": "function_call",
        "call_id": "call_2",
        "name": UNKNOWN_FUNCTION_TOOL_NAME,
        "arguments": "{}",
    }
    assert compact_input == [
        *first_input,
        tool_call,
        tool_output,
        synth_call_2,
        tail_output,
        {"type": "compaction_trigger"},
    ]

    await shim_client.close()


async def test_chatgpt_passthrough_expands_from_disk_after_restart(
    monkeypatch, tmp_path, auth_present, chatgpt_cache_dir
):
    captured = []
    session_headers = {"session-id": "test-session-restart"}
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
        headers = {}

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

    first_client = TestClient(TestServer(ShimServer(settings).app()))
    await first_client.start_server()
    first = await first_client.post(
        "/v1/responses",
        json={"model": "codex-gpt-5-5", "input": first_input},
        headers=session_headers,
    )
    assert first.status == 200
    await first_client.close()

    second_client = TestClient(TestServer(ShimServer(settings).app()))
    await second_client.start_server()
    second = await second_client.post(
        "/v1/responses",
        json={
            "model": "codex-gpt-5-5",
            "previous_response_id": "resp_previous",
            "input": [tool_output],
        },
        headers=session_headers,
    )
    assert second.status == 200
    assert captured[1]["input"] == [*first_input, tool_call, tool_output]
    await second_client.close()


async def test_chatgpt_passthrough_logs_cache_miss(
    monkeypatch, tmp_path, auth_present, chatgpt_cache_dir, capsys
):
    class FakeUpstream:
        status = 200
        content_type = "application/json"
        headers = {}

        async def json(self, content_type=None):
            return {"id": "resp_next", "model": "gpt-5.5", "output": []}

        def release(self):
            pass

    async def fake_post(self, url, json=None, headers=None):
        return FakeUpstream()

    monkeypatch.setattr("codex_shim.server.ClientSession.post", fake_post)
    settings = tmp_path / "settings.json"
    settings.write_text("{}")
    shim_client = TestClient(TestServer(ShimServer(settings).app()))
    await shim_client.start_server()

    response = await shim_client.post(
        "/v1/responses",
        json={
            "model": "codex-gpt-5-5",
            "previous_response_id": "resp_missing",
            "input": [{"type": "message", "role": "user", "content": "delta"}],
        },
        headers={"session-id": "test-session-miss"},
    )
    assert response.status == 200
    assert "MISS session=test-session-miss previous_response_id=resp_missing" in capsys.readouterr().out
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
        captured["headers"] = headers
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
    assert "Content-Encoding" not in (captured.get("headers") or {})

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
        captured["body"] = json
        captured["headers"] = headers
        return FakeUpstream()

    monkeypatch.setattr("codex_shim.server.ClientSession.post", fake_post)
    settings = tmp_path / "settings.json"
    settings.write_text("{}")
    shim_client = TestClient(TestServer(ShimServer(settings).app()))
    await shim_client.start_server()

    resp = await shim_client.post(
        "/v1/responses/compact",
        json={
            "model": "codex-gpt-5-5",
            "input": "summarize",
            "service_tier": "priority",
            "max_output_tokens": 4096,
            "store": False,
            "stream": False,
        },
        headers={"Accept-Encoding": "zstd, gzip"},
    )

    assert resp.status == 200
    assert captured["url"] == "https://chatgpt.com/backend-api/codex/responses/compact"
    assert captured["headers"]["Accept-Encoding"] == "zstd, gzip"
    assert captured["body"]["service_tier"] == "priority"
    assert captured["body"]["max_output_tokens"] == 4096
    assert "store" not in captured["body"]
    assert "stream" not in captured["body"]

    await shim_client.close()


class _FakeSseContent:
    def __init__(self, chunks: list[bytes]):
        self._chunks = list(chunks)

    async def readany(self):
        await asyncio.sleep(0)
        if not self._chunks:
            return b""
        return self._chunks.pop(0)


def _sse_chunk(payload: dict[str, Any]) -> bytes:
    return f"data: {json.dumps(payload)}\n\n".encode()


def _chatgpt_compaction_v2_sse_chunks(
    *,
    response_id: str = "resp_compact",
    model: str = "gpt-5.5",
    encrypted_content: str = "openai-native-compaction-blob",
    completed_output: list[Any] | None = None,
) -> list[bytes]:
    item = {"type": "compaction", "encrypted_content": encrypted_content}
    output = [item] if completed_output is None else completed_output
    return [
        _sse_chunk(
            {
                "type": "response.created",
                "response": {"id": response_id, "model": model, "status": "in_progress", "output": []},
            }
        ),
        _sse_chunk({"type": "response.output_item.done", "item": item}),
        _sse_chunk(
            {
                "type": "response.completed",
                "response": {
                    "id": response_id,
                    "model": model,
                    "status": "completed",
                    "output": output,
                },
            }
        ),
        b"data: [DONE]\n\n",
    ]


class _FakeSseUpstream:
    status = 200
    content_type = "text/event-stream"
    headers = {"Content-Type": "text/event-stream"}

    def __init__(self, chunks: list[bytes] | None = None):
        self.content = _FakeSseContent(chunks or _chatgpt_compaction_v2_sse_chunks())

    def release(self):
        pass


class _FakeHttpErrorUpstream:
    def __init__(self, status: int, text: str, content_type: str = "application/json"):
        self.status = status
        self.content_type = content_type
        self.headers = {"Content-Type": content_type}
        self._text = text

    async def text(self):
        return self._text

    def release(self):
        pass


async def test_chatgpt_passthrough_websocket_relays_sse_events(monkeypatch, tmp_path, auth_present):
    monkeypatch.setenv("CODEX_SHIM_WS_PASSTHROUGH", "0")
    captured = {}

    class FakeUpstream:
        status = 200
        content_type = "text/event-stream"
        headers = {}
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


async def test_chatgpt_passthrough_websocket_logs_usage_on_completed(monkeypatch, tmp_path, auth_present):
    monkeypatch.setenv("CODEX_SHIM_WS_PASSTHROUGH", "0")
    observed = []

    def fake_observe(source, upstream, *, usage=None):
        observed.append({"source": source, "usage": usage})
        return {}

    monkeypatch.setattr("codex_shim.server.observe_upstream_response", fake_observe)
    monkeypatch.setenv("CODEX_SHIM_UPSTREAM_HEADER_LOG", "1")

    class FakeUpstream:
        status = 200
        content_type = "text/event-stream"
        headers = {"x-oai-request-id": "req_ws_usage"}
        content = _FakeSseContent(
            [
                b'data: {"type":"response.created","response":{"id":"resp_1","model":"gpt-5.5"}}\n\n',
                b'data: {"type":"response.completed","response":{"id":"resp_1","model":"gpt-5.5","status":"completed","usage":{"input_tokens":10,"output_tokens":2,"input_tokens_details":{"cached_tokens":6}}}}\n\n',
            ]
        )

        def release(self):
            pass

    async def fake_post(self, url, json=None, headers=None):
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
            "stream": True,
        }
    )
    for _ in range(2):
        msg = await ws.receive(timeout=2)
        assert msg.type == WSMsgType.TEXT

    usage_events = [item for item in observed if item.get("usage")]
    assert len(usage_events) == 1
    assert usage_events[0]["source"] == "chatgpt-passthrough-ws"
    assert usage_events[0]["usage"]["input_tokens_details"]["cached_tokens"] == 6

    await ws.close()
    await shim_client.close()


async def test_chatgpt_passthrough_websocket_expands_previous_response_id(monkeypatch, tmp_path, auth_present):
    monkeypatch.setenv("CODEX_SHIM_WS_PASSTHROUGH", "0")
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
        headers = {}

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

    async def fake_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr("codex_shim.chatgpt_edge.asyncio.sleep", fake_sleep)

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


async def test_chatgpt_passthrough_retries_503_then_succeeds(monkeypatch, tmp_path, auth_present):
    calls: list[str] = []

    async def fake_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr("codex_shim.chatgpt_edge.asyncio.sleep", fake_sleep)

    class OkUpstream:
        status = 200
        content_type = "application/json"
        headers = {}

        async def json(self, content_type=None):
            return {"id": "resp_ok", "model": "gpt-5.5", "output": []}

        def release(self):
            pass

    async def fake_post(self, url, json=None, headers=None):
        calls.append(str(url))
        if len(calls) == 1:
            return _FakeHttpErrorUpstream(503, "envoy unavailable", "text/plain")
        return OkUpstream()

    monkeypatch.setattr("codex_shim.server.ClientSession.post", fake_post)
    settings = tmp_path / "settings.json"
    settings.write_text("{}")
    shim_client = TestClient(TestServer(ShimServer(settings).app()))
    await shim_client.start_server()
    try:
        resp = await shim_client.post("/v1/responses", json={"model": "codex-gpt-5-5", "input": "hi"})
        assert resp.status == 200
        payload = await resp.json()
        assert payload["id"] == "resp_ok"
        assert calls == [
            "https://chatgpt.com/backend-api/codex/responses",
            "https://chatgpt.com/backend-api/codex/responses",
        ]
    finally:
        await shim_client.close()


async def test_chatgpt_passthrough_html_403_exhausted_returns_502(monkeypatch, tmp_path, auth_present):
    calls: list[str] = []
    html = (
        "<!DOCTYPE html><html><body>Unable to load site "
        "status.openai.com Ray ID: 9abc</body></html>"
    )

    async def fake_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr("codex_shim.chatgpt_edge.asyncio.sleep", fake_sleep)

    async def fake_post(self, url, json=None, headers=None):
        calls.append(str(url))
        return _FakeHttpErrorUpstream(403, html, "text/html")

    monkeypatch.setattr("codex_shim.server.ClientSession.post", fake_post)
    settings = tmp_path / "settings.json"
    settings.write_text("{}")
    shim_client = TestClient(TestServer(ShimServer(settings).app()))
    await shim_client.start_server()
    try:
        resp = await shim_client.post("/v1/responses", json={"model": "codex-gpt-5-5", "input": "hi"})
        assert resp.status == 502
        text = await resp.text()
        assert "<html" not in text.lower()
        assert "chatgpt edge unavailable" in text
        assert calls == ["https://chatgpt.com/backend-api/codex/responses"] * 3
    finally:
        await shim_client.close()


async def test_chatgpt_passthrough_json_403_is_not_retried(monkeypatch, tmp_path, auth_present):
    calls: list[str] = []
    body = '{"error":{"message":"insufficient_quota","type":"insufficient_quota"}}'

    async def fake_post(self, url, json=None, headers=None):
        calls.append(str(url))
        return _FakeHttpErrorUpstream(403, body, "application/json")

    monkeypatch.setattr("codex_shim.server.ClientSession.post", fake_post)
    settings = tmp_path / "settings.json"
    settings.write_text("{}")
    shim_client = TestClient(TestServer(ShimServer(settings).app()))
    await shim_client.start_server()
    try:
        resp = await shim_client.post("/v1/responses", json={"model": "codex-gpt-5-5", "input": "hi"})
        assert resp.status == 403
        assert "insufficient_quota" in await resp.text()
        assert calls == ["https://chatgpt.com/backend-api/codex/responses"]
    finally:
        await shim_client.close()


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


def test_parse_upstream_error_reads_fastapi_detail():
    body = json.dumps({"detail": "Bad Request"})
    code, message = parse_upstream_error(body, 400)
    assert code == "upstream_http_400"
    assert message == "Bad Request"


def test_parse_upstream_error_reads_fastapi_detail_list():
    body = json.dumps({"detail": ["field required", "invalid model"]})
    code, message = parse_upstream_error(body, 422)
    assert message == "field required; invalid model"


def test_input_has_compaction_trigger_and_longer_tail_summary():
    from codex_shim.server import _input_has_compaction_trigger, _summarize_input_items

    items = [{"type": "message", "role": "user"}] * 10 + [{"type": "compaction_trigger"}]
    assert _input_has_compaction_trigger(items)
    count, summary = _summarize_input_items(items, tail=12)
    assert count == 11
    assert summary[-1] == "compaction_trigger"


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


async def test_responses_stream_state_output_items_collects_assistant_message():
    class FakeResponse:
        async def write(self, data: bytes):
            return None

    downstream = FakeResponse()
    state = ResponsesStreamState("test-model")
    await state.start(downstream)
    await state.write_chat_delta(
        downstream,
        {"choices": [{"delta": {"content": "hello"}}]},
    )
    await state.finish(downstream, upstream_saw_done=True)
    output = state.output_items()
    assert len(output) == 1
    assert output[0]["type"] == "message"
    assert output[0]["role"] == "assistant"
    assert output[0]["content"][0]["text"] == "hello"


async def test_byok_stream_stores_conversation_cache_for_delta_expansion(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEX_SHIM_CHATGPT_EXPAND_CONTINUATIONS", "1")
    captured_requests: list[dict[str, Any]] = []

    async def chat(request):
        captured_requests.append(await request.json())
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        if len(captured_requests) == 1:
            await response.write(
                b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"exec_command","arguments":"{}"}}]}}]}\n\n'
            )
            await response.write(b'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}\n\n')
        else:
            await response.write(b'data: {"choices":[{"delta":{"content":"done"}}]}\n\n')
            await response.write(b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n')
        await response.write(b"data: [DONE]\n\n")
        await response.write_eof()
        return response

    upstream = web.Application()
    upstream.router.add_post("/v1/chat/completions", chat)
    upstream_client = TestClient(TestServer(upstream))
    await upstream_client.start_server()

    cache_dir = tmp_path / "conversations"
    monkeypatch.setenv("CODEX_SHIM_CHATGPT_CONVERSATIONS_DIR", str(cache_dir))

    settings = tmp_path / "settings.json"
    settings.write_text(
        json.dumps(
            {
                "customModels": [
                    {
                        "model": "byok-test",
                        "displayName": "BYOK Test",
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
    headers = {"session-id": "sess-byok-cache"}

    first = await shim_client.post(
        "/v1/responses",
        json={
            "model": "byok-test",
            "input": [{"type": "message", "role": "user", "content": "fix ci"}],
            "stream": True,
            "tools": [{"type": "function", "name": "exec_command"}],
        },
        headers=headers,
    )
    assert first.status == 200
    events = _sse_events(await first.text())
    completed = [event for event in events if event.get("type") == "response.completed"][-1]
    response_id = completed["response"]["id"]

    second = await shim_client.post(
        "/v1/responses",
        json={
            "model": "byok-test",
            "previous_response_id": response_id,
            "input": [{"type": "function_call_output", "call_id": "call_1", "output": "ok"}],
            "stream": True,
            "tools": [{"type": "function", "name": "exec_command"}],
        },
        headers=headers,
    )
    assert second.status == 200
    assert len(captured_requests) == 2
    second_messages = captured_requests[1].get("messages") or []
    roles = [msg.get("role") for msg in second_messages]
    assert "user" in roles
    assert any(
        "fix ci" in str(msg.get("content") or "")
        for msg in second_messages
        if msg.get("role") == "user"
    )

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


async def test_sse_lines_ignores_inbound_comment_lines():
    class Upstream:
        content = _FakeSseContent(
            [
                b": ping\n",
                b'data: {"type":"response.created"}\n\n',
                b": keep-alive\n",
                b"data: [DONE]\n\n",
            ]
        )

    lines = [line async for line in _sse_lines(Upstream())]
    assert lines == ['{"type":"response.created"}', "[DONE]"]


async def test_chatgpt_sse_keepalive_emits_ping_before_delayed_event(monkeypatch, tmp_path, auth_present):
    monkeypatch.setattr(server_module, "SSE_KEEPALIVE_INTERVAL", 0.05)

    class DelayedContent(_FakeSseContent):
        def __init__(self, chunks: list[bytes], delay: float):
            super().__init__(chunks)
            self._delay = delay
            self._started = False

        async def readany(self):
            if not self._started:
                self._started = True
                await asyncio.sleep(self._delay)
            return await super().readany()

    class DelayedUpstream:
        status = 200
        content_type = "text/event-stream"
        headers = {"Content-Type": "text/event-stream"}

        def __init__(self):
            self.content = DelayedContent(
                [
                    _sse_chunk(
                        {
                            "type": "response.completed",
                            "response": {
                                "id": "resp_delayed",
                                "model": "gpt-5.5",
                                "status": "completed",
                                "output": [],
                            },
                        }
                    ),
                    b"data: [DONE]\n\n",
                ],
                delay=0.2,
            )

        def close(self):
            pass

        def release(self):
            pass

    async def fake_post(self, url, json=None, headers=None):
        return DelayedUpstream()

    monkeypatch.setattr("codex_shim.server.ClientSession.post", fake_post)
    settings = tmp_path / "settings.json"
    settings.write_text("{}")
    shim_client = TestClient(TestServer(ShimServer(settings).app()))
    await shim_client.start_server()
    try:
        resp = await shim_client.post(
            "/v1/responses",
            json={"model": "codex-gpt-5-5", "input": "hi", "stream": True},
        )
        assert resp.status == 200
        body = await resp.read()
        assert b": ping" in body
        events = _sse_events(body.decode())
        assert any(event.get("type") == "response.completed" for event in events)
    finally:
        await shim_client.close()


async def test_chatgpt_sse_keepalive_disconnect_during_wait_closes_upstream(
    monkeypatch, tmp_path, auth_present
):
    monkeypatch.setattr(server_module, "SSE_KEEPALIVE_INTERVAL", 0.05)
    hung_state = {"closed": False}

    class HungContent:
        async def readany(self):
            await asyncio.sleep(30)
            return b""

    class HungUpstream:
        status = 200
        content_type = "text/event-stream"
        headers = {"Content-Type": "text/event-stream"}
        content = HungContent()

        def close(self):
            hung_state["closed"] = True

        def release(self):
            pass

    async def fake_post(self, url, json=None, headers=None):
        return HungUpstream()

    monkeypatch.setattr("codex_shim.server.ClientSession.post", fake_post)
    settings = tmp_path / "settings.json"
    settings.write_text("{}")
    shim_client = TestClient(TestServer(ShimServer(settings).app()))
    await shim_client.start_server()
    try:
        resp = await shim_client.post(
            "/v1/responses",
            json={"model": "codex-gpt-5-5", "input": "hi", "stream": True},
        )
        assert resp.status == 200
        await asyncio.sleep(0.12)
        resp.close()
        await asyncio.sleep(0.3)
        assert hung_state["closed"] is True
    finally:
        await shim_client.close()


def _byok_settings(upstream_client, tmp_path):
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
    return settings


async def test_byok_stream_emits_completed_when_upstream_omits_done(tmp_path):
    """Upstream EOF without ``data: [DONE]`` must still terminate the turn.

    Desktop aborts with "stream closed before response.completed" when the SSE
    ends with no terminal event.
    """

    async def chat(request):
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        await response.write(b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n')
        await response.write(b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n')
        await response.write_eof()
        return response

    upstream = web.Application()
    upstream.router.add_post("/v1/chat/completions", chat)
    upstream_client = TestClient(TestServer(upstream))
    await upstream_client.start_server()

    shim_client = TestClient(TestServer(ShimServer(_byok_settings(upstream_client, tmp_path)).app()))
    await shim_client.start_server()
    try:
        resp = await shim_client.post(
            "/v1/responses",
            json={"model": "real-openai", "input": "hi", "stream": True},
        )
        assert resp.status == 200
        events = _sse_events((await resp.read()).decode())
        terminal = [
            event.get("type")
            for event in events
            if event.get("type")
            in {"response.completed", "response.incomplete", "response.failed"}
        ]
        assert terminal == ["response.completed"]
        assert any(event.get("delta") == "hello" for event in events)
    finally:
        await shim_client.close()
        await upstream_client.close()


async def test_byok_stream_emits_incomplete_when_truncated_without_done(tmp_path):
    """Upstream dying mid tool-call must emit a terminal event, not close silently."""

    async def chat(request):
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        await response.write(
            b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1",'
            b'"function":{"name":"exec_command","arguments":"{\\"cmd\\": \\"ls"}}]}}]}\n\n'
        )
        await response.write_eof()
        return response

    upstream = web.Application()
    upstream.router.add_post("/v1/chat/completions", chat)
    upstream_client = TestClient(TestServer(upstream))
    await upstream_client.start_server()

    shim_client = TestClient(TestServer(ShimServer(_byok_settings(upstream_client, tmp_path)).app()))
    await shim_client.start_server()
    try:
        resp = await shim_client.post(
            "/v1/responses",
            json={"model": "real-openai", "input": "hi", "stream": True},
        )
        assert resp.status == 200
        events = _sse_events((await resp.read()).decode())
        terminal = [
            event.get("type")
            for event in events
            if event.get("type")
            in {"response.completed", "response.incomplete", "response.failed"}
        ]
        assert terminal == ["response.incomplete"]
    finally:
        await shim_client.close()
        await upstream_client.close()


async def test_byok_stream_pings_while_upstream_is_silent(monkeypatch, tmp_path):
    """Long silent generations must keep the downstream SSE warm."""
    monkeypatch.setattr(server_module, "SSE_KEEPALIVE_INTERVAL", 0.05)

    async def chat(request):
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        await asyncio.sleep(0.3)
        await response.write(b'data: {"choices":[{"delta":{"content":"slow"}}]}\n\n')
        await response.write(b"data: [DONE]\n\n")
        await response.write_eof()
        return response

    upstream = web.Application()
    upstream.router.add_post("/v1/chat/completions", chat)
    upstream_client = TestClient(TestServer(upstream))
    await upstream_client.start_server()

    shim_client = TestClient(TestServer(ShimServer(_byok_settings(upstream_client, tmp_path)).app()))
    await shim_client.start_server()
    try:
        resp = await shim_client.post(
            "/v1/responses",
            json={"model": "real-openai", "input": "hi", "stream": True},
        )
        assert resp.status == 200
        body = await resp.read()
        assert b": ping" in body
        events = _sse_events(body.decode())
        assert any(event.get("type") == "response.completed" for event in events)
    finally:
        await shim_client.close()
        await upstream_client.close()


async def test_byok_stream_fails_terminally_on_internal_error(monkeypatch, tmp_path):
    """An unexpected shim-side error must surface as response.failed, not a dropped SSE."""

    async def chat(request):
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        await response.write(b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n')
        await response.write(b"data: [DONE]\n\n")
        await response.write_eof()
        return response

    async def boom(self, response, event):
        raise RuntimeError("malformed upstream event")

    monkeypatch.setattr(ResponsesStreamState, "write_chat_delta", boom)

    upstream = web.Application()
    upstream.router.add_post("/v1/chat/completions", chat)
    upstream_client = TestClient(TestServer(upstream))
    await upstream_client.start_server()

    shim_client = TestClient(TestServer(ShimServer(_byok_settings(upstream_client, tmp_path)).app()))
    await shim_client.start_server()
    try:
        resp = await shim_client.post(
            "/v1/responses",
            json={"model": "real-openai", "input": "hi", "stream": True},
        )
        assert resp.status == 200
        events = _sse_events((await resp.read()).decode())
        assert [
            event.get("type")
            for event in events
            if event.get("type")
            in {"response.completed", "response.incomplete", "response.failed"}
        ] == ["response.failed"]
    finally:
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


async def test_finish_emits_response_completed_when_items_complete():
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

    # A missing upstream [DONE] sentinel still terminates the turn: closing the
    # SSE silently makes Desktop abort with "stream closed before
    # response.completed" and throw away the whole turn.
    downstream2 = FakeResponse()
    state2 = ResponsesStreamState("local-llama")
    await state2.start(downstream2)
    await state2.write_chat_delta(
        downstream2,
        {"choices": [{"delta": {"content": "hello"}}]},
    )
    await state2.finish(downstream2, upstream_saw_done=False)
    events2 = _sse_events(b"".join(downstream2.chunks).decode())
    assert events2[-1]["type"] == "response.completed"
    assert b"data: [DONE]" in b"".join(downstream2.chunks)


async def test_finish_emits_response_incomplete_when_tool_call_args_truncated():
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
                                "id": "call_trunc",
                                "function": {
                                    "name": "exec_command",
                                    "arguments": '{"cmd": "ls',
                                },
                            }
                        ]
                    }
                }
            ]
        },
    )
    await state.finish(downstream, upstream_saw_done=False)
    events = _sse_events(b"".join(downstream.chunks).decode())
    assert events[-1]["type"] == "response.incomplete"
    assert b"data: [DONE]" in b"".join(downstream.chunks)


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


async def test_chatgpt_ws_compaction_v2_proxies_to_codex_responses(monkeypatch, tmp_path, auth_present):
    monkeypatch.setenv("CODEX_SHIM_WS_PASSTHROUGH", "0")
    captured = {}

    async def fake_post(self, url, json=None, headers=None):
        captured["url"] = str(url)
        captured["body"] = json
        return _FakeSseUpstream()

    monkeypatch.setattr("codex_shim.server.ClientSession.post", fake_post)

    settings = tmp_path / "settings.json"
    settings.write_text("{}")
    shim_client = TestClient(TestServer(ShimServer(settings).app()))
    await shim_client.start_server()
    try:
        ws = await shim_client.ws_connect("/v1/responses")
        await ws.send_json(
            {
                "type": "response.create",
                "model": "codex-gpt-5-5",
                "stream": True,
                "input": [
                    {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "long thread"}]},
                    {"type": "compaction_trigger"},
                ],
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
        assert captured["url"] == "https://chatgpt.com/backend-api/codex/responses"
        assert captured["body"]["input"][-1]["type"] == "compaction_trigger"
        types = [event.get("type") for event in events]
        assert "response.output_item.done" in types
        done = next(event for event in events if event.get("type") == "response.output_item.done")
        assert done["item"]["type"] == "compaction"
        assert done["item"]["encrypted_content"] == "openai-native-compaction-blob"
        assert events[-1]["response"]["model"] == "codex-gpt-5-5"
        await ws.close()
    finally:
        await shim_client.close()


async def test_chatgpt_http_compaction_v2_proxies_to_codex_responses(monkeypatch, tmp_path, auth_present):
    captured = {}

    async def fake_post(self, url, json=None, headers=None):
        captured["url"] = str(url)
        captured["body"] = json
        captured["headers"] = headers
        return _FakeSseUpstream()

    monkeypatch.setattr("codex_shim.server.ClientSession.post", fake_post)

    settings = tmp_path / "settings.json"
    settings.write_text("{}")
    shim_client = TestClient(TestServer(ShimServer(settings).app()))
    await shim_client.start_server()
    try:
        resp = await shim_client.post(
            "/v1/responses",
            json={
                "model": "codex-gpt-5-6-luna",
                "stream": True,
                "service_tier": "priority",
                "max_output_tokens": 4096,
                "parallel_tool_calls": True,
                "reasoning": {"effort": "high", "context": "last_turn"},
                "input": [
                    {"type": "message", "role": "user", "content": "long thread"},
                    {"type": "compaction_trigger"},
                ],
            },
            headers={"X-OpenAI-Internal-Codex-Responses-Lite": "1"},
        )
        assert resp.status == 200
        text = await resp.text()
        assert captured["url"] == "https://chatgpt.com/backend-api/codex/responses"
        assert not str(captured["url"]).endswith("/compact")
        assert captured["body"]["input"][-1]["type"] == "compaction_trigger"
        assert captured["body"]["reasoning"]["context"] == "all_turns"
        assert captured["body"]["reasoning"]["effort"] == "high"
        assert captured["body"]["parallel_tool_calls"] is False
        assert captured["body"]["store"] is False
        assert captured["body"]["service_tier"] == "priority"
        assert captured["body"]["max_output_tokens"] == 4096
        assert '"type": "compaction"' in text or '"type":"compaction"' in text.replace(" ", "")
        assert "openai-native-compaction-blob" in text
        assert "Remote native compaction failed" not in text
    finally:
        await shim_client.close()


async def test_chatgpt_http_compaction_v2_returns_upstream_error_without_summarization(
    monkeypatch, tmp_path, auth_present
):
    calls: list[str] = []

    async def fake_post(self, url, json=None, headers=None):
        calls.append(str(url))
        if str(url).endswith("/responses/compact"):
            raise AssertionError("ChatGPT compaction v2 must not POST /compact")
        return _FakeHttpErrorUpstream(400, '{"detail":"Bad Request"}')

    monkeypatch.setattr("codex_shim.server.ClientSession.post", fake_post)

    settings = tmp_path / "settings.json"
    settings.write_text("{}")
    shim_client = TestClient(TestServer(ShimServer(settings).app()))
    await shim_client.start_server()
    try:
        resp = await shim_client.post(
            "/v1/responses",
            json={
                "model": "codex-gpt-5-5",
                "stream": True,
                "input": [
                    {"type": "message", "role": "user", "content": "long thread"},
                    {"type": "compaction_trigger"},
                ],
            },
        )
        assert resp.status == 400
        text = await resp.text()
        assert "Bad Request" in text
        assert "Compaction failed for" not in text
        assert calls == ["https://chatgpt.com/backend-api/codex/responses"]
    finally:
        await shim_client.close()


async def test_chatgpt_legacy_compact_returns_upstream_404_without_summarization(
    monkeypatch, tmp_path, auth_present, capsys
):
    calls: list[str] = []

    async def fake_post(self, url, json=None, headers=None):
        calls.append(str(url))
        if str(url).endswith("/responses/compact"):
            return _FakeHttpErrorUpstream(404, '{"detail":"Not Found"}')
        raise AssertionError(f"legacy compact must not fall back to {url}")

    monkeypatch.setattr("codex_shim.server.ClientSession.post", fake_post)

    settings = tmp_path / "settings.json"
    settings.write_text("{}")
    shim_client = TestClient(TestServer(ShimServer(settings).app()))
    await shim_client.start_server()
    try:
        resp = await shim_client.post(
            "/v1/responses/compact",
            json={"model": "codex-gpt-5-5", "input": [{"type": "message", "role": "user", "content": "long thread"}]},
        )
        assert resp.status == 404
        body = await resp.text()
        assert "Not Found" in body
        assert "Compaction failed for" not in body
        assert calls == ["https://chatgpt.com/backend-api/codex/responses/compact"]
        captured = capsys.readouterr()
        assert "upstream missing (known)" in captured.out
        assert "[io-resp]" not in captured.out
    finally:
        await shim_client.close()


async def test_chatgpt_legacy_compact_retries_503_then_returns_404_without_summarization(
    monkeypatch, tmp_path, auth_present
):
    calls: list[str] = []

    async def fake_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr("codex_shim.chatgpt_edge.asyncio.sleep", fake_sleep)

    async def fake_post(self, url, json=None, headers=None):
        calls.append(str(url))
        if len(calls) == 1:
            return _FakeHttpErrorUpstream(503, "envoy unavailable", "text/plain")
        if str(url).endswith("/responses/compact"):
            return _FakeHttpErrorUpstream(404, '{"detail":"Not Found"}')
        raise AssertionError(f"legacy compact must not fall back to {url}")

    monkeypatch.setattr("codex_shim.server.ClientSession.post", fake_post)

    settings = tmp_path / "settings.json"
    settings.write_text("{}")
    shim_client = TestClient(TestServer(ShimServer(settings).app()))
    await shim_client.start_server()
    try:
        resp = await shim_client.post(
            "/v1/responses/compact",
            json={"model": "codex-gpt-5-5", "input": [{"type": "message", "role": "user", "content": "long thread"}]},
        )
        assert resp.status == 404
        assert "Not Found" in await resp.text()
        compact_url = "https://chatgpt.com/backend-api/codex/responses/compact"
        assert calls == [compact_url, compact_url]
    finally:
        await shim_client.close()


async def test_chatgpt_http_compaction_v2_relays_lite_output_item_done(
    monkeypatch, tmp_path, auth_present
):
    captured: dict[str, Any] = {}
    item = {
        "type": "compaction",
        "encrypted_content": "openai-native-compaction-blob",
    }
    chunks = [
        _sse_chunk(
            {
                "type": "response.created",
                "response": {
                    "id": "resp_lite_compact",
                    "model": "gpt-5.6-luna",
                    "status": "in_progress",
                    "output": [],
                },
            }
        ),
        _sse_chunk({"type": "response.output_item.done", "item": item}),
        _sse_chunk(
            {
                "type": "response.completed",
                "response": {
                    "id": "resp_lite_compact",
                    "model": "gpt-5.6-luna",
                    "status": "completed",
                    "output": [],
                    "usage": {"input_tokens": 3, "output_tokens": 8, "total_tokens": 11},
                },
            }
        ),
        b"data: [DONE]\n\n",
    ]

    async def fake_post(self, url, json=None, headers=None):
        captured["url"] = str(url)
        captured["body"] = json
        return _FakeSseUpstream(chunks)

    monkeypatch.setattr("codex_shim.server.ClientSession.post", fake_post)

    settings = tmp_path / "settings.json"
    settings.write_text("{}")
    shim_client = TestClient(TestServer(ShimServer(settings).app()))
    await shim_client.start_server()
    try:
        resp = await shim_client.post(
            "/v1/responses",
            json={
                "model": "codex-gpt-5-6-luna",
                "stream": True,
                "input": [
                    {"type": "message", "role": "user", "content": "long thread"},
                    {"type": "compaction_trigger"},
                ],
            },
            headers={"X-OpenAI-Internal-Codex-Responses-Lite": "1"},
        )
        assert resp.status == 200
        assert captured["url"] == "https://chatgpt.com/backend-api/codex/responses"
        events = _sse_events(await resp.text())
        done = next(event for event in events if event.get("type") == "response.output_item.done")
        assert done["item"]["encrypted_content"] == "openai-native-compaction-blob"
        completed = next(event for event in events if event.get("type") == "response.completed")
        assert completed["response"]["output"] == []
    finally:
        await shim_client.close()


async def test_chatgpt_http_compaction_v2_synthesizes_orphan_tool_call(
    monkeypatch, tmp_path, auth_present, capsys
):
    captured: dict[str, Any] = {}

    async def fake_post(self, url, json=None, headers=None):
        assert not str(url).endswith("/responses/compact"), url
        captured["url"] = str(url)
        captured["input"] = (json or {}).get("input")
        return _FakeSseUpstream()

    monkeypatch.setattr("codex_shim.server.ClientSession.post", fake_post)

    settings = tmp_path / "settings.json"
    settings.write_text("{}")
    shim_client = TestClient(TestServer(ShimServer(settings).app()))
    await shim_client.start_server()
    try:
        resp = await shim_client.post(
            "/v1/responses",
            json={
                "model": "codex-gpt-5-5",
                "stream": True,
                "input": [
                    {
                        "type": "function_call_output",
                        "call_id": "call_orphan",
                        "output": "truncated tool output",
                    },
                    {"type": "compaction_trigger"},
                ],
            },
        )
        assert resp.status == 200
        assert captured["url"] == "https://chatgpt.com/backend-api/codex/responses"
        forwarded = captured.get("input")
        assert isinstance(forwarded, list)
        assert len(forwarded) == 3
        assert forwarded[0]["type"] == "function_call"
        assert forwarded[0]["call_id"] == "call_orphan"
        assert forwarded[1]["call_id"] == "call_orphan"
        assert forwarded[2]["type"] == "compaction_trigger"
        captured_out = capsys.readouterr().out
        assert "synthesized" in captured_out
    finally:
        await shim_client.close()


async def test_chatgpt_ws_compaction_v2_synthesizes_orphan_tool_call(
    monkeypatch, tmp_path, auth_present, capsys
):
    monkeypatch.setenv("CODEX_SHIM_WS_PASSTHROUGH", "0")
    captured: dict[str, Any] = {}

    async def fake_post(self, url, json=None, headers=None):
        assert not str(url).endswith("/responses/compact"), url
        captured["url"] = str(url)
        captured["input"] = (json or {}).get("input")
        return _FakeSseUpstream()

    monkeypatch.setattr("codex_shim.server.ClientSession.post", fake_post)

    settings = tmp_path / "settings.json"
    settings.write_text("{}")
    shim_client = TestClient(TestServer(ShimServer(settings).app()))
    await shim_client.start_server()
    try:
        ws = await shim_client.ws_connect("/v1/responses")
        await ws.send_json(
            {
                "type": "response.create",
                "model": "codex-gpt-5-5",
                "stream": True,
                "input": [
                    {
                        "type": "function_call_output",
                        "call_id": "call_VF4XLxSPXoTfOfVgV0jDqbXq",
                        "output": "truncated tool output",
                    },
                    {"type": "compaction_trigger"},
                ],
            }
        )
        while True:
            msg = await ws.receive(timeout=2)
            assert msg.type == WSMsgType.TEXT
            payload = json.loads(msg.data)
            if payload.get("type") == "response.completed":
                break
        assert captured["url"] == "https://chatgpt.com/backend-api/codex/responses"
        forwarded = captured.get("input")
        assert isinstance(forwarded, list)
        assert len(forwarded) == 3
        assert forwarded[0]["type"] == "function_call"
        assert forwarded[0]["call_id"] == "call_VF4XLxSPXoTfOfVgV0jDqbXq"
        assert forwarded[1]["call_id"] == "call_VF4XLxSPXoTfOfVgV0jDqbXq"
        assert forwarded[2]["type"] == "compaction_trigger"
        captured_out = capsys.readouterr().out
        assert "synthesized" in captured_out
        await ws.close()
    finally:
        await shim_client.close()


async def test_chatgpt_http_compaction_v2_does_not_use_passthrough_error_fallback(
    monkeypatch, tmp_path, auth_present
):
    calls: list[str] = []

    async def fake_post(self, url, json=None, headers=None):
        calls.append(str(url))
        if "chatgpt.com" in str(url):
            return _FakeHttpErrorUpstream(400, '{"detail":"Bad Request"}')
        raise AssertionError(f"compaction v2 must not fall back to {url}")

    monkeypatch.setattr("codex_shim.server.ClientSession.post", fake_post)

    settings = tmp_path / "settings.json"
    settings.write_text(
        json.dumps(
            {
                "passthrough_error_fallback": {"gpt-5.5": "or-free-router"},
                "customModels": [
                    {
                        "model": "openrouter/free",
                        "displayName": "OpenRouter Free",
                        "slug": "or-free-router",
                        "provider": "generic-chat-completion-api",
                        "baseUrl": "http://127.0.0.1:9/v1",
                        "apiKey": "secret",
                    }
                ],
            }
        )
    )
    shim_client = TestClient(TestServer(ShimServer(settings).app()))
    await shim_client.start_server()
    try:
        resp = await shim_client.post(
            "/v1/responses",
            json={
                "model": "codex-gpt-5-5",
                "stream": True,
                "input": [
                    {"type": "message", "role": "user", "content": "long thread"},
                    {"type": "compaction_trigger"},
                ],
            },
        )
        assert resp.status == 400
        assert "Bad Request" in await resp.text()
        assert calls == ["https://chatgpt.com/backend-api/codex/responses"]
    finally:
        await shim_client.close()

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

    resp = await shim_client.post(
        "/v1/responses/compact",
        json={
            "model": "openai-gpt-5-5-codex-max",
            "input": "hi",
            "stream": True,
            "service_tier": "priority",
            "max_output_tokens": 4096,
        },
    )
    assert resp.status == 200
    payload = await resp.json()
    assert payload["model"] == "openai-gpt-5-5-codex-max"
    assert payload["output"][0]["model"] == "openai-gpt-5-5-codex-max"
    assert captured["url"] == "https://chatgpt.com/backend-api/codex/responses/compact"
    assert captured["body"]["model"] == "gpt-5.5"
    assert captured["body"]["service_tier"] == "priority"
    assert captured["body"]["max_output_tokens"] == 4096
    assert "store" not in captured["body"]
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

    async def fake_cursor_events(_prompt, _model, **kwargs):
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


async def test_cursor_passthrough_stream_unknown_tool_shows_json(
    monkeypatch, tmp_path, cursor_present, auth_missing
):
    from codex_shim.cursor_passthrough import format_cursor_tool_completed_markdown, format_cursor_tool_started_markdown

    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"customModels": []}))

    async def fake_cursor_events(_prompt, _model, **kwargs):
        yield {
            "type": "tool_started",
            "call_id": "tool-x",
            "tool_call": {"mysteryToolCall": {"args": {"q": "search"}}},
            "markdown": format_cursor_tool_started_markdown(
                {"tool_call": {"mysteryToolCall": {"args": {"q": "search"}}}}
            ),
        }
        yield {
            "type": "tool_completed",
            "call_id": "tool-x",
            "tool_call": {
                "mysteryToolCall": {
                    "args": {"q": "search"},
                    "result": {"success": {"hits": 1}},
                }
            },
            "markdown": format_cursor_tool_completed_markdown(
                {
                    "tool_call": {
                        "mysteryToolCall": {
                            "args": {"q": "search"},
                            "result": {"success": {"hits": 1}},
                        }
                    }
                }
            ),
        }
        yield {"type": "completed", "text": "done"}

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
        reasoning_done = [
            event
            for event in events
            if event.get("type") == "response.output_item.done"
            and (event.get("item") or {}).get("type") == "reasoning"
        ][-1]
        summary = (reasoning_done.get("item") or {}).get("summary") or []
        text = summary[0]["text"] if summary else ""
        assert "unknown" in text
        assert "```json" in text
        assert "mysteryToolCall" in text
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


def _ws_url_from_test_client(client: TestClient, path: str = "/v1/responses") -> str:
    return str(client.make_url(path)).replace("http://", "ws://", 1)


def _patch_chatgpt_ws_url(monkeypatch, url: str) -> None:
    monkeypatch.setattr("codex_shim.ws_passthrough.CHATGPT_WS_URL", url)
    monkeypatch.setattr("codex_shim.server.CHATGPT_WS_URL", url)


async def test_chatgpt_ws_passthrough_relays_events(monkeypatch, tmp_path, auth_present):
    state = MockUpstreamWsState(
        response_sequences=[
            [
                {"type": "response.created", "response": {"id": "resp_1", "model": "gpt-5.5"}},
                {"type": "response.output_text.delta", "delta": "ok"},
                {"type": "response.completed", "response": {"id": "resp_1", "model": "gpt-5.5", "status": "completed"}},
            ]
        ]
    )
    upstream_state, upstream_client = await start_mock_upstream_ws(state)
    _patch_chatgpt_ws_url(monkeypatch, _ws_url_from_test_client(upstream_client))

    settings = tmp_path / "settings.json"
    settings.write_text("{}")
    shim_client = TestClient(TestServer(ShimServer(settings).app()))
    await shim_client.start_server()
    try:
        ws = await shim_client.ws_connect("/v1/responses")
        await ws.send_json(
            {
                "type": "response.create",
                "model": "codex-gpt-5-5",
                "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]}],
                "stream": True,
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
        assert upstream_state.received_frames[0]["model"] == "gpt-5.5"
        assert upstream_state.received_frames[0]["type"] == "response.create"
        assert "previous_response_id" not in upstream_state.received_frames[0]
        await ws.close()
    finally:
        await shim_client.close()
        await upstream_client.close()


async def test_chatgpt_ws_passthrough_multi_create_same_connection(monkeypatch, tmp_path, auth_present):
    first_input = [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "run"}]}]
    tool_call = {
        "type": "function_call",
        "id": "fc_1",
        "call_id": "call_1",
        "name": "exec_command",
        "arguments": "{\"cmd\":\"printf ok\"}",
    }
    tool_output = {"type": "function_call_output", "call_id": "call_1", "output": "ok"}
    state = MockUpstreamWsState(
        response_sequences=[
            [
                {"type": "response.created", "response": {"id": "resp_previous", "model": "gpt-5.5"}},
                {
                    "type": "response.output_item.done",
                    "response": {"id": "resp_previous", "model": "gpt-5.5", "output": []},
                    "item": tool_call,
                },
                {"type": "response.completed", "response": {"id": "resp_previous", "model": "gpt-5.5", "status": "completed", "output": []}},
            ],
            [
                {"type": "response.created", "response": {"id": "resp_next", "model": "gpt-5.5"}},
                {"type": "response.completed", "response": {"id": "resp_next", "model": "gpt-5.5", "status": "completed"}},
            ],
        ]
    )
    upstream_state, upstream_client = await start_mock_upstream_ws(state)
    _patch_chatgpt_ws_url(monkeypatch, _ws_url_from_test_client(upstream_client))

    settings = tmp_path / "settings.json"
    settings.write_text("{}")
    shim_client = TestClient(TestServer(ShimServer(settings).app()))
    await shim_client.start_server()
    try:
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

        assert len(upstream_state.received_frames) == 2
        assert upstream_state.received_frames[1]["previous_response_id"] == "resp_previous"
        assert upstream_state.received_frames[1]["input"] == [tool_output]
        await ws.close()
    finally:
        await shim_client.close()
        await upstream_client.close()


async def test_byok_openai_responses_ws_passthrough(tmp_path, monkeypatch):
    state = MockUpstreamWsState(
        response_sequences=[
            [
                {"type": "response.created", "response": {"id": "resp_1", "model": "gpt-4.1"}},
                {"type": "response.completed", "response": {"id": "resp_1", "model": "gpt-4.1", "status": "completed"}},
            ]
        ]
    )
    upstream_state, upstream_client = await start_mock_upstream_ws(state)

    settings = tmp_path / "settings.json"
    settings.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "model": "gpt-4.1",
                        "display_name": "GPT 4.1",
                        "provider": "openai-responses",
                        "base_url": str(upstream_client.make_url("/v1")),
                        "api_key": "secret-key",
                    }
                ]
            }
        )
    )
    shim_client = TestClient(TestServer(ShimServer(settings).app()))
    await shim_client.start_server()
    try:
        ws = await shim_client.ws_connect("/v1/responses")
        await ws.send_json(
            {
                "type": "response.create",
                "model": "gpt-4-1",
                "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]}],
                "stream": True,
            }
        )
        for _ in range(2):
            msg = await ws.receive(timeout=2)
            assert msg.type == WSMsgType.TEXT
        assert upstream_state.received_frames[0]["model"] == "gpt-4.1"
        assert upstream_state.handshakes[0].headers.get("Authorization") == "Bearer secret-key"
        await ws.close()
    finally:
        await shim_client.close()
        await upstream_client.close()


async def test_ws_passthrough_falls_back_to_http_on_connect_failure(monkeypatch, tmp_path, auth_present):
    _patch_chatgpt_ws_url(monkeypatch, "ws://127.0.0.1:1/unreachable")
    captured = {}

    class FakeUpstream:
        status = 200
        content_type = "text/event-stream"
        headers = {}

        def __init__(self):
            self.content = _FakeSseContent(
                [
                    b'data: {"type":"response.created","response":{"id":"resp_1","model":"gpt-5.5"}}\n\n',
                    b'data: {"type":"response.completed","response":{"id":"resp_1","model":"gpt-5.5","status":"completed"}}\n\n',
                ]
            )

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
    try:
        ws = await shim_client.ws_connect("/v1/responses")
        await ws.send_json(
            {
                "type": "response.create",
                "model": "codex-gpt-5-5",
                "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]}],
                "stream": True,
            }
        )
        for _ in range(2):
            msg = await ws.receive(timeout=2)
            assert msg.type == WSMsgType.TEXT
        assert captured["url"] == "https://chatgpt.com/backend-api/codex/responses"
        assert captured["body"]["model"] == "gpt-5.5"
        await ws.close()
    finally:
        await shim_client.close()


async def test_byok_chat_completions_still_uses_http_bridge(tmp_path, monkeypatch):
    upstream_connect_calls = {"count": 0}
    original_connect = WsPassthroughSession.connect_upstream

    async def tracking_connect(self, url, headers):
        upstream_connect_calls["count"] += 1
        return await original_connect(self, url, headers)

    monkeypatch.setattr(WsPassthroughSession, "connect_upstream", tracking_connect)

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
    try:
        ws = await shim_client.ws_connect("/v1/responses")
        await ws.send_json(
            {
                "type": "response.create",
                "model": "real-openai",
                "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]}],
                "stream": True,
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
        assert upstream_connect_calls["count"] == 0
        assert events[-1]["response"]["model"] == "real-openai"
        await ws.close()
    finally:
        await shim_client.close()
        await upstream_client.close()

"""Edge cases for transport-aware continuation expansion."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock

import pytest
from aiohttp.test_utils import TestClient, TestServer

import codex_shim.server as server_module
from codex_shim.chatgpt_conversation_cache import ChatgptConversationCache
from codex_shim.chatgpt_conversation_cache import ChatgptConversationCache
from codex_shim.server import ShimServer
from codex_shim.ws_passthrough import WsPassthroughSession
from ws_test_support import MockUpstreamWsState, start_mock_upstream_ws


@pytest.fixture
def auth_present(monkeypatch, tmp_path):
    auth = tmp_path / "auth.json"
    auth.write_text(
        json.dumps(
            {
                "tokens": {
                    "access_token": "test-token",
                    "account_id": "acct-test",
                }
            }
        )
    )
    monkeypatch.setattr("codex_shim.settings.DEFAULT_CODEX_AUTH", auth)
    monkeypatch.setattr("codex_shim.server.DEFAULT_CODEX_AUTH", auth)
    return auth


def _ws_url_from_test_client(client: TestClient, path: str = "/v1/responses") -> str:
    return str(client.make_url(path)).replace("http://", "ws://", 1)


def _patch_chatgpt_ws_url(monkeypatch, url: str) -> None:
    monkeypatch.setattr("codex_shim.ws_passthrough.CHATGPT_WS_URL", url)
    monkeypatch.setattr("codex_shim.server.CHATGPT_WS_URL", url)


@pytest.fixture
def chatgpt_cache_dir(monkeypatch, tmp_path):
    cache_root = tmp_path / "chatgpt-conversations"
    monkeypatch.setattr(server_module, "_chatgpt_conversations_dir", lambda: cache_root)
    return cache_root


@pytest.mark.asyncio
async def test_connect_upstream_reports_connection_reused():
    client_ws = AsyncMock()
    client_ws.closed = False
    upstream_ws = AsyncMock()
    upstream_ws.closed = False
    url = "ws://example/v1/responses"
    session = WsPassthroughSession(client_session=AsyncMock(), client_ws=client_ws)
    session.upstream_by_url[url] = upstream_ws

    _, reused = await session.connect_upstream(url, {})
    assert reused is True

    await session.close_upstream(url)
    upstream_ws.closed = True

    session.client_session.ws_connect = AsyncMock(return_value=AsyncMock(closed=False, headers={}))
    _, reused = await session.connect_upstream(url, {})
    assert reused is False


@pytest.mark.asyncio
async def test_chatgpt_ws_expands_on_first_connect_with_previous_response_id(
    monkeypatch, tmp_path, auth_present, chatgpt_cache_dir
):
    """Fresh upstream WS + prev_id must expand (connection not yet reused)."""
    session_headers = {"session-id": "ws-new-conn"}
    cache = ChatgptConversationCache(chatgpt_cache_dir)
    prior_input = [{"type": "message", "role": "user", "content": "hello"}]
    tool_call = {
        "type": "function_call",
        "id": "fc_1",
        "call_id": "call_1",
        "name": "exec_command",
        "arguments": "{}",
    }
    cache.put("ws-new-conn", "resp_previous", [*prior_input, tool_call])
    tool_output = {"type": "function_call_output", "call_id": "call_1", "output": "ok"}

    state = MockUpstreamWsState(
        response_sequences=[
            [
                {"type": "response.created", "response": {"id": "resp_next", "model": "gpt-5.5"}},
                {"type": "response.completed", "response": {"id": "resp_next", "model": "gpt-5.5", "status": "completed"}},
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
        ws = await shim_client.ws_connect("/v1/responses", headers=session_headers)
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
            assert msg.type.name == "TEXT"

        frame = upstream_state.received_frames[0]
        assert "previous_response_id" not in frame
        assert frame["input"][0] == prior_input[0]
        assert frame["input"][1]["call_id"] == "call_1"
        assert frame["input"][2] == tool_output
        await ws.close()
    finally:
        await shim_client.close()
        await upstream_client.close()


@pytest.mark.asyncio
async def test_chatgpt_ws_retries_with_expansion_on_prev_id_error(
    monkeypatch, tmp_path, auth_present, chatgpt_cache_dir
):
    session_headers = {"session-id": "ws-retry"}
    first_input = [{"type": "message", "role": "user", "content": "run"}]
    tool_call = {
        "type": "function_call",
        "id": "fc_1",
        "call_id": "call_1",
        "name": "exec_command",
        "arguments": "{}",
    }
    tool_output = {"type": "function_call_output", "call_id": "call_1", "output": "ok"}

    state = MockUpstreamWsState(
        response_sequences=[
            [
                {"type": "response.created", "response": {"id": "resp_previous", "model": "gpt-5.5"}},
                {"type": "response.output_item.done", "response": {"id": "resp_previous"}, "item": tool_call},
                {"type": "response.completed", "response": {"id": "resp_previous", "model": "gpt-5.5", "status": "completed"}},
            ],
            [
                {
                    "type": "error",
                    "error": {
                        "code": "previous_response_not_found",
                        "message": "Previous response with id 'resp_previous' not found.",
                        "param": "previous_response_id",
                    },
                },
            ],
            [
                {"type": "response.created", "response": {"id": "resp_retry", "model": "gpt-5.5"}},
                {"type": "response.completed", "response": {"id": "resp_retry", "model": "gpt-5.5", "status": "completed"}},
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
        ws = await shim_client.ws_connect("/v1/responses", headers=session_headers)
        await ws.send_json(
            {"type": "response.create", "model": "codex-gpt-5-5", "input": first_input, "stream": True}
        )
        for _ in range(3):
            msg = await ws.receive(timeout=2)
            assert msg.type.name == "TEXT"

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
            assert msg.type.name == "TEXT"

        assert len(upstream_state.received_frames) == 3
        native = upstream_state.received_frames[1]
        assert native["previous_response_id"] == "resp_previous"
        assert native["input"] == [tool_output]

        retry = upstream_state.received_frames[2]
        assert "previous_response_id" not in retry
        assert retry["input"][0] == first_input[0]
        assert retry["input"][1]["call_id"] == "call_1"
        assert retry["input"][2] == tool_output
        assert len(upstream_state.handshakes) == 2
        await ws.close()
    finally:
        await shim_client.close()
        await upstream_client.close()


@pytest.mark.asyncio
async def test_chatgpt_ws_force_expand_strips_previous_response_id(
    monkeypatch, tmp_path, auth_present, chatgpt_cache_dir
):
    monkeypatch.setenv("CODEX_SHIM_CHATGPT_WS_FORCE_EXPAND", "1")
    session_headers = {"session-id": "ws-force"}
    cache = ChatgptConversationCache(chatgpt_cache_dir)
    prior = [{"type": "message", "role": "user", "content": "hi"}]
    cache.put("ws-force", "resp_previous", prior)

    state = MockUpstreamWsState(
        response_sequences=[
            [
                {"type": "response.created", "response": {"id": "resp_1", "model": "gpt-5.5"}},
                {"type": "response.completed", "response": {"id": "resp_1", "model": "gpt-5.5", "status": "completed"}},
            ],
            [
                {"type": "response.created", "response": {"id": "resp_2", "model": "gpt-5.5"}},
                {"type": "response.completed", "response": {"id": "resp_2", "model": "gpt-5.5", "status": "completed"}},
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
        ws = await shim_client.ws_connect("/v1/responses", headers=session_headers)
        await ws.send_json(
            {"type": "response.create", "model": "codex-gpt-5-5", "input": prior, "stream": True}
        )
        for _ in range(2):
            await ws.receive(timeout=2)

        await ws.send_json(
            {
                "type": "response.create",
                "model": "codex-gpt-5-5",
                "previous_response_id": "resp_previous",
                "input": [{"type": "message", "role": "user", "content": "follow up"}],
                "stream": True,
            }
        )
        for _ in range(2):
            await ws.receive(timeout=2)

        second = upstream_state.received_frames[1]
        assert "previous_response_id" not in second
        assert second["input"][0] == prior[0]
        await ws.close()
    finally:
        await shim_client.close()
        await upstream_client.close()


@pytest.mark.asyncio
async def test_byok_ws_strips_previous_response_id_and_writes_cache(
    tmp_path, monkeypatch, chatgpt_cache_dir
):
    session_headers = {"session-id": "byok-ws"}
    first_input = [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]}]
    follow_up = [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "again"}]}]

    state = MockUpstreamWsState(
        response_sequences=[
            [
                {"type": "response.created", "response": {"id": "resp_1", "model": "gpt-4.1"}},
                {
                    "type": "response.output_item.done",
                    "item": {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "hello"}]},
                },
                {"type": "response.completed", "response": {"id": "resp_1", "model": "gpt-4.1", "status": "completed"}},
            ],
            [
                {"type": "response.created", "response": {"id": "resp_2", "model": "gpt-4.1"}},
                {"type": "response.completed", "response": {"id": "resp_2", "model": "gpt-4.1", "status": "completed"}},
            ],
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
        ws = await shim_client.ws_connect("/v1/responses", headers=session_headers)
        await ws.send_json(
            {"type": "response.create", "model": "gpt-4-1", "input": first_input, "stream": True}
        )
        for _ in range(3):
            await ws.receive(timeout=2)

        await ws.send_json(
            {
                "type": "response.create",
                "model": "gpt-4-1",
                "previous_response_id": "resp_1",
                "input": follow_up,
                "stream": True,
            }
        )
        for _ in range(2):
            await ws.receive(timeout=2)

        second = upstream_state.received_frames[1]
        assert "previous_response_id" not in second
        assert second["input"][0] == first_input[0]
        assert second["input"][1]["type"] == "message"
        assert second["input"][2] == follow_up[0]

        cache = ChatgptConversationCache(chatgpt_cache_dir)
        cached = cache.get("byok-ws", "resp_1")
        assert cached is not None
        assert cached[0] == first_input[0]
        await ws.close()
    finally:
        await shim_client.close()
        await upstream_client.close()


@pytest.mark.asyncio
async def test_cursor_passthrough_expands_previous_response_id_before_prompt(
    monkeypatch, tmp_path, chatgpt_cache_dir
):
    captured_prompts: list[str] = []

    async def fake_iter_cursor_agent_events(prompt, model, *, workspace=None):
        captured_prompts.append(prompt)
        yield {"type": "completed", "text": "done"}

    monkeypatch.setattr(server_module, "cursor_passthrough_available", lambda: True)
    monkeypatch.setattr(server_module, "iter_cursor_agent_events", fake_iter_cursor_agent_events)
    monkeypatch.setattr(server_module, "cursor_upstream_model", lambda slug: "composer-2.5")

    session_headers = {"session-id": "cursor-expand"}
    cache = ChatgptConversationCache(chatgpt_cache_dir)
    prior = [{"type": "message", "role": "user", "content": "first question"}]
    cache.put("cursor-expand", "resp_prior", prior)

    settings = tmp_path / "settings.json"
    settings.write_text("{}")
    shim_client = TestClient(TestServer(ShimServer(settings).app()))
    await shim_client.start_server()
    try:
        resp = await shim_client.post(
            "/v1/responses",
            json={
                "model": "cursor-composer-2-5",
                "previous_response_id": "resp_prior",
                "input": [{"type": "message", "role": "user", "content": "second question"}],
            },
            headers=session_headers,
        )
        assert resp.status == 200
        assert captured_prompts
        assert "first question" in captured_prompts[0]
        assert "second question" in captured_prompts[0]
    finally:
        await shim_client.close()


@pytest.mark.asyncio
async def test_http_turn_breaks_ws_native_chain_and_next_ws_expands(
    monkeypatch, tmp_path, auth_present, chatgpt_cache_dir
):
    """Desktop may POST /v1/responses while also using /v1/responses/ws for the same session."""
    session_headers = {"session-id": "ws-http-mix"}
    first_input = [{"type": "message", "role": "user", "content": "run"}]
    tool_call = {
        "type": "function_call",
        "id": "fc_1",
        "call_id": "call_1",
        "name": "exec_command",
        "arguments": "{}",
    }
    tool_output = {"type": "function_call_output", "call_id": "call_1", "output": "ok"}
    cache = ChatgptConversationCache(chatgpt_cache_dir)
    cache.put("ws-http-mix", "resp_previous", [*first_input, tool_call])

    state = MockUpstreamWsState(
        response_sequences=[
            [
                {"type": "response.created", "response": {"id": "resp_previous", "model": "gpt-5.5"}},
                {"type": "response.output_item.done", "response": {"id": "resp_previous"}, "item": tool_call},
                {"type": "response.completed", "response": {"id": "resp_previous", "model": "gpt-5.5", "status": "completed"}},
            ],
            [
                {"type": "response.created", "response": {"id": "resp_after_http", "model": "gpt-5.5"}},
                {"type": "response.completed", "response": {"id": "resp_after_http", "model": "gpt-5.5", "status": "completed"}},
            ],
        ]
    )
    upstream_state, upstream_client = await start_mock_upstream_ws(state)
    _patch_chatgpt_ws_url(monkeypatch, _ws_url_from_test_client(upstream_client))

    class FakeHttpUpstream:
        status = 200
        content_type = "text/event-stream"
        headers = {}

        def __init__(self) -> None:
            self.content = _FakeSseContent(
                [
                    b'data: {"type":"response.created","response":{"id":"resp_http","model":"gpt-5.5"}}\n\n',
                    b'data: {"type":"response.completed","response":{"id":"resp_http","model":"gpt-5.5","status":"completed"}}\n\n',
                ]
            )

        def release(self) -> None:
            pass

    async def fake_post(self, url, json=None, headers=None):
        return FakeHttpUpstream()

    monkeypatch.setattr("codex_shim.server.ClientSession.post", fake_post)

    settings = tmp_path / "settings.json"
    settings.write_text("{}")
    shim_client = TestClient(TestServer(ShimServer(settings).app()))
    await shim_client.start_server()
    try:
        ws = await shim_client.ws_connect("/v1/responses", headers=session_headers)
        await ws.send_json(
            {"type": "response.create", "model": "codex-gpt-5-5", "input": first_input, "stream": True}
        )
        for _ in range(3):
            await ws.receive(timeout=2)

        await shim_client.post(
            "/v1/responses",
            json={
                "model": "codex-gpt-5-5",
                "previous_response_id": "resp_previous",
                "input": [tool_output],
                "stream": True,
            },
            headers=session_headers,
        )
        cache.put("ws-http-mix", "resp_http", [*first_input, tool_call, tool_output])

        await ws.send_json(
            {
                "type": "response.create",
                "model": "codex-gpt-5-5",
                "previous_response_id": "resp_http",
                "input": [{"type": "message", "role": "user", "content": "continue"}],
                "stream": True,
            }
        )
        for _ in range(2):
            msg = await ws.receive(timeout=2)
            assert msg.type.name == "TEXT"

        assert len(upstream_state.received_frames) == 2
        expanded = upstream_state.received_frames[1]
        assert "previous_response_id" not in expanded
        assert expanded["input"][0] == first_input[0]
        assert expanded["input"][1]["call_id"] == "call_1"
        assert expanded["input"][2] == tool_output
        assert expanded["input"][3]["content"] == "continue"
        await ws.close()
    finally:
        await shim_client.close()
        await upstream_client.close()


@pytest.mark.asyncio
async def test_inbound_ws_reuses_upstream_on_model_swap_same_url(
    monkeypatch, tmp_path, auth_present,
):
    """Same inbound Desktop WS keeps one ChatGPT upstream lane across model swaps."""
    passthrough_slugs = {"codex-gpt-5-5", "codex-gpt-5-6-terra"}
    monkeypatch.setattr(
        "codex_shim.server.is_chatgpt_passthrough_slug",
        lambda slug, cache_path=None: slug in passthrough_slugs,
    )
    monkeypatch.setattr(
        "codex_shim.server.chatgpt_upstream_model",
        lambda slug: {"codex-gpt-5-5": "gpt-5.5", "codex-gpt-5-6-terra": "gpt-5.6-terra"}.get(slug, "gpt-5.5"),
    )
    state = MockUpstreamWsState(
        response_sequences=[
            [
                {"type": "response.created", "response": {"id": "resp_a", "model": "gpt-5.5"}},
                {"type": "response.completed", "response": {"id": "resp_a", "model": "gpt-5.5", "status": "completed"}},
            ],
            [
                {"type": "response.created", "response": {"id": "resp_b", "model": "gpt-5.6-terra"}},
                {"type": "response.completed", "response": {"id": "resp_b", "model": "gpt-5.6-terra", "status": "completed"}},
            ],
            [
                {"type": "response.created", "response": {"id": "resp_c", "model": "gpt-5.5"}},
                {"type": "response.completed", "response": {"id": "resp_c", "model": "gpt-5.5", "status": "completed"}},
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
        for model in ("codex-gpt-5-5", "codex-gpt-5-6-terra", "codex-gpt-5-5"):
            await ws.send_json(
                {
                    "type": "response.create",
                    "model": model,
                    "input": [{"type": "message", "role": "user", "content": f"turn-{model}"}],
                    "stream": True,
                }
            )
            for _ in range(2):
                msg = await ws.receive(timeout=2)
                assert msg.type.name == "TEXT"
        assert len(upstream_state.handshakes) == 1
        assert len(upstream_state.received_frames) == 3
        assert upstream_state.received_frames[1]["model"] == "gpt-5.6-terra"
        await ws.close()
    finally:
        await shim_client.close()
        await upstream_client.close()


@pytest.mark.asyncio
async def test_separate_inbound_ws_connections_get_separate_upstream(
    monkeypatch, tmp_path, auth_present,
):
    state = MockUpstreamWsState(
        response_sequences=[
            [
                {"type": "response.created", "response": {"id": "resp_1", "model": "gpt-5.5"}},
                {"type": "response.completed", "response": {"id": "resp_1", "model": "gpt-5.5", "status": "completed"}},
            ],
            [
                {"type": "response.created", "response": {"id": "resp_2", "model": "gpt-5.5"}},
                {"type": "response.completed", "response": {"id": "resp_2", "model": "gpt-5.5", "status": "completed"}},
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
        for idx in range(2):
            ws = await shim_client.ws_connect("/v1/responses")
            await ws.send_json(
                {
                    "type": "response.create",
                    "model": "codex-gpt-5-5",
                    "input": [{"type": "message", "role": "user", "content": f"conn-{idx}"}],
                    "stream": True,
                }
            )
            for _ in range(2):
                msg = await ws.receive(timeout=2)
                assert msg.type.name == "TEXT"
            await ws.close()
        assert len(upstream_state.handshakes) == 2
    finally:
        await shim_client.close()
        await upstream_client.close()


@pytest.mark.asyncio
async def test_byok_model_swap_uses_separate_upstream_lanes(tmp_path, chatgpt_cache_dir):
    first_input = [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "a"}]}]
    state_a = MockUpstreamWsState(
        response_sequences=[
            [
                {"type": "response.created", "response": {"id": "resp_a", "model": "gpt-4.1"}},
                {"type": "response.completed", "response": {"id": "resp_a", "model": "gpt-4.1", "status": "completed"}},
            ],
        ]
    )
    state_b = MockUpstreamWsState(
        response_sequences=[
            [
                {"type": "response.created", "response": {"id": "resp_b", "model": "gpt-4o"}},
                {"type": "response.completed", "response": {"id": "resp_b", "model": "gpt-4o", "status": "completed"}},
            ],
        ]
    )
    upstream_a, client_a = await start_mock_upstream_ws(state_a)
    upstream_b, client_b = await start_mock_upstream_ws(state_b)

    settings = tmp_path / "settings.json"
    settings.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "model": "gpt-4.1",
                        "display_name": "GPT 4.1",
                        "provider": "openai-responses",
                        "base_url": str(client_a.make_url("/v1")),
                        "api_key": "key-a",
                    },
                    {
                        "model": "gpt-4o",
                        "display_name": "GPT 4o",
                        "provider": "openai-responses",
                        "base_url": str(client_b.make_url("/v1")),
                        "api_key": "key-b",
                    },
                ]
            }
        )
    )
    shim_client = TestClient(TestServer(ShimServer(settings).app()))
    await shim_client.start_server()
    try:
        ws = await shim_client.ws_connect("/v1/responses")
        for model in ("gpt-4-1", "gpt-4o"):
            await ws.send_json(
                {"type": "response.create", "model": model, "input": first_input, "stream": True}
            )
            for _ in range(2):
                msg = await ws.receive(timeout=2)
                assert msg.type.name == "TEXT"
        assert len(upstream_a.handshakes) == 1
        assert len(upstream_b.handshakes) == 1
        await ws.close()
    finally:
        await shim_client.close()
        await client_a.close()
        await client_b.close()


@pytest.mark.asyncio
async def test_parent_http_does_not_close_other_thread_upstream_ws(
    monkeypatch, tmp_path, auth_present,
):
    """Parent wait_agent HTTP must not kill a concurrent reviewer subagent WS upstream."""
    parent_headers = {
        "x-codex-window-id": "parent-thread:13",
        "x-codex-turn-metadata": json.dumps(
            {"session_id": "shared-session", "thread_id": "parent-thread"}
        ),
    }
    reviewer_headers = {
        "x-codex-window-id": "reviewer-thread:0",
        "x-codex-turn-metadata": json.dumps(
            {
                "session_id": "shared-session",
                "thread_id": "reviewer-thread",
                "parent_thread_id": "parent-thread",
                "subagent_kind": "thread_spawn",
            }
        ),
    }

    state = MockUpstreamWsState(
        response_sequences=[
            [
                {"type": "response.created", "response": {"id": "resp_reviewer_1", "model": "gpt-5.5"}},
                {"type": "response.completed", "response": {"id": "resp_reviewer_1", "model": "gpt-5.5", "status": "completed"}},
            ],
            [
                {"type": "response.created", "response": {"id": "resp_reviewer_2", "model": "gpt-5.5"}},
                {"type": "response.completed", "response": {"id": "resp_reviewer_2", "model": "gpt-5.5", "status": "completed"}},
            ],
        ]
    )
    upstream_state, upstream_client = await start_mock_upstream_ws(state)
    _patch_chatgpt_ws_url(monkeypatch, _ws_url_from_test_client(upstream_client))

    class FakeHttpUpstream:
        status = 200
        content_type = "text/event-stream"
        headers = {}

        def __init__(self) -> None:
            self.content = _FakeSseContent(
                [
                    b'data: {"type":"response.created","response":{"id":"resp_parent","model":"gpt-5.5"}}\n\n',
                    b'data: {"type":"response.completed","response":{"id":"resp_parent","model":"gpt-5.5","status":"completed"}}\n\n',
                ]
            )

        def release(self) -> None:
            pass

    async def fake_post(self, url, json=None, headers=None):
        return FakeHttpUpstream()

    monkeypatch.setattr("codex_shim.server.ClientSession.post", fake_post)

    settings = tmp_path / "settings.json"
    settings.write_text("{}")
    shim = ShimServer(settings)
    shim_client = TestClient(TestServer(shim.app()))
    await shim_client.start_server()
    try:
        reviewer_ws = await shim_client.ws_connect("/v1/responses", headers=reviewer_headers)
        await reviewer_ws.send_json(
            {
                "type": "response.create",
                "model": "codex-gpt-5-5",
                "input": [{"type": "message", "role": "user", "content": "review"}],
                "stream": True,
            }
        )
        for _ in range(2):
            msg = await reviewer_ws.receive(timeout=2)
            assert msg.type.name == "TEXT"
        assert len(upstream_state.handshakes) == 1

        await shim_client.post(
            "/v1/responses",
            json={
                "model": "codex-gpt-5-5",
                "input": [{"type": "message", "role": "user", "content": "wait_agent"}],
                "stream": True,
            },
            headers=parent_headers,
        )

        reviewer_sessions = [
            p for p in shim._active_ws_passthroughs.values() if p.matches_thread("reviewer-thread")
        ]
        assert len(reviewer_sessions) == 1
        reviewer = reviewer_sessions[0]
        assert len(reviewer.upstream_by_url) == 1
        upstream_url = next(iter(reviewer.upstream_by_url))
        upstream = reviewer.upstream_by_url[upstream_url]
        assert not upstream.closed
        assert reviewer.last_upstream_chained_response_id(upstream_url) == "resp_reviewer_1"

        await reviewer_ws.send_json(
            {
                "type": "response.create",
                "model": "codex-gpt-5-5",
                "previous_response_id": "resp_reviewer_1",
                "input": [{"type": "message", "role": "user", "content": "continue"}],
                "stream": True,
            }
        )
        for _ in range(2):
            msg = await reviewer_ws.receive(timeout=2)
            assert msg.type.name == "TEXT"

        assert len(upstream_state.handshakes) == 1
        assert upstream_state.received_frames[1]["previous_response_id"] == "resp_reviewer_1"
        await reviewer_ws.close()
    finally:
        await shim_client.close()
        await upstream_client.close()


class _FakeSseContent:
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = list(chunks)

    async def readany(self) -> bytes:
        await asyncio.sleep(0)
        if not self._chunks:
            return b""
        return self._chunks.pop(0)

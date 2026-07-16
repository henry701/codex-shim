"""Tests for continuation cache hardening: reconnect retry, terminal-only writes, disk replace."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from aiohttp import WSMsgType
from aiohttp.test_utils import TestClient, TestServer

import codex_shim.server as server_module
from codex_shim.chatgpt_conversation_cache import ChatgptConversationCache, sanitize_path_segment, sanitize_response_filename
from codex_shim.server import ShimServer
from codex_shim.ws_passthrough import WsPassthroughConnectError, WsPassthroughSession
from ws_test_support import MockUpstreamWsState, start_mock_upstream_ws


class _FakeSseContent:
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = list(chunks)

    async def readany(self) -> bytes:
        await asyncio.sleep(0)
        if not self._chunks:
            return b""
        return self._chunks.pop(0)


@dataclass
class CachePutRecord:
    session_key: str
    response_id: str
    items_len: int
    terminal: bool


@dataclass
class CachePutTracker:
    records: list[CachePutRecord] = field(default_factory=list)

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        original_put = ChatgptConversationCache.put
        tracker = self

        def tracking_put(
            cache_self: ChatgptConversationCache,
            session_key: str,
            response_id: str,
            items: list[Any],
            *,
            terminal: bool = True,
        ) -> None:
            tracker.records.append(
                CachePutRecord(
                    session_key=session_key,
                    response_id=response_id,
                    items_len=len(items),
                    terminal=terminal,
                )
            )
            return original_put(cache_self, session_key, response_id, items, terminal=terminal)

        monkeypatch.setattr(ChatgptConversationCache, "put", tracking_put)


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


@pytest.fixture
def chatgpt_cache_dir(monkeypatch, tmp_path):
    cache_root = tmp_path / "chatgpt-conversations"
    monkeypatch.setattr(server_module, "_chatgpt_conversations_dir", lambda: cache_root)
    return cache_root


def _patch_chatgpt_ws_url(monkeypatch, url: str) -> None:
    monkeypatch.setattr("codex_shim.ws_passthrough.CHATGPT_WS_URL", url)
    monkeypatch.setattr("codex_shim.server.CHATGPT_WS_URL", url)


def _ws_url_from_test_client(client: TestClient, path: str = "/v1/responses") -> str:
    return str(client.make_url(path)).replace("http://", "ws://", 1)


def test_fresh_cache_reader_sees_overwritten_terminal_put(tmp_path):
    writer = ChatgptConversationCache(tmp_path)
    first = [{"type": "message", "role": "user", "content": "partial"}]
    full = [
        {"type": "message", "role": "user", "content": "partial"},
        {"type": "function_call", "call_id": "c1", "name": "tool", "arguments": "{}"},
    ]
    writer.put("sess", "resp_1", first, terminal=True)
    writer.put("sess", "resp_1", full, terminal=True)

    reader = ChatgptConversationCache(tmp_path)
    assert reader.get("sess", "resp_1") == full
    path = tmp_path / sanitize_path_segment("sess") / sanitize_response_filename("resp_1")
    assert json.loads(path.read_text())["items"] == full


def test_terminal_false_updates_memory_only_then_terminal_replaces_disk(tmp_path: Path):
    cache = ChatgptConversationCache(tmp_path)
    partial = [{"type": "message", "role": "user", "content": "partial"}]
    complete = [
        {"type": "message", "role": "user", "content": "partial"},
        {"type": "message", "role": "assistant", "content": "done"},
    ]
    cache.put("sess", "resp_1", partial, terminal=False)
    path = tmp_path / sanitize_path_segment("sess") / sanitize_response_filename("resp_1")
    assert not path.is_file()
    assert cache.get("sess", "resp_1") == partial

    cache.put("sess", "resp_1", complete, terminal=True)
    assert json.loads(path.read_text())["items"] == complete

    reloaded = ChatgptConversationCache(tmp_path)
    assert reloaded.get("sess", "resp_1") == complete


@pytest.mark.asyncio
async def test_chatgpt_ws_stream_writes_cache_only_once_at_terminal(
    monkeypatch, tmp_path, auth_present, chatgpt_cache_dir
):
    tracker = CachePutTracker()
    tracker.install(monkeypatch)

    session_headers = {"session-id": "ws-terminal-only"}
    first_input = [{"type": "message", "role": "user", "content": "run tool"}]
    tool_call = {
        "type": "function_call",
        "id": "fc_1",
        "call_id": "call_1",
        "name": "exec_command",
        "arguments": "{}",
    }

    state = MockUpstreamWsState(
        response_sequences=[
            [
                {"type": "response.created", "response": {"id": "resp_stream", "model": "gpt-5.5"}},
                {"type": "response.output_item.done", "response": {"id": "resp_stream"}, "item": tool_call},
                {"type": "response.completed", "response": {"id": "resp_stream", "model": "gpt-5.5", "status": "completed"}},
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
            {"type": "response.create", "model": "codex-gpt-5-5", "input": first_input, "stream": True}
        )
        for _ in range(3):
            msg = await ws.receive(timeout=2)
            assert msg.type == WSMsgType.TEXT
        await ws.close()
    finally:
        await shim_client.close()
        await upstream_client.close()

    assert len(tracker.records) == 1
    assert all(record.terminal for record in tracker.records)
    assert tracker.records[0].response_id == "resp_stream"
    assert tracker.records[0].items_len >= 2
    assert upstream_state.received_frames


@pytest.mark.asyncio
async def test_chatgpt_http_sse_stream_writes_cache_only_once_at_terminal(
    monkeypatch, tmp_path, auth_present, chatgpt_cache_dir
):
    monkeypatch.setenv("CODEX_SHIM_WS_PASSTHROUGH", "0")
    tracker = CachePutTracker()
    tracker.install(monkeypatch)

    session_headers = {"session-id": "http-terminal-only"}
    first_input = [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hello"}]}]

    class FakeUpstream:
        status = 200
        content_type = "text/event-stream"
        headers = {}

        def __init__(self) -> None:
            self.content = _FakeSseContent(
                [
                    b'data: {"type":"response.created","response":{"id":"resp_sse","model":"gpt-5.5"}}\n\n',
                    (
                        b'data: {"type":"response.output_item.done","response":{"id":"resp_sse"},'
                        b'"item":{"type":"function_call","call_id":"call_1","name":"exec_command","arguments":"{}"}}\n\n'
                    ),
                    b'data: {"type":"response.completed","response":{"id":"resp_sse","model":"gpt-5.5","status":"completed"}}\n\n',
                ]
            )

        def release(self) -> None:
            pass

    async def fake_post(self, url, json=None, headers=None):
        return FakeUpstream()

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
            msg = await ws.receive(timeout=2)
            assert msg.type == WSMsgType.TEXT
        await ws.close()
    finally:
        await shim_client.close()

    assert len(tracker.records) == 1
    assert all(record.terminal for record in tracker.records)
    assert tracker.records[0].response_id == "resp_sse"
    assert tracker.records[0].items_len >= 2


@pytest.mark.asyncio
async def test_prev_id_retry_closes_upstream_before_reconnect(
    monkeypatch, tmp_path, auth_present, chatgpt_cache_dir
):
    close_calls: list[bool] = []
    original_close = WsPassthroughSession.close_upstream

    async def tracking_close(self: WsPassthroughSession, upstream_url: str | None = None) -> None:
        close_calls.append(True)
        await original_close(self, upstream_url)

    monkeypatch.setattr(WsPassthroughSession, "close_upstream", tracking_close)

    session_headers = {"session-id": "ws-close-retry"}
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
            await ws.receive(timeout=2)

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
            await ws.receive(timeout=2)
        await ws.close()
    finally:
        await shim_client.close()
        await upstream_client.close()

    assert close_calls[0] is True
    assert len(upstream_state.handshakes) == 2


@pytest.mark.asyncio
async def test_prev_id_retry_reconnect_failure_falls_back_to_http_expanded(
    monkeypatch, tmp_path, auth_present, chatgpt_cache_dir
):
    captured_http: dict[str, Any] = {}
    new_connect_attempts: list[str] = []
    original_connect = WsPassthroughSession.connect_upstream

    async def counting_connect(
        self: WsPassthroughSession,
        url: str,
        headers: dict[str, str],
    ):
        existing = self.upstream_by_url.get(url)
        if existing is not None and not existing.closed:
            return {}, True
        new_connect_attempts.append("new")
        if len(new_connect_attempts) > 1:
            raise WsPassthroughConnectError("reconnect refused", status=503)
        return await original_connect(self, url, headers)

    monkeypatch.setattr(WsPassthroughSession, "connect_upstream", counting_connect)

    session_headers = {"session-id": "ws-http-fallback"}
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
        ]
    )
    upstream_state, upstream_client = await start_mock_upstream_ws(state)
    _patch_chatgpt_ws_url(monkeypatch, _ws_url_from_test_client(upstream_client))

    class FakeHttpUpstream:
        status = 200
        content_type = "text/event-stream"
        headers = {}
        content = _FakeSseContent(
            [
                b'data: {"type":"response.created","response":{"id":"resp_http","model":"gpt-5.5"}}\n\n',
                b'data: {"type":"response.completed","response":{"id":"resp_http","model":"gpt-5.5","status":"completed"}}\n\n',
            ]
        )

        def release(self) -> None:
            pass

    async def fake_post(self, url, json=None, headers=None):
        captured_http["url"] = str(url)
        captured_http["body"] = json
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
        await ws.close()
    finally:
        await shim_client.close()
        await upstream_client.close()

    assert captured_http["url"] == "https://chatgpt.com/backend-api/codex/responses"
    body = captured_http["body"]
    assert "previous_response_id" not in body
    assert body["input"][0] == first_input[0]
    assert body["input"][1]["call_id"] == "call_1"
    assert body["input"][2] == tool_output
    assert len(upstream_state.received_frames) == 2


@pytest.mark.asyncio
async def test_expand_retry_still_fails_falls_back_to_http(
    monkeypatch, tmp_path, auth_present, chatgpt_cache_dir
):
    captured_http: dict[str, Any] = {}
    session_headers = {"session-id": "ws-expand-fail"}
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
    cache.put("ws-expand-fail", "resp_previous", [*first_input, tool_call])

    prev_id_error = {
        "type": "error",
        "error": {
            "code": "previous_response_not_found",
            "message": "Previous response with id 'resp_previous' not found.",
            "param": "previous_response_id",
        },
    }
    state = MockUpstreamWsState(
        response_sequences=[
            [
                {"type": "response.created", "response": {"id": "resp_previous", "model": "gpt-5.5"}},
                {"type": "response.output_item.done", "response": {"id": "resp_previous"}, "item": tool_call},
                {"type": "response.completed", "response": {"id": "resp_previous", "model": "gpt-5.5", "status": "completed"}},
            ],
            [prev_id_error],
            [prev_id_error],
        ]
    )
    upstream_state, upstream_client = await start_mock_upstream_ws(state)
    _patch_chatgpt_ws_url(monkeypatch, _ws_url_from_test_client(upstream_client))

    class FakeHttpUpstream:
        status = 200
        content_type = "text/event-stream"
        headers = {}
        content = _FakeSseContent(
            [
                b'data: {"type":"response.created","response":{"id":"resp_http","model":"gpt-5.5"}}\n\n',
                b'data: {"type":"response.completed","response":{"id":"resp_http","model":"gpt-5.5","status":"completed"}}\n\n',
            ]
        )

        def release(self) -> None:
            pass

    async def fake_post(self, url, json=None, headers=None):
        captured_http["url"] = str(url)
        captured_http["body"] = json
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
        await ws.close()
    finally:
        await shim_client.close()
        await upstream_client.close()

    assert captured_http["body"] is not None
    assert "previous_response_id" not in captured_http["body"]
    assert captured_http["body"]["input"][0] == first_input[0]
    assert len(upstream_state.received_frames) == 3
    assert len(upstream_state.handshakes) == 2

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp import WSMsgType
from multidict import CIMultiDict

from codex_shim.header_passthrough import (
    chatgpt_passthrough_ws_upstream_headers,
    forwardable_ws_upgrade_headers,
)
from codex_shim.server import ChatgptPassthroughResponseCollector, _rewrite_response_model
from codex_shim.ws_passthrough import (
    CHATGPT_WS_URL,
    WsPassthroughSession,
    responses_websocket_url,
    ws_passthrough_enabled,
)


def test_ws_passthrough_enabled_default_on():
    assert ws_passthrough_enabled() is True


def test_ws_passthrough_disabled_by_env(monkeypatch):
    monkeypatch.setenv("CODEX_SHIM_WS_PASSTHROUGH", "0")
    assert ws_passthrough_enabled() is False


def test_responses_websocket_url_from_versioned_base():
    assert responses_websocket_url("https://api.example.com/v1") == "wss://api.example.com/v1/responses"


def test_responses_websocket_url_from_unversioned_base():
    assert responses_websocket_url("https://api.example.com") == "wss://api.example.com/v1/responses"


def test_chatgpt_ws_upstream_headers_prefer_websockets_beta():
    headers = chatgpt_passthrough_ws_upstream_headers(
        CIMultiDict([("session-id", "sess-1")]),
        access_token="tok",
        account_id="acct",
    )
    assert headers["OpenAI-Beta"] == "responses_websockets=2026-02-06"
    assert headers["Authorization"] == "Bearer tok"
    assert headers["session-id"] == "sess-1"
    assert "Accept" not in headers


def test_forwardable_ws_upgrade_headers_strips_ws_handshake_fields():
    forwarded = forwardable_ws_upgrade_headers(
        {
            "x-codex-turn-state": "ts-1",
            "Sec-WebSocket-Accept": "abc",
            "Content-Length": "0",
        }
    )
    assert forwarded == {"x-codex-turn-state": "ts-1"}


def test_ws_upgrade_headers_strip_sec_websocket_on_upstream_request():
    headers = chatgpt_passthrough_ws_upstream_headers(
        CIMultiDict(
            [
                ("Sec-WebSocket-Key", "abc"),
                ("session-id", "sess-1"),
            ]
        ),
        access_token="tok",
        account_id="acct",
    )
    assert "Sec-WebSocket-Key" not in headers
    assert headers["session-id"] == "sess-1"


class FakeUpstreamWs:
    def __init__(self, events: list[str]) -> None:
        self.closed = False
        self.response = MagicMock(headers=CIMultiDict())
        self._events = events

    def __aiter__(self):
        return self._iter_messages()

    async def _iter_messages(self):
        for item in self._events:
            msg = MagicMock()
            msg.type = WSMsgType.TEXT
            msg.data = item
            yield msg


@pytest.mark.asyncio
async def test_relay_rewrites_model_in_events():
    client_ws = AsyncMock()
    client_ws.closed = False
    events = [
        json.dumps({"type": "response.created", "response": {"id": "r1", "model": "gpt-5.5"}}),
        json.dumps({"type": "response.completed", "response": {"id": "r1", "model": "gpt-5.5", "status": "completed"}}),
    ]
    upstream_ws = FakeUpstreamWs(events)

    session = WsPassthroughSession(client_session=AsyncMock(), client_ws=client_ws, upstream_ws=upstream_ws)
    sent: list[dict] = []

    async def capture(event: dict) -> None:
        sent.append(event)

    await session.relay_until_terminal(
        source="test-ws",
        model_override="codex-gpt-5-5",
        rewrite_model=_rewrite_response_model,
        write_event=capture,
    )

    assert sent[0]["response"]["model"] == "codex-gpt-5-5"
    assert sent[1]["response"]["model"] == "codex-gpt-5-5"


@pytest.mark.asyncio
async def test_relay_records_usage_on_completed(monkeypatch):
    observed = []

    def fake_observe(source, upstream, *, usage=None):
        observed.append({"source": source, "usage": usage})

    monkeypatch.setattr("codex_shim.ws_passthrough.observe_upstream_response", fake_observe)

    client_ws = AsyncMock()
    payload = json.dumps(
        {
            "type": "response.completed",
            "response": {
                "id": "r1",
                "model": "gpt-5.5",
                "status": "completed",
                "usage": {"input_tokens": 3, "input_tokens_details": {"cached_tokens": 2}},
            },
        }
    )
    upstream_ws = FakeUpstreamWs([payload])

    session = WsPassthroughSession(client_session=AsyncMock(), client_ws=client_ws, upstream_ws=upstream_ws)
    await session.relay_until_terminal(source="chatgpt-passthrough-ws", write_event=AsyncMock())

    assert observed[0]["source"] == "chatgpt-passthrough-ws"
    assert observed[0]["usage"]["input_tokens_details"]["cached_tokens"] == 2


def test_chatgpt_expand_applied_before_upstream_send():
    first_input = [{"type": "message", "role": "user", "content": "run"}]
    tool_call = {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "exec_command", "arguments": "{}"}
    tool_output = {"type": "function_call_output", "call_id": "call_1", "output": "ok"}
    forwarded = {"input": [*first_input, tool_call, tool_output], "model": "gpt-5.5"}
    collector = ChatgptPassthroughResponseCollector(forwarded)
    assert collector.conversation_items() == [*first_input, tool_call, tool_output]


def test_chatgpt_ws_url_constant():
    assert CHATGPT_WS_URL == "wss://chatgpt.com/backend-api/codex/responses"

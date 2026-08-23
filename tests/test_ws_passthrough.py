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
    UPSTREAM_WS_HEARTBEAT,
    WsPassthroughConnectError,
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
    def __init__(
        self,
        events: list[str],
        *,
        raise_after: BaseException | None = None,
        close_after: bool = False,
    ) -> None:
        self.closed = False
        self.response = MagicMock(headers=CIMultiDict())
        self._events = events
        self._raise_after = raise_after
        self._close_after = close_after

    def __aiter__(self):
        return self._iter_messages()

    async def _iter_messages(self):
        for item in self._events:
            msg = MagicMock()
            msg.type = WSMsgType.TEXT
            msg.data = item
            yield msg
        if self._raise_after is not None:
            raise self._raise_after
        if self._close_after:
            msg = MagicMock()
            msg.type = WSMsgType.CLOSE
            yield msg

    async def close(self):
        self.closed = True


@pytest.mark.asyncio
async def test_relay_rewrites_model_in_events():
    client_ws = AsyncMock()
    client_ws.closed = False
    events = [
        json.dumps({"type": "response.created", "response": {"id": "r1", "model": "gpt-5.5"}}),
        json.dumps({"type": "response.completed", "response": {"id": "r1", "model": "gpt-5.5", "status": "completed"}}),
    ]
    upstream_ws = FakeUpstreamWs(events)

    session = WsPassthroughSession(client_session=AsyncMock(), client_ws=client_ws)
    url = "ws://example/v1/responses"
    session.upstream_by_url[url] = upstream_ws
    sent: list[dict] = []

    async def capture(event: dict) -> None:
        sent.append(event)

    await session.relay_until_terminal(
        source="test-ws",
        upstream_url=url,
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

    session = WsPassthroughSession(client_session=AsyncMock(), client_ws=client_ws)
    url = "ws://example/v1/responses"
    session.upstream_by_url[url] = upstream_ws
    await session.relay_until_terminal(
        source="chatgpt-passthrough-ws",
        upstream_url=url,
        write_event=AsyncMock(),
    )

    assert observed[0]["source"] == "chatgpt-passthrough-ws"
    assert observed[0]["usage"]["input_tokens_details"]["cached_tokens"] == 2


def test_chatgpt_collector_keeps_output_item_done_when_completed_output_empty():
    collector = ChatgptPassthroughResponseCollector({"input": []})
    collector.record(
        {
            "type": "response.output_item.done",
            "item": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "## Goal\n- Compact"}],
            },
        }
    )
    collector.record(
        {
            "type": "response.completed",
            "response": {"id": "resp_lite", "status": "completed", "output": []},
        }
    )
    output = collector.output_items()
    assert output[0]["type"] == "message"
    assert output[0]["content"][0]["text"] == "## Goal\n- Compact"

    first_input = [{"type": "message", "role": "user", "content": "run"}]
    tool_call = {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "exec_command", "arguments": "{}"}
    tool_output = {"type": "function_call_output", "call_id": "call_1", "output": "ok"}
    forwarded = {"input": [*first_input, tool_call, tool_output], "model": "gpt-5.5"}
    collector = ChatgptPassthroughResponseCollector(forwarded)
    assert collector.conversation_items() == [*first_input, tool_call, tool_output]


def test_chatgpt_ws_url_constant():
    assert CHATGPT_WS_URL == "wss://chatgpt.com/backend-api/codex/responses"


@pytest.mark.asyncio
async def test_connect_upstream_passes_heartbeat():
    client_ws = AsyncMock()
    client_ws.closed = False
    new_ws = AsyncMock()
    new_ws.closed = False
    new_ws.headers = {}
    new_ws.exception = MagicMock(return_value=None)
    client_session = AsyncMock()
    client_session.ws_connect = AsyncMock(return_value=new_ws)
    session = WsPassthroughSession(client_session=client_session, client_ws=client_ws)
    url = "wss://example/v1/responses"
    _, reused = await session.connect_upstream(url, {"Authorization": "Bearer x"})
    assert reused is False
    kwargs = client_session.ws_connect.call_args.kwargs
    assert kwargs["heartbeat"] == UPSTREAM_WS_HEARTBEAT
    assert kwargs["headers"] == {"Authorization": "Bearer x"}


@pytest.mark.asyncio
async def test_connect_upstream_does_not_reuse_ws_with_exception():
    client_ws = AsyncMock()
    client_ws.closed = False
    dead_ws = AsyncMock()
    dead_ws.closed = False
    dead_ws.exception = MagicMock(return_value=ConnectionError("broken"))
    new_ws = AsyncMock()
    new_ws.closed = False
    new_ws.headers = {}
    new_ws.exception = MagicMock(return_value=None)
    client_session = AsyncMock()
    client_session.ws_connect = AsyncMock(return_value=new_ws)
    session = WsPassthroughSession(client_session=client_session, client_ws=client_ws)
    url = "wss://example/v1/responses"
    session.upstream_by_url[url] = dead_ws
    _, reused = await session.connect_upstream(url, {})
    assert reused is False
    client_session.ws_connect.assert_awaited()


@pytest.mark.asyncio
async def test_send_response_create_failure_closes_lane():
    client_ws = AsyncMock()
    client_ws.closed = False
    upstream_ws = AsyncMock()
    upstream_ws.closed = False
    upstream_ws.exception = MagicMock(return_value=None)
    upstream_ws.send_str = AsyncMock(side_effect=ConnectionError("broken pipe"))
    session = WsPassthroughSession(client_session=AsyncMock(), client_ws=client_ws)
    url = "wss://example/v1/responses"
    session.upstream_by_url[url] = upstream_ws
    with pytest.raises(WsPassthroughConnectError):
        await session.send_response_create({"model": "gpt-5.5"}, upstream_url=url)
    assert url not in session.upstream_by_url
    upstream_ws.close.assert_awaited()


@pytest.mark.asyncio
async def test_relay_close_frame_drops_lane():
    client_ws = AsyncMock()
    client_ws.closed = False

    class ClosingWs:
        def __init__(self) -> None:
            self.closed = False

        def __aiter__(self):
            return self._iter()

        async def _iter(self):
            msg = MagicMock()
            msg.type = WSMsgType.CLOSE
            yield msg

        async def close(self):
            self.closed = True

    session = WsPassthroughSession(client_session=AsyncMock(), client_ws=client_ws)
    url = "wss://example/v1/responses"
    session.upstream_by_url[url] = ClosingWs()
    terminal = await session.relay_until_terminal(source="test-ws", upstream_url=url)
    assert terminal is not None
    assert terminal["type"] == "response.incomplete"
    assert url not in session.upstream_by_url
    sent = json.loads(client_ws.send_str.await_args.args[0])
    assert sent["type"] == "response.incomplete"


@pytest.mark.asyncio
async def test_relay_text_without_terminal_synthesizes_incomplete():
    client_ws = AsyncMock()
    client_ws.closed = False
    events = [
        json.dumps({"type": "response.created", "response": {"id": "r1", "model": "gpt-5.5"}}),
    ]
    session = WsPassthroughSession(client_session=AsyncMock(), client_ws=client_ws)
    url = "wss://example/v1/responses"
    session.upstream_by_url[url] = FakeUpstreamWs(events)
    sent: list[dict] = []

    async def capture(event: dict) -> None:
        sent.append(event)

    terminal = await session.relay_until_terminal(
        source="test-ws",
        upstream_url=url,
        write_event=capture,
    )
    assert terminal is not None
    assert terminal["type"] == "response.incomplete"
    assert [event["type"] for event in sent] == ["response.created", "response.incomplete"]
    assert url not in session.upstream_by_url


@pytest.mark.asyncio
async def test_relay_connection_reset_on_pong_closes_lane_and_raises():
    client_ws = AsyncMock()
    client_ws.closed = False

    class ResettingWs:
        def __init__(self) -> None:
            self.closed = False
            self.close = AsyncMock()

        def __aiter__(self):
            return self._iter()

        async def _iter(self):
            from aiohttp.client_exceptions import ClientConnectionResetError

            raise ClientConnectionResetError("Cannot write to closing transport")
            yield  # pragma: no cover

    session = WsPassthroughSession(client_session=AsyncMock(), client_ws=client_ws)
    url = "wss://example/v1/responses"
    session.upstream_by_url[url] = ResettingWs()
    with pytest.raises(WsPassthroughConnectError, match="closing transport"):
        await session.relay_until_terminal(source="test-ws", upstream_url=url)
    assert url not in session.upstream_by_url


@pytest.mark.asyncio
async def test_relay_error_frame_drops_lane():
    client_ws = AsyncMock()
    client_ws.closed = False

    class ErrorWs:
        def __init__(self) -> None:
            self.closed = False

        def __aiter__(self):
            return self._iter()

        async def _iter(self):
            msg = MagicMock()
            msg.type = WSMsgType.ERROR
            yield msg

        async def close(self):
            self.closed = True

    session = WsPassthroughSession(client_session=AsyncMock(), client_ws=client_ws)
    url = "wss://example/v1/responses"
    session.upstream_by_url[url] = ErrorWs()
    terminal = await session.relay_until_terminal(source="test-ws", upstream_url=url)
    assert terminal is not None
    assert terminal["type"] == "response.incomplete"
    assert url not in session.upstream_by_url
    sent = json.loads(client_ws.send_str.await_args.args[0])
    assert sent["type"] == "response.incomplete"


def _relay_session(events: list[str], **ws_kwargs):
    client_ws = AsyncMock()
    client_ws.closed = False
    session = WsPassthroughSession(client_session=AsyncMock(), client_ws=client_ws)
    url = "wss://example/v1/responses"
    session.upstream_by_url[url] = FakeUpstreamWs(events, **ws_kwargs)
    sent: list[dict] = []

    async def capture(event: dict) -> None:
        sent.append(event)

    return session, url, sent, capture


@pytest.mark.asyncio
async def test_relay_reset_before_first_event_raises_for_fallback():
    session, url, sent, capture = _relay_session([], raise_after=ConnectionResetError("reset"))
    with pytest.raises(WsPassthroughConnectError):
        await session.relay_until_terminal(source="test-ws", upstream_url=url, write_event=capture)
    assert sent == []
    assert url not in session.upstream_by_url


@pytest.mark.asyncio
async def test_relay_reset_after_forwarded_event_synthesizes_incomplete_without_fallback_signal():
    events = [json.dumps({"type": "response.created", "response": {"id": "r1", "model": "gpt-5.5"}})]
    session, url, sent, capture = _relay_session(events, raise_after=ConnectionResetError("reset"))
    terminal = await session.relay_until_terminal(source="test-ws", upstream_url=url, write_event=capture)
    assert terminal is not None
    assert terminal["type"] == "response.incomplete"
    assert [event["type"] for event in sent] == ["response.created", "response.incomplete"]
    assert url not in session.upstream_by_url


@pytest.mark.asyncio
async def test_relay_close_after_forwarded_event_emits_one_terminal():
    events = [json.dumps({"type": "response.created", "response": {"id": "r1", "model": "gpt-5.5"}})]
    session, url, sent, capture = _relay_session(events, close_after=True)
    terminal = await session.relay_until_terminal(source="test-ws", upstream_url=url, write_event=capture)
    assert terminal is not None
    assert terminal["type"] == "response.incomplete"
    assert [event["type"] for event in sent] == ["response.created", "response.incomplete"]
    assert url not in session.upstream_by_url


@pytest.mark.asyncio
async def test_suppressed_previous_response_id_error_remains_precontent():
    events = [
        json.dumps(
            {
                "type": "error",
                "error": {"code": "previous_response_not_found", "message": "not found"},
            }
        )
    ]
    session, url, sent, capture = _relay_session(events)

    def forward_terminal(event: dict) -> bool:
        return False

    terminal = await session.relay_until_terminal(
        source="test-ws",
        upstream_url=url,
        write_event=capture,
        forward_terminal=forward_terminal,
    )
    assert sent == []
    assert terminal is not None
    assert terminal["type"] == "error"
    assert url in session.upstream_by_url

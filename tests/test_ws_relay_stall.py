"""A silent-but-alive upstream WS must never hang a passthrough turn.

Regression for a Sol turn that waited 4.5h: after a reused-lane turn completed,
the next no-prev_id response.create got no upstream events while heartbeats kept
the socket open, and the relay had no bound on silence nor watched the client.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp import ClientWSTimeout, WSMsgType
from aiohttp.test_utils import TestClient, TestServer

import codex_shim.server as server_module
from codex_shim.net.sse import ClientDisconnected
from codex_shim.server import ShimServer
from codex_shim.ws_passthrough import (
    DEFAULT_WS_FIRST_EVENT_TIMEOUT_SEC,
    DEFAULT_WS_IDLE_TIMEOUT_SEC,
    WS_FIRST_EVENT_TIMEOUT_ENV,
    WS_IDLE_TIMEOUT_ENV,
    WS_RELAY_STALL_ERROR_CODE,
    WsPassthroughSession,
    WsRelayTimeouts,
    ws_relay_timeouts_from_env,
)
from ws_test_support import MockUpstreamWsState, start_mock_upstream_ws

LANE = "ws://example/v1/responses"


class SilentUpstreamWs:
    """Accepts frames, yields ``events`` then blocks forever (like a heartbeat-only peer)."""

    def __init__(self, events: list[dict] | None = None) -> None:
        self.closed = False
        self._events = [json.dumps(e) for e in events or []]

    def __aiter__(self):
        return self._iter()

    async def _iter(self):
        for data in self._events:
            yield MagicMock(type=WSMsgType.TEXT, data=data)
        await asyncio.Event().wait()

    async def close(self) -> None:
        self.closed = True


def _session(upstream: SilentUpstreamWs) -> WsPassthroughSession:
    client_ws = AsyncMock()
    client_ws.closed = False
    session = WsPassthroughSession(client_session=AsyncMock(), client_ws=client_ws)
    session.upstream_by_url[LANE] = upstream
    session.last_chained_response_id_by_url[LANE] = "resp_prev"
    return session


def _short_timeouts(monkeypatch, *, first: float = 0.2, idle: float = 0.2) -> None:
    monkeypatch.setenv(WS_FIRST_EVENT_TIMEOUT_ENV, str(first))
    monkeypatch.setenv(WS_IDLE_TIMEOUT_ENV, str(idle))


def test_timeouts_default_and_zero_disables(monkeypatch):
    monkeypatch.delenv(WS_FIRST_EVENT_TIMEOUT_ENV, raising=False)
    monkeypatch.delenv(WS_IDLE_TIMEOUT_ENV, raising=False)
    assert ws_relay_timeouts_from_env() == WsRelayTimeouts(
        first_event=DEFAULT_WS_FIRST_EVENT_TIMEOUT_SEC, idle=DEFAULT_WS_IDLE_TIMEOUT_SEC
    )
    _short_timeouts(monkeypatch, first=0, idle=0)
    assert ws_relay_timeouts_from_env() == WsRelayTimeouts(first_event=None, idle=None)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("events", "phrase"),
    [
        ([], "no upstream event"),
        (
            [
                {
                    "type": "response.created",
                    "response": {"id": "r1", "model": "gpt-5.5"},
                }
            ],
            "upstream went silent",
        ),
    ],
    ids=["first-event", "idle"],
)
async def test_silent_upstream_fails_turn_and_drops_lane(
    monkeypatch, capsys, events, phrase
):
    _short_timeouts(monkeypatch)
    upstream = SilentUpstreamWs(events)
    session = _session(upstream)
    sent: list[dict] = []

    async def capture(event: dict) -> None:
        sent.append(event)

    terminal = await asyncio.wait_for(
        session.relay_until_terminal(
            source="chatgpt-passthrough-ws", upstream_url=LANE, write_event=capture
        ),
        timeout=3,
    )

    assert terminal is not None and terminal["type"] == "response.failed"
    assert terminal["response"]["error"]["code"] == WS_RELAY_STALL_ERROR_CODE
    assert phrase in terminal["response"]["error"]["message"]
    assert [e["type"] for e in sent][-2:] == ["error", "response.failed"]
    assert upstream.closed
    assert LANE not in session.upstream_by_url
    assert LANE not in session.last_chained_response_id_by_url
    out = capsys.readouterr().out
    assert f"[stream-end] chatgpt-passthrough-ws elapsed=" in out
    assert f"upstream_events={len(events)}" in out
    assert "terminal=response.failed" in out


@pytest.mark.asyncio
async def test_client_gone_mid_relay_closes_upstream(monkeypatch):
    _short_timeouts(monkeypatch, first=0, idle=0)
    monkeypatch.setattr("codex_shim.ws_passthrough.CLIENT_GONE_POLL_SEC", 0.02)
    upstream = SilentUpstreamWs()
    session = _session(upstream)
    gone = False
    session.client_transport_closed = lambda: gone

    relay = asyncio.ensure_future(
        session.relay_until_terminal(
            source="chatgpt-passthrough-ws", upstream_url=LANE, write_event=AsyncMock()
        )
    )
    await asyncio.sleep(0.1)
    assert not relay.done()
    gone = True
    with pytest.raises(ClientDisconnected):
        await asyncio.wait_for(relay, timeout=2)
    assert upstream.closed
    assert LANE not in session.upstream_by_url


# --- end to end through the shim's /v1/responses websocket -------------------


@pytest.fixture
def auth_present(monkeypatch, tmp_path):
    auth = tmp_path / "auth.json"
    auth.write_text(
        json.dumps(
            {"tokens": {"access_token": "test-token", "account_id": "acct-test"}}
        )
    )
    monkeypatch.setattr("codex_shim.settings.DEFAULT_CODEX_AUTH", auth)
    monkeypatch.setattr("codex_shim.server.DEFAULT_CODEX_AUTH", auth)
    monkeypatch.setattr(
        server_module,
        "_chatgpt_conversations_dir",
        lambda: tmp_path / "chatgpt-conversations",
    )
    return auth


async def _start(
    monkeypatch, tmp_path, state: MockUpstreamWsState
) -> tuple[TestClient, TestClient]:
    _, upstream_client = await start_mock_upstream_ws(state)
    url = str(upstream_client.make_url("/v1/responses")).replace("http://", "ws://", 1)
    monkeypatch.setattr("codex_shim.ws_passthrough.CHATGPT_WS_URL", url)
    monkeypatch.setattr("codex_shim.server.CHATGPT_WS_URL", url)
    settings = tmp_path / "settings.json"
    settings.write_text("{}")
    shim_client = TestClient(TestServer(ShimServer(settings).app()))
    await shim_client.start_server()
    return shim_client, upstream_client


def _completed(resp_id: str) -> list[dict]:
    return [
        {"type": "response.created", "response": {"id": resp_id, "model": "gpt-5.5"}},
        {
            "type": "response.completed",
            "response": {"id": resp_id, "model": "gpt-5.5", "status": "completed"},
        },
    ]


async def _recv_until_terminal(ws, frame_wait_sec: float) -> list[dict]:
    events: list[dict] = []
    while True:
        msg = await ws.receive(timeout=frame_wait_sec)
        assert msg.type.name == "TEXT", msg
        event = json.loads(msg.data)
        events.append(event)
        if event.get("type") in {
            "response.completed",
            "response.failed",
            "response.incomplete",
        }:
            return events


@pytest.mark.asyncio
async def test_post_compaction_no_prev_id_turn_on_reused_lane_does_not_hang(
    monkeypatch, tmp_path, auth_present
):
    _short_timeouts(monkeypatch, first=0.3, idle=0.3)
    state = MockUpstreamWsState(
        response_sequences=[_completed("resp_compacted"), [], _completed("resp_retry")]
    )
    shim_client, upstream_client = await _start(monkeypatch, tmp_path, state)
    compacted_input = [
        {"type": "message", "role": "user", "content": "summary of earlier turns"}
    ]
    try:
        ws = await shim_client.ws_connect(
            "/v1/responses", headers={"session-id": "ws-stall"}
        )
        await ws.send_json(
            {
                "type": "response.create",
                "model": "codex-gpt-5-5",
                "input": [{"type": "message", "role": "user", "content": "hi"}],
            }
        )
        assert (await _recv_until_terminal(ws, 2))[-1]["type"] == "response.completed"

        # Post-compaction shape: no previous_response_id, full compacted input, reused lane, silent upstream.
        await ws.send_json(
            {
                "type": "response.create",
                "model": "codex-gpt-5-5",
                "input": compacted_input,
            }
        )
        stalled = await _recv_until_terminal(ws, 3)
        assert [e["type"] for e in stalled][-2:] == ["error", "response.failed"]
        assert stalled[-1]["response"]["error"]["code"] == WS_RELAY_STALL_ERROR_CODE
        for _ in range(50):
            if state.closed_connections:
                break
            await asyncio.sleep(0.02)
        assert state.closed_connections == 1
        assert len(state.received_frames) == 2
        assert "previous_response_id" not in state.received_frames[1]

        # Desktop's retry lands on a fresh upstream lane and completes.
        await ws.send_json(
            {
                "type": "response.create",
                "model": "codex-gpt-5-5",
                "input": compacted_input,
            }
        )
        assert (await _recv_until_terminal(ws, 2))[-1]["type"] == "response.completed"
        assert len(state.handshakes) == 2
        await ws.close()
    finally:
        await shim_client.close()
        await upstream_client.close()


@pytest.mark.asyncio
async def test_client_drop_mid_relay_closes_upstream_lane(
    monkeypatch, tmp_path, auth_present
):
    _short_timeouts(monkeypatch, first=0, idle=0)
    monkeypatch.setattr("codex_shim.ws_passthrough.CLIENT_GONE_POLL_SEC", 0.02)
    state = MockUpstreamWsState(response_sequences=[[]])
    shim_client, upstream_client = await _start(monkeypatch, tmp_path, state)
    try:
        ws = await shim_client.ws_connect(
            "/v1/responses",
            headers={"session-id": "ws-drop"},
            timeout=ClientWSTimeout(ws_close=0.1),
        )
        await ws.send_json(
            {"type": "response.create", "model": "codex-gpt-5-5", "input": []}
        )
        for _ in range(50):
            if state.received_frames:
                break
            await asyncio.sleep(0.02)
        assert state.received_frames and state.closed_connections == 0
        await ws.close()
        for _ in range(100):
            if state.closed_connections:
                break
            await asyncio.sleep(0.02)
        assert state.closed_connections == 1
    finally:
        await shim_client.close()
        await upstream_client.close()

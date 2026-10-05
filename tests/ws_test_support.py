from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from aiohttp import WSMsgType, web
from aiohttp.test_utils import TestClient, TestServer

RESPONSES_TERMINAL_TYPES = frozenset({"response.completed", "response.failed", "response.incomplete"})


@dataclass
class RecordedWsHandshake:
    headers: dict[str, str]
    path: str


@dataclass
class MockUpstreamWsState:
    handshakes: list[RecordedWsHandshake] = field(default_factory=list)
    received_frames: list[dict[str, Any]] = field(default_factory=list)
    response_sequences: list[list[dict[str, Any]]] = field(default_factory=list)
    upgrade_headers: dict[str, str] = field(default_factory=dict)
    closed_connections: int = 0
    _sequence_index: int = 0

    def next_responses(self) -> list[dict[str, Any]]:
        if self._sequence_index >= len(self.response_sequences):
            return [
                {"type": "response.created", "response": {"id": "resp_default", "model": "gpt-5.5"}},
                {"type": "response.completed", "response": {"id": "resp_default", "model": "gpt-5.5", "status": "completed"}},
            ]
        responses = self.response_sequences[self._sequence_index]
        self._sequence_index += 1
        return responses


def build_mock_upstream_ws_app(state: MockUpstreamWsState, path: str = "/v1/responses") -> web.Application:
    app = web.Application()

    async def ws_handler(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        for key, value in state.upgrade_headers.items():
            ws.headers[key] = value
        await ws.prepare(request)
        state.handshakes.append(
            RecordedWsHandshake(
                headers={key: request.headers.get(key, "") for key in request.headers},
                path=request.path,
            )
        )
        async for msg in ws:
            if msg.type != WSMsgType.TEXT:
                continue
            payload = json.loads(msg.data)
            if not isinstance(payload, dict):
                continue
            state.received_frames.append(payload)
            for event in state.next_responses():
                await ws.send_str(json.dumps(event, separators=(",", ":")))
        state.closed_connections += 1
        return ws

    app.router.add_get(path, ws_handler)
    return app


async def start_mock_upstream_ws(
    state: MockUpstreamWsState | None = None,
    *,
    path: str = "/v1/responses",
) -> tuple[MockUpstreamWsState, TestClient]:
    state = state or MockUpstreamWsState()
    client = TestClient(TestServer(build_mock_upstream_ws_app(state, path=path)))
    await client.start_server()
    return state, client


def install_chatgpt_auth(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Fake ~/.codex/auth.json and an isolated ChatGPT conversation cache dir."""
    import codex_shim.server as server_module

    auth = tmp_path / "auth.json"
    auth.write_text(json.dumps({"tokens": {"access_token": "test-token", "account_id": "acct-test"}}))
    monkeypatch.setattr("codex_shim.settings.DEFAULT_CODEX_AUTH", auth)
    monkeypatch.setattr("codex_shim.server.DEFAULT_CODEX_AUTH", auth)
    monkeypatch.setattr(server_module, "_chatgpt_conversations_dir", lambda: tmp_path / "chatgpt-conversations")
    return auth


async def start_shim_with_mock_chatgpt_ws(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, state: MockUpstreamWsState
) -> tuple[TestClient, TestClient]:
    """Shim whose ChatGPT WS upstream is the mock; returns (shim_client, upstream_client)."""
    from codex_shim.server import ShimServer

    _, upstream_client = await start_mock_upstream_ws(state)
    url = str(upstream_client.make_url("/v1/responses")).replace("http://", "ws://", 1)
    monkeypatch.setattr("codex_shim.ws_passthrough.CHATGPT_WS_URL", url)
    monkeypatch.setattr("codex_shim.server.CHATGPT_WS_URL", url)
    settings = tmp_path / "settings.json"
    settings.write_text("{}")
    shim_client = TestClient(TestServer(ShimServer(settings).app()))
    await shim_client.start_server()
    return shim_client, upstream_client


def completed_events(resp_id: str) -> list[dict[str, Any]]:
    return [
        {"type": "response.created", "response": {"id": resp_id, "model": "gpt-5.5"}},
        {"type": "response.completed", "response": {"id": resp_id, "model": "gpt-5.5", "status": "completed"}},
    ]


async def recv_until_terminal(ws: Any, frame_wait_sec: float) -> list[dict[str, Any]]:
    """Client frames up to and including the first Responses terminal; fails if none arrives."""
    events: list[dict[str, Any]] = []
    while True:
        msg = await ws.receive(timeout=frame_wait_sec)
        assert msg.type.name == "TEXT", msg
        event = json.loads(msg.data)
        events.append(event)
        if event.get("type") in RESPONSES_TERMINAL_TYPES:
            return events

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from aiohttp import WSMsgType, web
from aiohttp.test_utils import TestClient, TestServer


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

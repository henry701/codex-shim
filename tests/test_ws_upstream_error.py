"""An upstream WS ``error`` frame Codex doesn't map must still end the turn.

Regression for a Luna thread wedged since 05:41:42 (2026-10-05): upstream sent
response.created, a few events, then ``{type:error, error:{code,message}}`` with no
status. The shim forwarded that frame and stopped relaying; Codex only maps WS error
frames that carry a status (or the connection-limit code), so it kept waiting for a
terminal, and the shim's pings kept its idle timeout from firing. The errored
response was also stored in the turn cache.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from codex_shim.net.emitters import (
    WsRelayEmitter,
    codex_maps_ws_error,
    describe_ws_error,
    ws_error_code_message,
)
from ws_test_support import (
    MockUpstreamWsState,
    completed_events,
    install_chatgpt_auth,
    recv_until_terminal,
    start_shim_with_mock_chatgpt_ws,
)


@pytest.mark.parametrize(
    ("event", "mapped"),
    [
        ({"type": "error", "error": {"code": "server_error"}}, False),
        (
            {"type": "error", "code": "upstream_error", "message": "flat shim frame"},
            False,
        ),
        ({"type": "error", "status": 200, "error": {"code": "x"}}, False),
        ({"type": "error", "status": True, "error": {"code": "x"}}, False),
        ({"type": "error", "error": {"code": "x", "status": 429}}, False),
        ({"type": "error", "status": 429, "error": {"code": "x"}}, True),
        ({"type": "error", "status_code": 500, "error": {"code": "x"}}, True),
        (
            {"type": "error", "error": {"code": "websocket_connection_limit_reached"}},
            True,
        ),
        ({"type": "response.failed", "status": 500}, False),
    ],
)
def test_codex_maps_ws_error_mirrors_codex(event, mapped):
    assert codex_maps_ws_error(event) is mapped


def test_ws_error_code_message_and_description():
    wrapped = {
        "type": "error",
        "error": {"type": "server_error", "code": "boom", "message": "x" * 400},
    }
    assert ws_error_code_message(wrapped) == ("boom", "x" * 400)
    assert ws_error_code_message({"type": "error", "code": "c", "message": "m"}) == (
        "c",
        "m",
    )
    assert ws_error_code_message({"type": "error"}) == ("upstream_error", "")
    described = describe_ws_error(wrapped)
    assert "type='server_error'" in described and "code='boom'" in described
    assert len(described) < 400 and described.endswith("...'")


@pytest.mark.asyncio
async def test_emitter_complete_after_unmapped_error_fails_with_upstream_code():
    # Covers relays that keep reading after an error (SSE-to-WS http fallback).
    sent: list[dict] = []

    async def write_event(event: dict) -> None:
        sent.append(event)

    emitter = WsRelayEmitter(write_event, model="test")
    emitter.observe({"type": "response.created", "response": {"id": "r9"}})
    emitter.observe(
        {"type": "error", "error": {"code": "insufficient_quota", "message": "quota"}}
    )
    assert emitter.saw_terminal is False
    assert await emitter.complete() == "response.failed"
    assert [e["type"] for e in sent] == ["response.failed"]
    assert sent[0]["response"]["id"] == "r9"
    assert sent[0]["response"]["error"] == {
        "code": "insufficient_quota",
        "message": "quota",
    }
    assert await emitter.complete() == "response.failed"
    assert len(sent) == 1


@pytest.mark.asyncio
async def test_emitter_unmapped_error_then_upstream_failed_adds_nothing():
    sent: list[dict] = []

    async def write_event(event: dict) -> None:
        sent.append(event)

    emitter = WsRelayEmitter(write_event, model="test")
    emitter.observe({"type": "error", "error": {"code": "server_error"}})
    emitter.observe(
        {"type": "response.failed", "response": {"id": "r1", "status": "failed"}}
    )
    assert await emitter.complete() == "response.failed"
    assert sent == []


@pytest.mark.asyncio
async def test_mid_stream_upstream_error_ends_turn_without_advancing_cache_or_chain(
    monkeypatch, tmp_path, capsys
):
    install_chatgpt_auth(monkeypatch, tmp_path)
    errored = [
        {
            "type": "response.created",
            "response": {"id": "resp_errored", "model": "gpt-5.5"},
        },
        {
            "type": "response.in_progress",
            "response": {"id": "resp_errored", "model": "gpt-5.5"},
        },
        {
            "type": "response.output_text.delta",
            "item_id": "msg_1",
            "output_index": 0,
            "delta": "partial",
        },
        {
            "type": "error",
            "error": {"code": "server_error", "message": "upstream exploded"},
        },
    ]
    state = MockUpstreamWsState(
        response_sequences=[
            completed_events("resp_ok"),
            errored,
            completed_events("resp_retry"),
        ]
    )
    shim_client, upstream_client = await start_shim_with_mock_chatgpt_ws(
        monkeypatch, tmp_path, state
    )
    cache_dir = tmp_path / "chatgpt-conversations"
    tool_output = {"type": "function_call_output", "call_id": "call_1", "output": "ok"}
    try:
        ws = await shim_client.ws_connect(
            "/v1/responses", headers={"session-id": "ws-upstream-error"}
        )
        await ws.send_json(
            {
                "type": "response.create",
                "model": "codex-gpt-5-5",
                "input": [{"type": "message", "role": "user", "content": "hi"}],
            }
        )
        assert (await recv_until_terminal(ws, 2))[-1]["type"] == "response.completed"
        # The store runs after the terminal frame is written; wait for it.
        for _ in range(50):
            if list(cache_dir.rglob("resp_ok.json")):
                break
            await asyncio.sleep(0.02)
        assert list(cache_dir.rglob("resp_ok.json"))

        await ws.send_json(
            {
                "type": "response.create",
                "model": "codex-gpt-5-5",
                "previous_response_id": "resp_ok",
                "input": [tool_output],
            }
        )
        events = await recv_until_terminal(ws, 2)
        assert [e["type"] for e in events] == [
            "response.created",
            "response.in_progress",
            "response.output_text.delta",
            "error",
            "response.failed",
        ]
        assert events[-1]["response"]["error"] == {
            "code": "server_error",
            "message": "upstream exploded",
        }
        for _ in range(50):
            if state.closed_connections:
                break
            await asyncio.sleep(0.02)
        assert state.closed_connections == 1
        out = capsys.readouterr().out
        assert "upstream_error[" in out and "code='server_error'" in out
        assert "message='upstream exploded'" in out

        # Codex retries the same request: fresh lane, chain still anchored at resp_ok (expanded).
        await ws.send_json(
            {
                "type": "response.create",
                "model": "codex-gpt-5-5",
                "previous_response_id": "resp_ok",
                "input": [tool_output],
            }
        )
        assert (await recv_until_terminal(ws, 2))[-1]["type"] == "response.completed"
        assert len(state.handshakes) == 2
        retried = state.received_frames[2]
        assert "previous_response_id" not in retried
        assert retried["input"][-1]["call_id"] == "call_1"
        assert len(retried["input"]) > 1
        # Turns on one socket are handled in order, so once the retry is stored the errored
        # turn's handler has long finished: it must not have stored anything.
        for _ in range(50):
            if list(cache_dir.rglob("resp_retry.json")):
                break
            await asyncio.sleep(0.02)
        assert list(cache_dir.rglob("resp_retry.json"))
        assert not list(cache_dir.rglob("resp_errored.json"))
        await ws.close()
    finally:
        await shim_client.close()
        await upstream_client.close()


def test_codex_maps_flat_shim_error_frames_only_with_status():
    # The shim's own _write_ws_error frames always carry a status Codex maps.
    frame = json.loads(
        json.dumps(
            {
                "type": "error",
                "status": 502,
                "error": {"type": "x", "code": "x", "message": "m"},
            }
        )
    )
    assert codex_maps_ws_error(frame)

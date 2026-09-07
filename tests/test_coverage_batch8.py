from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from aiohttp.test_utils import make_mocked_request

from codex_shim import cli
from codex_shim.cursor_bridge import (
    BridgeError,
    BridgeNotAttachedError,
    CursorBridgeSession,
    cursor_bridge_registry,
)
from codex_shim.server import (
    ResponsesStreamState,
    ShimServer,
    _apply_cursor_stream_event,
)
from codex_shim.settings import ShimModel
from codex_shim.ws_passthrough import WsPassthroughConnectError


def _settings(tmp_path: Path) -> Path:
    path = tmp_path / "models.json"
    path.write_text("{}")
    return path


def _model(**kwargs) -> ShimModel:
    defaults = dict(
        slug="console-model",
        model="foo",
        display_name="Foo",
        provider="openai-responses",
        base_url="http://127.0.0.1:9/v1",
        api_key="k",
    )
    defaults.update(kwargs)
    return ShimModel(**defaults)


class _FakeStream:
    def __init__(self) -> None:
        self.chunks: list[bytes] = []

    async def write(self, data: bytes) -> None:
        self.chunks.append(data)


async def test_ws_passthrough_and_create_websocket_branches(monkeypatch, tmp_path):
    shim = ShimServer(_settings(tmp_path))
    request = make_mocked_request("POST", "/v1/responses")
    route = _model()
    target = ShimServer._WsPassthroughTarget(
        kind="openai_responses",
        requested_slug=route.slug,
        upstream_model=route.model,
        response_model_override=None,
        route=route,
    )

    class DeadPassthrough:
        async def connect_upstream(self, *args, **kwargs):
            raise WsPassthroughConnectError("nope")

    handled = await shim._handle_ws_passthrough_response_create(
        request, DeadPassthrough(), {"model": route.slug, "input": []}, target
    )
    assert handled is False

    async def resolve_none(self, payload):
        del payload
        return None

    async def chatgpt_http(self, request, ws, payload, target, *, http_session):
        del self, request, ws, payload, target, http_session

    monkeypatch.setattr(ShimServer, "_resolve_ws_passthrough_target", resolve_none)
    monkeypatch.setattr(ShimServer, "_handle_chatgpt_response_create_websocket_http", chatgpt_http)
    monkeypatch.setattr("codex_shim.server.is_chatgpt_passthrough_slug", lambda slug: True)
    monkeypatch.setattr("codex_shim.server.chatgpt_upstream_model", lambda slug: "gpt-5.6")
    auth = tmp_path / "auth.json"
    auth.write_text(json.dumps({"tokens": {"access_token": "tok", "account_id": "acct"}}))
    monkeypatch.setattr("codex_shim.server.DEFAULT_CODEX_AUTH", auth)

    class Ws:
        async def send_str(self, payload):
            del payload

    await shim._handle_response_create_websocket(request, Ws(), {"model": "codex-gpt-5-6-luna"})

    auth.write_text(json.dumps({"tokens": {}}))
    await shim._handle_response_create_websocket(request, Ws(), {"model": "codex-gpt-5-6-luna"})
    auth.write_text("{")
    await shim._handle_response_create_websocket(request, Ws(), {"model": "codex-gpt-5-6-luna"})

    async def resolve_target(self, payload):
        del payload
        return target

    async def handle_false(self, request, passthrough, payload, target):
        del request, passthrough, payload, target
        return False

    async def local_ws(self, request, ws, payload, *, http_session=None):
        del request, ws, payload, http_session

    monkeypatch.setattr(ShimServer, "_resolve_ws_passthrough_target", resolve_target)
    monkeypatch.setattr(ShimServer, "_handle_ws_passthrough_response_create", handle_false)
    monkeypatch.setattr(ShimServer, "_handle_local_response_create_websocket", local_ws)
    monkeypatch.setattr("codex_shim.server.ws_passthrough_enabled", lambda: True)
    await shim._handle_response_create_websocket(request, Ws(), {"model": route.slug})


async def test_cursor_bridge_invoke_wait_ingest_and_stream_events():
    stream = _FakeStream()
    state = ResponsesStreamState("cursor")
    await _apply_cursor_stream_event(state, stream, {"type": "segment_boundary"})
    await _apply_cursor_stream_event(
        state, stream, {"type": "tool_started", "call_id": "c1", "markdown": "run"}
    )
    await _apply_cursor_stream_event(
        state, stream, {"type": "tool_completed", "call_id": "c1", "markdown": "done"}
    )
    await _apply_cursor_stream_event(state, stream, {"type": "thinking_delta", "delta": "hmm"})
    await _apply_cursor_stream_event(state, stream, {"type": "thinking_completed"})
    await _apply_cursor_stream_event(
        state, stream, {"type": "connection_interrupted", "message": "cut"}
    )

    pending = ResponsesStreamState("m")
    pending.tool_calls[0] = {"id": "call_0", "name": "tool_search", "arguments": "", "emitted": False}
    await pending._chat_tool_delta(
        stream, {"index": 0, "function": {"name": "", "arguments": "{}"}}
    )

    session = CursorBridgeSession.create(allowed_tools=frozenset({"shell"}), tool_types={}, tool_resolve={})
    with pytest.raises(BridgeNotAttachedError):
        await session.invoke(tool="shell", arguments={"command": "echo"})
    session.attach_collector(SimpleNamespace(append_function_call=lambda **kwargs: None))
    session.mark_turn_closed()
    with pytest.raises(BridgeError, match="already completed"):
        await session.invoke(tool="shell", arguments={"command": "echo"})
    session.reopen_turn()
    accepted = await session.invoke(tool="shell", arguments={"command": "echo"})
    empty = await session.wait_job("")
    assert empty["error"] == "job_id_required"
    timed_out = await session.wait_job(accepted["job_id"], timeout_s=0)
    assert timed_out["error"] == "timeout"
    timed_out2 = await session.wait_job(accepted["job_id"], timeout_s=0.01)
    assert timed_out2["error"] == "timeout"
    session.complete_call(accepted["codex_call_id"], "ok")
    consumed = await session.wait_job(accepted["job_id"], timeout_s=0.01)
    assert consumed.get("ok") is True or consumed.get("error") == "unknown_job"
    missing = await session.wait_job("no-such")
    assert missing["error"] == "unknown_job"
    wait_default = await session.wait_job("still-missing")
    assert wait_default["error"] == "unknown_job"

    assert cursor_bridge_registry.ingest_function_call_outputs("nope") == 0
    assert cursor_bridge_registry.ingest_function_call_outputs(["skip", {"type": "message"}]) == 0
    assert cursor_bridge_registry.ingest_function_call_outputs(
        [{"type": "function_call_output", "call_id": ""}]
    ) == 0
    session2 = CursorBridgeSession.create(allowed_tools=frozenset({"shell"}), tool_types={}, tool_resolve={})
    session2.attach_collector(SimpleNamespace(append_function_call=lambda **kwargs: None))
    await cursor_bridge_registry.register(session2)
    try:
        job = await session2.invoke(tool="shell", arguments={"command": "echo"})
        cursor_bridge_registry.unindex_call(job["codex_call_id"])
        ingested = cursor_bridge_registry.ingest_function_call_outputs(
            [{"type": "function_call_output", "call_id": job["codex_call_id"], "output": "out"}]
        )
        assert ingested == 1
        assert cursor_bridge_registry.cancel_sessions_for_call_ids("nope") == 0
        assert cursor_bridge_registry.cancel_sessions_for_call_ids(["skip", {"type": "message"}]) == 0
        assert cursor_bridge_registry.cancel_sessions_for_call_ids(
            [{"type": "function_call_output", "call_id": ""}]
        ) == 0
        other = await session2.invoke(tool="shell", arguments={"command": "echo2"})
        cancelled = cursor_bridge_registry.cancel_sessions_for_call_ids(
            [
                {"type": "function_call_output", "call_id": other["codex_call_id"]},
                {"type": "function_call_output", "call_id": other["codex_call_id"]},
            ]
        )
        assert cancelled == 0
    finally:
        cursor_bridge_registry.close(session2.bridge_id)


def test_cli_isolated_paths(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "SYSTEMD_USER_UNIT", tmp_path / "missing-unit")
    monkeypatch.setattr(cli, "PID_PATH", tmp_path / "shim.pid")
    monkeypatch.setattr(cli, "CODEX_CONFIG_PATH", tmp_path / "config.toml")
    monkeypatch.setattr(cli, "CODEX_CONFIG_BACKUP_PATH", tmp_path / "backup.toml")
    monkeypatch.setattr(cli, "_stop_systemd_unit_if_active", lambda: False)
    monkeypatch.setattr(cli, "_read_pid", lambda: None)
    monkeypatch.setattr(cli, "_pid_running", lambda pid: False)
    monkeypatch.setattr(cli, "_health", lambda port: None)
    assert cli.stop() == 0

    monkeypatch.setattr(cli, "_wait_for_pid_exit", lambda pid, timeout: True)
    monkeypatch.setattr(cli, "_wait_for_port_free", lambda port, timeout: True)
    monkeypatch.setattr(cli, "_terminate_pid", lambda pid: None)
    assert cli._stop_pid(1234, 8767) is True

    monkeypatch.setattr(cli, "CATALOG_PATH", tmp_path / "catalog.json")
    monkeypatch.setattr(cli, "CONFIG_PATH", tmp_path / "runtime-config.toml")
    monkeypatch.setattr(cli, "RUNTIME_DIR", tmp_path)

    def _no_default(*_args, **_kwargs):
        raise ValueError("no models")

    monkeypatch.setattr(cli, "default_model_slug", _no_default)
    monkeypatch.setattr(cli, "_load_models", lambda path: [])
    with pytest.raises(SystemExit):
        cli.generate(tmp_path / "models.json", 8767)
    with pytest.raises(SystemExit):
        cli._resolve_model_slug([], None)
    with pytest.raises(SystemExit, match="Unknown"):
        cli._resolve_model_slug([], "nope")
    models = [
        _model(slug="a", model="m", display_name="Same Name", api_key="k", provider="openai"),
        _model(slug="b", model="n", display_name="Same Name", api_key="k", provider="openai"),
        _model(slug="c", model="only", display_name="Only One", api_key="", provider="openai"),
    ]
    with pytest.raises(SystemExit, match="Ambiguous"):
        cli._resolve_model_slug(models, "Same")
    assert cli._resolve_model_slug(models, "only") == "c"
    monkeypatch.setattr(cli, "_current_managed_model", lambda: "missing")
    with pytest.raises(SystemExit):
        cli._resolve_model_slug([], None)

    config = tmp_path / "config.toml"
    config.write_text('model_provider = "other"\nopenai_base_url = "http://example/v1"\n')
    monkeypatch.setattr(cli, "CODEX_CONFIG_PATH", config)
    checks = cli._doctor_codex_config(8767)
    assert any(check.status in {"WARN", "FAIL", "INFO"} for check in checks)
    blocked = tmp_path / "blocked"
    blocked.mkdir()
    monkeypatch.setattr(cli, "CODEX_CONFIG_PATH", blocked)
    unread = cli._doctor_codex_config(8767)
    assert unread[0].status in {"WARN", "INFO"}

    live = tmp_path / "live.toml"
    live.write_text("model = \"x\"\n")
    backup = tmp_path / "backup.toml"
    backup.write_text("model = \"old\"\n")
    monkeypatch.setattr(cli, "CODEX_CONFIG_PATH", live)
    monkeypatch.setattr(cli, "CODEX_CONFIG_BACKUP_PATH", backup)
    cli.restore_codex_config()
    assert not backup.exists()

    settings = tmp_path / "models.json"
    settings.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "model": "llama",
                        "provider": "openai",
                        "base_url": "http://127.0.0.1:1/v1",
                        "api_key": "k",
                        "slug": "local-llama",
                    }
                ],
                "router": {"enabled": True, "slug": "auto", "candidates": [{"slug": "missing"}]},
            }
        )
    )
    monkeypatch.setattr(cli, "_active_router", lambda models, path: None)
    doctor = cli._doctor_settings(settings)
    assert any("inactive" in check.message or "disabled" in check.message or "auto router" in check.message for check in doctor)

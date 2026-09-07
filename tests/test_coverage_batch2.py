from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer, make_mocked_request

from codex_shim import cli
from codex_shim.compaction.orchestrator import CompactionOrchestratorError
from codex_shim.compaction.pipeline import PreparedInput
from codex_shim.compaction.types import CompactionRequest
from codex_shim.cursor_bridge import (
    DEFAULT_BRIDGE_TTL_S,
    DEFAULT_BRIDGE_WAIT_TIMEOUT_MS,
    DEFAULT_SUFFIX_TOOL_LIST_CAP,
    bridge_tool_list_cap,
    bridge_ttl_seconds,
    bridge_wait_timeout_ms,
    shim_port_from_request_host,
)
from codex_shim.cursor_passthrough import (
    CursorStreamParser,
    build_cursor_prompt,
    resolve_cursor_workspace,
    _format_glob_result,
    _format_grep_result,
)
from codex_shim.server import (
    ShimServer,
    _as_compact_response,
    _compaction_orchestrator_error_response,
    _restart_codex_app,
    _stream_responses_error_from_http_response,
    _stream_responses_upstream_error,
)
from codex_shim.settings import ShimModel


def _settings(tmp_path: Path) -> Path:
    settings = tmp_path / "models.json"
    settings.write_text("{}")
    return settings


def _prepared() -> PreparedInput:
    return PreparedInput(
        native_input=[{"type": "message", "role": "user", "content": "hi"}],
        summarization_input=[{"type": "message", "role": "assistant", "content": "ok"}],
        previous_summary=None,
    )


def _llama() -> ShimModel:
    return ShimModel(
        slug="local-llama",
        model="llama",
        display_name="Local Llama",
        provider="openai",
        base_url="http://example.invalid/v1",
        api_key="secret",
    )


def _claude() -> ShimModel:
    return ShimModel(
        slug="claude-local",
        model="claude-3",
        display_name="Claude",
        provider="anthropic",
        base_url="http://example.invalid/v1",
        api_key="secret",
    )


def _console() -> ShimModel:
    return ShimModel(
        slug="console-model",
        model="gpt-x",
        display_name="Console",
        provider="openai-responses",
        base_url="http://example.invalid/v1",
        api_key="secret",
    )


async def test_compaction_summarization_chatgpt_and_byok(monkeypatch, tmp_path):
    shim = ShimServer(_settings(tmp_path))
    request = CompactionRequest(
        http_request=make_mocked_request("POST", "/v1/responses/compact", headers={"session-id": "s1"}),
        body={"model": "codex-gpt-5-5", "input": [{"type": "message", "role": "user", "content": "hi"}]},
        stripped_input=[{"type": "message", "role": "user", "content": "hi"}],
        requested_slug="codex-gpt-5-5",
        provider="chatgpt",
        session_key="s1",
        route=_llama(),
    )

    async def chatgpt_ok(self, *args, **kwargs):
        del self, args, kwargs
        return web.json_response(
            {
                "output": [{"type": "message", "content": [{"type": "output_text", "text": "chatgpt summary"}]}],
                "usage": {"input_tokens": 1, "output_tokens": 2, "input_tokens_details": {"cached_tokens": 4}},
            }
        )

    monkeypatch.setattr(ShimServer, "_chatgpt_passthrough", chatgpt_ok)
    summarized = await shim._compaction_summarization_chatgpt(request, _prepared(), "native failed")
    assert summarized.summary == "chatgpt summary"
    assert summarized.usage["output_tokens"] == 2

    async def fake_route(self, body):
        del self
        if body.get("model") == "claude-local":
            return _claude()
        return _llama()

    async def fetch_ok(self, *args, **kwargs):
        del self, args, kwargs
        return "byok summary", {"input_tokens": 3, "output_tokens": 1}, None

    async def fetch_err(self, *args, **kwargs):
        del self, args, kwargs
        return "", None, web.Response(text="fail", status=502)

    monkeypatch.setattr(ShimServer, "_route", fake_route)
    monkeypatch.setattr(ShimServer, "_fetch_byok_compact_summary", fetch_ok)
    byok = await shim._compaction_summarization_byok(request, _prepared(), "native failed")
    assert byok.summary == "byok summary"

    monkeypatch.setattr(ShimServer, "_fetch_byok_compact_summary", fetch_err)
    byok_err = await shim._compaction_summarization_byok(request, _prepared(), "native failed")
    assert byok_err.error_response.status == 502

    tertiary = await shim._compaction_tertiary_byok(request, _prepared(), "native failed", "claude-local")
    assert tertiary.error_response.status == 502
    monkeypatch.setattr(ShimServer, "_fetch_byok_compact_summary", fetch_ok)
    tertiary_ok = await shim._compaction_tertiary_byok(request, _prepared(), "native failed", "local-llama")
    assert tertiary_ok.summary == "byok summary"


async def test_compaction_native_chatgpt_http_error_enriches_warnings(monkeypatch, tmp_path):
    shim = ShimServer(_settings(tmp_path))
    request = CompactionRequest(
        http_request=make_mocked_request("POST", "/v1/responses/compact", headers={"session-id": "s1"}),
        body={"model": "codex-gpt-5-5", "input": []},
        stripped_input=[],
        requested_slug="codex-gpt-5-5",
        provider="chatgpt",
        session_key="s1",
    )

    async def native_400(self, *args, **kwargs):
        del self, args, kwargs
        return web.Response(
            text='{"error":{"message":"No tool call found for function call output"}}',
            status=400,
        )

    monkeypatch.setattr(ShimServer, "_post_chatgpt_native_compact", native_400)
    result = await shim._compaction_native_chatgpt(request, _prepared())
    assert result.native_status == 400
    assert result.error_response.status == 400


async def test_dispatch_byok_compact_responses_provider_branches(monkeypatch, tmp_path):
    shim = ShimServer(_settings(tmp_path))
    monkeypatch.setattr(ShimServer, "_apply_responses_input_pipeline", lambda self, body, request, **kwargs: body)
    request = make_mocked_request("POST", "/v1/responses/compact", headers={"session-id": "s1"})

    async def route_for(self, body):
        del self
        model = body.get("model")
        if model == "claude-local":
            return _claude()
        if model == "console-model":
            return _console()
        if model == "mystery":
            return ShimModel(
                slug="mystery",
                model="x",
                display_name="X",
                provider="mystery",
                base_url="http://x",
                api_key="k",
            )
        return _llama()

    async def post_responses(self, request, route, body):
        del self, request, route, body
        return web.json_response({"id": "resp_1", "output": []})

    async def post_chat(self, *args, **kwargs):
        del args, kwargs
        return web.json_response(
            {"output": [{"type": "message", "content": [{"type": "output_text", "text": "chat compact"}]}]}
        )

    async def post_anthropic(self, *args, **kwargs):
        del args, kwargs
        return web.json_response(
            {"output": [{"type": "message", "content": [{"type": "output_text", "text": "claude compact"}]}]}
        )

    monkeypatch.setattr(ShimServer, "_route", route_for)
    monkeypatch.setattr(ShimServer, "_post_openai_responses", post_responses)
    monkeypatch.setattr(ShimServer, "_post_openai_chat", post_chat)
    monkeypatch.setattr(ShimServer, "_post_anthropic", post_anthropic)

    responses = await shim._dispatch_byok_compact_responses(request, {"model": "console-model", "input": []})
    assert responses.status == 200
    chat = await shim._dispatch_byok_compact_responses(request, {"model": "local-llama", "input": []})
    assert chat.status == 200
    anthropic = await shim._dispatch_byok_compact_responses(request, {"model": "claude-local", "input": []})
    assert anthropic.status == 200
    with pytest.raises(web.HTTPBadGateway):
        await shim._dispatch_byok_compact_responses(request, {"model": "mystery", "input": []})


async def test_as_compact_and_orchestrator_error_helpers():
    passthrough = web.Response(text="fail", status=502)
    assert await _as_compact_response(passthrough, "local") is passthrough
    invalid = await _as_compact_response(web.Response(text="{nope", status=200), "local")
    assert invalid.text == "{nope"
    compacted = await _as_compact_response(
        web.json_response(
            {
                "output": [{"type": "message", "content": [{"type": "output_text", "text": "sum"}]}],
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }
        ),
        "local-llama",
    )
    payload = json.loads(compacted.text)
    assert payload["model"] == "local-llama"

    err = _compaction_orchestrator_error_response(
        "local-llama",
        CompactionOrchestratorError(
            "boom",
            error_response=web.Response(text='{"error":{"code":"invalid_request_error","message":"nope"}}', status=400),
        ),
    )
    assert err.status == 400
    body = json.loads(err.text)
    assert "boom" in body["error"]["message"]


async def test_stream_responses_error_wrappers():
    class Upstream:
        status = 429
        headers = {}

        async def text(self):
            return '{"error":{"message":"slow down"}}'

        def release(self):
            pass

    async def handler(request):
        return await _stream_responses_upstream_error(request, "local-llama", Upstream(), slug="local-llama")

    async def http_handler(request):
        return await _stream_responses_error_from_http_response(
            request,
            "local-llama",
            web.Response(text='{"error":{"message":"nope"}}', status=400),
            slug="local-llama",
        )

    app = web.Application()
    app.router.add_post("/up", handler)
    app.router.add_post("/http", http_handler)
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        up = await client.post("/up")
        assert "slow down" in await up.text()
        http = await client.post("/http")
        assert "nope" in await http.text()
    finally:
        await client.close()


async def test_chatgpt_passthrough_auth_missing_and_empty_token(monkeypatch, tmp_path):
    missing = tmp_path / "missing-auth.json"
    monkeypatch.setattr("codex_shim.server.DEFAULT_CODEX_AUTH", missing)
    shim = ShimServer(_settings(tmp_path))
    request = make_mocked_request("POST", "/v1/responses", headers={"session-id": "s1"})
    with pytest.raises(web.HTTPUnauthorized) as missing_exc:
        await shim._chatgpt_passthrough(
            request,
            {"model": "codex-gpt-5-5", "input": "hi"},
            allow_byok_fallback=False,
        )
    assert "auth.json not found" in missing_exc.value.text

    empty = tmp_path / "empty-auth.json"
    empty.write_text(json.dumps({"tokens": {}}))
    monkeypatch.setattr("codex_shim.server.DEFAULT_CODEX_AUTH", empty)
    with pytest.raises(web.HTTPUnauthorized) as empty_exc:
        await shim._chatgpt_passthrough(
            request,
            {"model": "codex-gpt-5-5", "input": "hi"},
            allow_byok_fallback=False,
        )
    assert "no access_token" in empty_exc.value.text


def test_content_and_latest_user_text_helpers(tmp_path):
    shim = ShimServer(_settings(tmp_path))
    assert shim._content_to_debug_text(None) == ""
    assert shim._content_to_debug_text("hi") == "hi"
    assert shim._content_to_debug_text([{"text": "a"}, "b", {"content": "c"}]) == "a\nb\nc"
    assert shim._content_to_debug_text({"text": "d"}) == "d"
    assert shim._content_to_debug_text(12) == "12"
    assert shim._latest_user_text({"input": "plain"}) == "plain"
    assert shim._latest_user_text({"input": {"role": "user"}}) == ""
    assert shim._latest_user_text({"input": [{"role": "user", "content": "hello"}]}) == "hello"
    assert shim._latest_user_text({"input": [{"type": "input_text", "text": "tail"}]}) == "tail"
    assert shim._has_image_generation_history({"input": [{"type": "image_generation_call"}]}) is True
    assert shim._has_image_generation_history({"input": "nope"}) is False


def test_restart_codex_app_windows_and_macos(monkeypatch, tmp_path):
    class ImmediateThread:
        def __init__(self, target, daemon=True):
            self.target = target
            del daemon

        def start(self):
            self.target()

    runs: list[list[str]] = []
    pops: list[list[str]] = []
    created = {"exe": False}

    class FakePath:
        def __init__(self, *parts):
            self._value = "/".join(str(part) for part in parts)

        def __truediv__(self, other):
            return FakePath(self._value, str(other))

        def __str__(self):
            return self._value

        def exists(self):
            return created["exe"] and self._value.endswith("Codex.exe")

    monkeypatch.setattr("codex_shim.server.Path", FakePath)
    monkeypatch.setattr("threading.Thread", ImmediateThread)
    monkeypatch.setattr("time.sleep", lambda _s: None)
    monkeypatch.setattr(
        "subprocess.run",
        lambda cmd, check=False, stdout=None, stderr=None: runs.append(list(cmd)) or SimpleNamespace(returncode=0),
    )
    monkeypatch.setattr(
        "subprocess.Popen",
        lambda cmd, shell=False: pops.append(list(cmd) if not isinstance(cmd, str) else [cmd]) or SimpleNamespace(pid=1),
    )

    monkeypatch.setattr("os.name", "nt")
    monkeypatch.setattr("codex_shim.server.sys.platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    _restart_codex_app()
    assert any(cmd[:1] == ["taskkill"] for cmd in runs)
    assert pops[-1] == ["Codex.exe"]

    created["exe"] = True
    _restart_codex_app()
    assert pops[-1][0].endswith("Codex.exe")
    assert pops[-1] != ["Codex.exe"]

    monkeypatch.setattr("os.name", "posix")
    monkeypatch.setattr("codex_shim.server.sys.platform", "darwin")
    _restart_codex_app()
    assert any(cmd[:1] == ["osascript"] for cmd in runs)
    assert pops[-1] == ["open", "-a", "Codex"]

    monkeypatch.setattr(
        "subprocess.run",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("nope")),
    )
    _restart_codex_app()


def test_bridge_env_helpers_and_host_port(monkeypatch):
    monkeypatch.delenv("CODEX_SHIM_CURSOR_BRIDGE_TTL_S", raising=False)
    assert bridge_ttl_seconds() == DEFAULT_BRIDGE_TTL_S
    monkeypatch.setenv("CODEX_SHIM_CURSOR_BRIDGE_TTL_S", "nope")
    assert bridge_ttl_seconds() == DEFAULT_BRIDGE_TTL_S
    monkeypatch.setenv("CODEX_SHIM_CURSOR_BRIDGE_TTL_S", "12.5")
    assert bridge_ttl_seconds() == 12.5

    monkeypatch.delenv("CODEX_SHIM_CURSOR_BRIDGE_TOOL_LIST_CAP", raising=False)
    assert bridge_tool_list_cap() == DEFAULT_SUFFIX_TOOL_LIST_CAP
    monkeypatch.setenv("CODEX_SHIM_CURSOR_BRIDGE_TOOL_LIST_CAP", "bad")
    assert bridge_tool_list_cap() == DEFAULT_SUFFIX_TOOL_LIST_CAP
    monkeypatch.setenv("CODEX_SHIM_CURSOR_BRIDGE_TOOL_LIST_CAP", "7")
    assert bridge_tool_list_cap() == 7

    monkeypatch.delenv("CODEX_SHIM_CURSOR_BRIDGE_WAIT_TIMEOUT_MS", raising=False)
    assert bridge_wait_timeout_ms() == DEFAULT_BRIDGE_WAIT_TIMEOUT_MS
    monkeypatch.setenv("CODEX_SHIM_CURSOR_BRIDGE_WAIT_TIMEOUT_MS", "x")
    assert bridge_wait_timeout_ms() == DEFAULT_BRIDGE_WAIT_TIMEOUT_MS
    monkeypatch.setenv("CODEX_SHIM_CURSOR_BRIDGE_WAIT_TIMEOUT_MS", "2500")
    assert bridge_wait_timeout_ms() == 2500

    assert shim_port_from_request_host("") == 8765
    assert shim_port_from_request_host("[::1]:9999") == 9999
    assert shim_port_from_request_host("[::1]:bad") == 8765
    assert shim_port_from_request_host("[::1]") == 8765
    assert shim_port_from_request_host("127.0.0.1:9001") == 9001
    assert shim_port_from_request_host("127.0.0.1:bad") == 8765
    assert shim_port_from_request_host("localhost") == 8765


def test_resolve_workspace_and_cursor_prompt(monkeypatch):
    monkeypatch.setenv("CODEX_SHIM_CURSOR_WORKSPACE", "/tmp/override")
    assert resolve_cursor_workspace({"metadata": {"cwd": "/tmp/meta"}}) == "/tmp/override"
    monkeypatch.delenv("CODEX_SHIM_CURSOR_WORKSPACE")
    assert resolve_cursor_workspace({"metadata": {"cwd": "/tmp/meta"}}) == "/tmp/meta"
    assert resolve_cursor_workspace({}, request_headers={"x-codex-turn-metadata": "{not-json"}) != "/tmp/meta"
    assert (
        resolve_cursor_workspace(
            {},
            request_headers={"x-codex-turn-metadata": json.dumps({"workspace": "/tmp/from-header"})},
        )
        == "/tmp/from-header"
    )
    assert resolve_cursor_workspace({"instructions": "cwd: /tmp/from-instructions"}) == "/tmp/from-instructions"
    assert resolve_cursor_workspace({}, prompt="Current working directory: /tmp/from-prompt") == "/tmp/from-prompt"

    prompt = build_cursor_prompt(
        {
            "model": "cursor-composer-2-5",
            "instructions": "System",
            "input": [
                {"type": "message", "role": "user", "content": "hi"},
                {"type": "function_call", "name": "shell", "arguments": "{}", "call_id": "c1"},
            ],
        }
    )
    assert "[USER]" in prompt or "[SYSTEM]" in prompt


def test_glob_grep_formatters_and_stream_parser_junk():
    assert _format_glob_result({"result": {}}) == ""
    assert "a.py" in _format_glob_result({"result": {"success": {"files": ["a.py"] * 21}}})
    assert "3 file" in _format_glob_result({"result": {"success": {"totalFiles": 3}}})
    grep = _format_grep_result(
        {
            "result": {
                "success": {
                    "workspaceResults": {
                        "/repo": {
                            "content": {
                                "matches": [
                                    {"file": "cli.py", "matches": [{"lineNumber": 12, "content": "def main"}]},
                                    "skip",
                                ]
                            }
                        },
                        "/other": "skip",
                    }
                }
            }
        }
    )
    assert "cli.py:12" in grep

    parser = CursorStreamParser()
    assert parser.feed_events("") == []
    assert parser.feed_events("{nope") == []
    assert parser.feed_events("[]") == []
    assert parser.feed_events(json.dumps({"type": "tool_call", "subtype": "started"}))[0]["type"] == "segment_boundary"


def test_resolve_model_slug_and_override_args(monkeypatch, tmp_path):
    models = [_llama()]
    monkeypatch.setattr(cli, "chatgpt_passthrough_available", lambda *a, **k: False)
    monkeypatch.setattr(cli, "cursor_passthrough_available", lambda *a, **k: False)
    monkeypatch.setattr(cli, "_current_managed_model", lambda: "local-llama")
    assert cli._resolve_model_slug(models, None) == "local-llama"
    assert cli._resolve_model_slug(models, "local-llama") == "local-llama"
    assert cli._resolve_model_slug(models, "llama") == "local-llama"
    assert cli._resolve_model_slug(models, "Local") == "local-llama"
    with pytest.raises(SystemExit, match="Unknown shim model"):
        cli._resolve_model_slug(models, "nope")

    router = SimpleNamespace(slug="auto-router")
    assert cli._resolve_model_slug(models, "auto-router", router_config=router) == "auto-router"

    monkeypatch.setattr(cli, "_current_managed_model", lambda: "missing")
    monkeypatch.setattr(cli, "default_model_slug", lambda models, include_chatgpt=None: models[0].slug)
    assert cli._resolve_model_slug(models, None) == "local-llama"

    monkeypatch.setattr(cli, "is_chatgpt_passthrough_slug", lambda slug: slug.startswith("codex-"))
    with pytest.raises(SystemExit, match="Codex login"):
        cli._resolve_model_slug(models, "codex-gpt-5-5")
    monkeypatch.setattr(cli, "chatgpt_passthrough_available", lambda *a, **k: True)
    monkeypatch.setattr(cli, "chatgpt_catalog_slug", lambda slug: "openai-gpt-5-5")
    assert cli._resolve_model_slug(models, "codex-gpt-5-5") == "openai-gpt-5-5"

    monkeypatch.setattr(
        cli,
        "is_cursor_passthrough_slug",
        lambda slug: slug.startswith("cursor-") or slug == "composer-2.5",
    )
    monkeypatch.setattr(cli, "cursor_passthrough_available", lambda *a, **k: False)
    with pytest.raises(SystemExit, match="Cursor CLI"):
        cli._resolve_model_slug(models, "cursor-composer-2-5")
    monkeypatch.setattr(cli, "cursor_passthrough_available", lambda *a, **k: True)
    monkeypatch.setattr(cli, "cursor_passthrough_display_names", lambda: {"cursor-composer-2-5": "Composer"})
    assert cli._resolve_model_slug(models, "cursor-composer-2-5") == "cursor-composer-2-5"
    monkeypatch.setattr(cli, "cursor_passthrough_display_names", lambda: {})
    monkeypatch.setattr(
        cli,
        "cursor_catalog_models",
        lambda: [SimpleNamespace(catalog_slug="cursor-composer-2-5", upstream_id="composer-2.5")],
    )
    assert cli._resolve_model_slug(models, "composer-2.5") == "cursor-composer-2-5"
    monkeypatch.setattr(cli, "cursor_catalog_models", lambda: [])
    monkeypatch.setattr(cli, "cursor_upstream_model", lambda slug: "composer.mapped")
    assert cli._resolve_model_slug(models, "cursor-unknown") == "composer.mapped"

    monkeypatch.setattr(cli, "_load_models", lambda path: models)
    monkeypatch.setattr(cli, "codex_config_overrides", lambda catalog, slug, port: [f"model={slug}", f"port={port}"])
    args = cli._override_args(tmp_path / "models.json", 8767)
    assert "-c" in args


def test_discover_models_status_pid_and_runtime(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, "chatgpt_passthrough_available", lambda *a, **k: False)
    monkeypatch.setattr(cli, "cursor_passthrough_available", lambda *a, **k: False)
    missing = tmp_path / "missing.json"
    assert cli.discover_models(missing) == 1
    assert "No discoverable models" in capsys.readouterr().out

    bad = tmp_path / "bad.json"
    bad.write_text("{")
    with pytest.raises(SystemExit, match="not valid JSON"):
        cli.discover_models(bad)

    class BoomSettings:
        def __init__(self, path):
            del path

        def load_explicit(self):
            raise FileNotFoundError("gone")

    ok = tmp_path / "ok.json"
    ok.write_text("{}")
    monkeypatch.setattr(cli, "ModelSettings", BoomSettings)
    assert cli.discover_models(ok) == 1

    monkeypatch.setattr(cli, "ModelSettings", lambda path: SimpleNamespace(load_explicit=lambda: []))
    monkeypatch.setattr(cli, "discover_summary", lambda explicit, settings_data=None: [])
    monkeypatch.setattr(cli, "cursor_passthrough_available", lambda *a, **k: True)
    monkeypatch.setattr(
        cli,
        "cursor_catalog_models",
        lambda force_refresh=False: [
            SimpleNamespace(catalog_slug="cursor-auto", upstream_id="auto", display_name="Auto")
        ],
    )
    monkeypatch.setattr(cli, "chatgpt_passthrough_available", lambda *a, **k: True)
    monkeypatch.setattr(cli, "chatgpt_passthrough_display_names", lambda: {"codex-gpt-5-5": "GPT-5.5"})
    monkeypatch.setattr(cli, "_refresh_published_catalog", lambda *args, **kwargs: None)
    assert cli.discover_models(ok, refresh=True) == 0
    out = capsys.readouterr().out
    assert "cursor-auto" in out
    assert "codex-gpt-5-5" in out

    monkeypatch.setattr(cli, "_read_pid", lambda: 99)
    monkeypatch.setattr(cli, "_pid_running", lambda pid: True)
    monkeypatch.setattr(cli, "_health", lambda port: {"models": 3})
    assert cli.status(8767) == 0
    monkeypatch.setattr(cli, "_health", lambda port: None)
    assert cli.status(8767) == 1
    monkeypatch.setattr(cli, "_pid_running", lambda pid: False)
    monkeypatch.setattr(cli, "_health", lambda port: {"models": 2})
    pid_path = tmp_path / "shim.pid"
    pid_path.write_text("1")
    monkeypatch.setattr(cli, "PID_PATH", pid_path)
    assert cli.status(8767) == 0
    assert not pid_path.exists()
    monkeypatch.setattr(cli, "_health", lambda port: None)
    assert cli.status(8767) == 1


def test_pid_running_posix_and_wait_listener(monkeypatch):
    monkeypatch.setattr(cli.os, "name", "posix")
    assert cli._pid_running(os.getpid()) is True
    assert cli._pid_running(99_999_999) is False

    monkeypatch.setattr(cli, "_port_is_free", lambda port: False)
    monkeypatch.setattr(cli, "_SHUTDOWN_POLL_INTERVAL_S", 0)
    assert cli._wait_for_port_free(1, 0) is False

    monkeypatch.setattr(cli.os, "name", "nt")
    assert cli._listener_pid(8767) is None
    monkeypatch.setattr(cli.os, "name", "posix")
    monkeypatch.setattr(cli.shutil, "which", lambda name: None)
    assert cli._listener_pid(8767) is None
    monkeypatch.setattr(cli.shutil, "which", lambda name: "/usr/bin/ss")
    monkeypatch.setattr(
        cli.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="users:((python3,pid=4242,fd=3))\n"),
    )
    assert cli._listener_pid(8767) == 4242


def test_upsert_key_in_section_and_patched_bundle(tmp_path, monkeypatch):
    text = "[other]\nfoo = 1\n\n[features]\nkeep = true\n"
    updated = cli._upsert_key_in_section(text, "features", "enable_x", "enable_x = true")
    assert "enable_x = true" in updated
    replaced = cli._upsert_key_in_section(updated, "features", "enable_x", "enable_x = false")
    assert "enable_x = false" in replaced
    appended = cli._upsert_key_in_section("[alpha]\na = 1\n", "features", "k", "k = 1")
    assert "[features]" in appended

    user = tmp_path / "user" / "Codex.app"
    system = tmp_path / "system" / "Codex.app"
    monkeypatch.setattr(cli, "USER_CODEX_APP", user)
    monkeypatch.setattr(cli, "SYSTEM_CODEX_APP", system)
    assert cli.patched_codex_app_bundle() is None
    asar = user / "Contents/Resources/app.asar"
    asar.parent.mkdir(parents=True)
    asar.write_bytes(b"asar")
    monkeypatch.setattr(cli, "_app_asar_is_patched", lambda path: path == asar)
    assert cli.patched_codex_app_bundle() == user


def test_doctor_runtime_files_catalog_variants(monkeypatch, tmp_path):
    catalog = tmp_path / "custom_model_catalog.json"
    monkeypatch.setattr(cli, "CATALOG_PATH", catalog)
    monkeypatch.setattr(cli, "CONFIG_PATH", tmp_path / "config.toml")
    monkeypatch.setattr(cli, "PID_PATH", tmp_path / "shim.pid")
    monkeypatch.setattr(cli, "LOG_PATH", tmp_path / "shim.log")
    checks = cli._doctor_runtime_files()
    assert any("catalog missing" in check.message for check in checks)
    catalog.write_text("{")
    checks = cli._doctor_runtime_files()
    assert any(check.status == "WARN" for check in checks)
    catalog.write_text(json.dumps({"models": [{"slug": "a"}]}))
    checks = cli._doctor_runtime_files()
    assert any("catalog models: 1" in check.message for check in checks)

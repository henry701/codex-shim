from __future__ import annotations

import json
from types import SimpleNamespace
import pytest

from codex_shim import cli
from codex_shim.cursor_bridge import (
    BridgeToolNotAllowedError,
    BridgeToolSpec,
    CursorBridgeSession,
    _compact_parameters,
    _format_tool_catalog,
    _iter_bridge_tool_candidates,
    tool_output_call_ids,
)
from codex_shim.cursor_passthrough import cursor_passthrough_available
from codex_shim.discover import (
    _models_dev_model_cost_input,
    _parse_chatgpt_codex_models_payload,
    _parse_models_dev_opencode_free_ids,
    _parse_models_dev_opencode_paid_ids,
    _positive_int,
    lookup_models_dev_row,
)
from codex_shim.opencode_go import (
    _json_payload,
    _read_settings_json,
    fetch_opencode_go_model_ids,
    opencode_go_model_row,
    probe_chat_model,
    probe_messages_model,
    refresh_opencode_go_settings,
)
from codex_shim.server import main as server_main
from codex_shim.settings import ShimModel
from codex_shim.translate import (
    anthropic_messages_to_chat,
    is_hosted_codex_tool,
    responses_to_chat,
    sanitize_function_call_arguments,
    tighten_json_schema_required,
    unwrap_custom_tool_input,
    wrap_custom_tool_input,
    _anthropic_content_to_chat_parts,
    _anthropic_image_block_to_chat_part,
    _anthropic_tool_choice_to_chat,
    _chat_image_part_to_anthropic,
    _chat_parts_from_content,
    _sanitize_chat_content_parts,
    _web_search_call_item,
)


def test_translate_schema_tools_and_history_edges():
    assert tighten_json_schema_required([{"type": "string"}]) == [{"type": "string"}]
    assert tighten_json_schema_required("nope") == "nope"
    nested = tighten_json_schema_required(
        {
            "type": "object",
            "anyOf": [{"type": "object", "properties": {"a": {"type": "string"}}}],
            "items": {"type": "object", "properties": {"b": {"type": "number"}}},
            "additionalProperties": {"type": "object", "properties": {"c": {"type": "boolean"}}},
            "$defs": {"d": {"type": "object", "properties": {"e": {"type": "string"}}}},
            "properties": {"keep": {"type": "string"}, "drop": {"type": "number"}},
            "required": ["keep", 12],
        }
    )
    assert "keep" in nested["properties"]
    assert "drop" not in nested["properties"]
    empty_object = tighten_json_schema_required({"type": "object"})
    assert empty_object["required"] == []

    assert is_hosted_codex_tool("nope") is False
    assert is_hosted_codex_tool({"type": "web_search"}) is True
    assert is_hosted_codex_tool({"type": "computer_use_preview"}) is True
    assert is_hosted_codex_tool({"type": "image_generation"}) is True
    assert is_hosted_codex_tool({"function": {"name": "web_search"}}) is True

    assert sanitize_function_call_arguments({"a": 1.0}) == '{"a": 1}'
    assert sanitize_function_call_arguments(12) == 12
    assert sanitize_function_call_arguments("   ") == "{}"
    assert sanitize_function_call_arguments("not-json") == "not-json"
    assert sanitize_function_call_arguments("{nope") == "{nope"

    assert unwrap_custom_tool_input({"input": "patch"}) == "patch"
    assert unwrap_custom_tool_input("plain") == "plain"
    assert unwrap_custom_tool_input("{nope") == "{nope"
    assert unwrap_custom_tool_input(None) == ""
    assert json.loads(wrap_custom_tool_input(None)) == {"input": ""}
    assert wrap_custom_tool_input({"input": "x"}).startswith("{")
    assert json.loads(wrap_custom_tool_input("{nope"))["input"] == "{nope"

    search = _web_search_call_item("c1", "{nope")
    assert search["action"]["query"] == "{nope"
    assert _web_search_call_item("c2", {"q": "repo"})["action"]["query"] == "repo"

    assert _chat_parts_from_content(None) == []
    assert _chat_parts_from_content("") == []
    assert _chat_parts_from_content([{"type": "text", "text": "hi"}, "there"])[0]["text"] == "hi"
    assert _chat_parts_from_content({"type": "computer_call_output", "output": "out"})
    nested_parts = _sanitize_chat_content_parts(
        ["plain", 12, {"type": "text", "text": "ok"}, {"type": "image_url", "image_url": "https://x"}, {"type": "image_url", "image_url": {"url": "https://y"}}]
    )
    assert nested_parts[0]["text"] == "plain"

    assert _anthropic_content_to_chat_parts(None) == []
    assert _anthropic_content_to_chat_parts("") == []
    assert _anthropic_content_to_chat_parts(12)[0]["text"] == "12"
    parts = _anthropic_content_to_chat_parts(
        [
            "skip-str-as-nested",
            {"type": "text", "text": "hello"},
            {"type": "image", "source": {"type": "url", "url": "https://img"}},
            {"type": "image_url", "image_url": "https://img2"},
            {"content": [{"type": "text", "text": "nested"}]},
        ]
    )
    assert any(part.get("text") == "hello" for part in parts)
    assert _anthropic_image_block_to_chat_part({"source": "nope"}) is None
    assert _anthropic_image_block_to_chat_part({"source": {"type": "base64", "data": "AAA", "media_type": "image/png"}})
    assert _chat_image_part_to_anthropic({"image_url": {"url": ""}}) is None
    assert _chat_image_part_to_anthropic({"image_url": "https://x"})["source"]["type"] == "url"
    assert _chat_image_part_to_anthropic({"image_url": "data:image/png;base64,AAA"})["source"]["type"] == "base64"
    assert _chat_image_part_to_anthropic({"image_url": "data:not-base64"}) is None

    assert _anthropic_tool_choice_to_chat(None) is None
    assert _anthropic_tool_choice_to_chat("any") == "required"
    assert _anthropic_tool_choice_to_chat("shell")["function"]["name"] == "shell"
    assert _anthropic_tool_choice_to_chat({"type": "auto"}) == "auto"
    assert _anthropic_tool_choice_to_chat({"type": "any"}) == "required"
    assert _anthropic_tool_choice_to_chat({"type": "tool", "name": "x"})["function"]["name"] == "x"
    assert _anthropic_tool_choice_to_chat({"type": "other"}) == {"type": "other"}

    chat = responses_to_chat(
        {
            "model": "slug",
            "input": [
                "bare user",
                12,
                {"type": "input_text", "text": "typed"},
                {"type": "computer_call_output", "output": "screen"},
                {"type": "custom_tool_call", "call_id": "c1", "input": "patch"},
                {"type": "function_call_output", "call_id": "c1", "output": "ok"},
                {"role": "developer", "content": "dev"},
            ],
        },
        "real",
    )
    roles = [message["role"] for message in chat["messages"]]
    assert "user" in roles
    assert "tool" in roles or "assistant" in roles

    converted = anthropic_messages_to_chat(
        {
            "model": "claude",
            "system": "sys",
            "messages": [
                "skip",
                {"role": "user", "content": "hi"},
                {"role": "system", "content": "other"},
            ],
            "thinking": {"effort": "low"},
            "output_config": {"effort": "high"},
            "tools": [{"name": "lookup", "input_schema": {"type": "object"}}],
            "tool_choice": {"type": "tool", "name": "lookup"},
        },
        "real-model",
        max_tokens=9,
    )
    assert converted["model"] == "real-model"
    assert converted["reasoning_effort"] == "high"
    assert converted["tools"]


def test_discover_parser_helpers():
    assert _parse_chatgpt_codex_models_payload("nope") == []
    assert _parse_chatgpt_codex_models_payload({"models": "nope"}) == []
    rows = _parse_chatgpt_codex_models_payload(
        {
            "models": [
                "skip",
                {"slug": ""},
                {"slug": "gpt-5.5", "visibility": "hidden"},
                {"slug": "claude-3"},
                {"slug": "gpt-5.6-luna", "display_name": "Luna"},
                {"slug": "codex-auto-review"},
            ]
        }
    )
    slugs = {row["slug"] for row in rows}
    assert slugs == {"gpt-5.6-luna", "codex-auto-review"}

    assert _models_dev_model_cost_input({}) is None
    assert _models_dev_model_cost_input({"cost": {}}) is None
    assert _models_dev_model_cost_input({"cost": {"input": "nope"}}) is None
    assert _models_dev_model_cost_input({"cost": {"input": "0"}}) == 0.0

    assert _parse_models_dev_opencode_free_ids("nope") == []
    assert _parse_models_dev_opencode_free_ids({"opencode": "nope"}) == []
    from codex_shim.discover import MODELS_DEV_OPENCODE_PROVIDER

    payload = {
        MODELS_DEV_OPENCODE_PROVIDER: {
            "models": {
                "skip": "nope",
                "free": {"id": "free-model", "cost": {"input": 0}},
                "paid": {"id": "paid-model", "cost": {"input": 1.5}},
            }
        }
    }
    assert "free-model" in _parse_models_dev_opencode_free_ids(payload)
    assert "paid-model" in _parse_models_dev_opencode_paid_ids(payload)
    assert _parse_models_dev_opencode_paid_ids({"x": 1}) == []

    assert lookup_models_dev_row({}, providers=("openai",), model_id="") is None
    catalog = {"openai": {"models": {"gpt-x": {"id": "openai/gpt-x"}, "other": "skip"}}}
    assert lookup_models_dev_row(catalog, providers=("openai",), model_id="gpt-x")["id"] == "openai/gpt-x"
    assert lookup_models_dev_row(catalog, providers=("missing",), model_id="gpt-x") is None
    assert _positive_int("nope") is None
    assert _positive_int(0) is None
    assert _positive_int(8) == 8


def test_opencode_go_helpers(monkeypatch, tmp_path):
    assert _json_payload("{nope") == {}
    assert _json_payload("[1]") == {}
    missing = tmp_path / "missing.json"
    assert _read_settings_json(missing) == {}
    (tmp_path / "bad.json").write_text("{")
    with pytest.raises(RuntimeError, match="not valid JSON"):
        _read_settings_json(tmp_path / "bad.json")

    monkeypatch.setattr(
        "codex_shim.opencode_go._request_json",
        lambda *args, **kwargs: (500, {}),
    )
    with pytest.raises(RuntimeError, match="HTTP 500"):
        fetch_opencode_go_model_ids("https://go.example", "k")
    monkeypatch.setattr(
        "codex_shim.opencode_go._request_json",
        lambda *args, **kwargs: (200, {"data": "nope"}),
    )
    with pytest.raises(RuntimeError, match="model list"):
        fetch_opencode_go_model_ids("https://go.example", "k")
    monkeypatch.setattr(
        "codex_shim.opencode_go._request_json",
        lambda *args, **kwargs: (200, {"data": [{"id": "alpha"}, {"id": ""}, "skip"]}),
    )
    assert fetch_opencode_go_model_ids("https://go.example/", "k") == ["alpha"]
    assert probe_chat_model("https://go.example", "k", "alpha") == 200
    assert probe_messages_model("https://go.example", "k", "alpha") == 200

    assert opencode_go_model_row("m", chat_status=500, messages_status=500, api_key_env="E", base_url="u", prefer="chat") is None
    chat_row = opencode_go_model_row("m", chat_status=200, messages_status=500, api_key_env="E", base_url="u", prefer="messages")
    assert chat_row["provider"] == "generic-chat-completion-api"
    msg_row = opencode_go_model_row("m", chat_status=500, messages_status=200, api_key_env="E", base_url="u", prefer="chat")
    assert msg_row["provider"] == "anthropic"

    monkeypatch.delenv("OPENCODE_GO_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="Set "):
        refresh_opencode_go_settings(tmp_path / "models.json")

    monkeypatch.setenv("OPENCODE_API_KEY", "secret")
    monkeypatch.setattr("codex_shim.opencode_go.fetch_opencode_go_model_ids", lambda *a, **k: ["good", "bad"])
    monkeypatch.setattr("codex_shim.opencode_go.probe_chat_model", lambda *a, **k: 200 if a[2] == "good" else 500)
    monkeypatch.setattr("codex_shim.opencode_go.probe_messages_model", lambda *a, **k: 500)
    monkeypatch.setattr("codex_shim.opencode_go.write_opencode_go_models", lambda *a, **k: None)
    result = refresh_opencode_go_settings(tmp_path / "models.json", api_key_env="OPENCODE_API_KEY")
    assert [row["model"] for row in result.models] == ["good"]
    assert result.skipped[0][0] == "bad"


def test_opencode_go_request_json_urlerror(monkeypatch):
    from urllib.error import URLError as UE

    from codex_shim.opencode_go import _request_json

    monkeypatch.setattr(
        "codex_shim.opencode_go.request_urllib",
        lambda *args, **kwargs: (_ for _ in ()).throw(UE("down")),
    )
    with pytest.raises(RuntimeError, match="Could not reach OpenCode Go"):
        _request_json("GET", "https://go.example/models", {})


def test_cursor_passthrough_available_cache(monkeypatch):
    import codex_shim.cursor_passthrough as cp

    cp._auth_probe_cache = None
    probes = {"n": 0}

    def probe():
        probes["n"] += 1
        return True

    monkeypatch.setattr(cp, "_probe_cursor_auth", probe)
    assert cursor_passthrough_available() is True
    assert cursor_passthrough_available() is True
    assert probes["n"] == 1
    assert cursor_passthrough_available(force_refresh=True) is True
    assert probes["n"] == 2
    cp._auth_probe_cache = None


async def test_cursor_bridge_session_helpers():
    session = CursorBridgeSession.create(allowed_tools=frozenset({"shell"}), tool_types={}, tool_resolve={})
    session.append_post_terminal_text("")
    session.append_post_terminal_text(" leftover ")
    assert session.take_post_terminal_text() == "leftover"
    assert session.take_post_terminal_text() == ""
    assert await session.wait_passthrough_finished(timeout_s=0.1) is False
    session.mark_passthrough_finished()
    assert await session.wait_passthrough_finished(timeout_s=1) is True
    with pytest.raises(BridgeToolNotAllowedError):
        session._resolve_tool("", None)
    resolved = session._resolve_tool("shell", None)
    assert resolved[0] == "shell"

    assert _compact_parameters("nope")["properties"] == {}
    compact = _compact_parameters(
        {
            "properties": {
                "plain": "x",
                "typed": {"type": "number", "description": "n", "enum": list(range(20))},
                "empty": {},
            },
            "required": ["typed"],
        }
    )
    assert compact["properties"]["plain"]["type"] == "string"
    assert compact["required"] == ["typed"]
    assert _format_tool_catalog([], cap=2).startswith("(none")
    specs = [
        BridgeToolSpec(
            chat_name="ns.tool",
            emit_name="tool",
            namespace="ns",
            description="d",
            parameters={"type": "object", "properties": {str(i): {"type": "string"} for i in range(80)}},
        )
        for _ in range(3)
    ]
    catalog = _format_tool_catalog(specs, cap=1)
    assert "more bridged tools" in catalog
    assert _iter_bridge_tool_candidates("nope") == []
    assert _iter_bridge_tool_candidates(["skip", {"type": "function", "name": "shell"}])
    namespaced = _iter_bridge_tool_candidates(
        [
            {
                "type": "namespace",
                "name": "mcp__exa",
                "tools": [
                    {
                        "type": "function",
                        "name": "search",
                        "description": "find",
                        "parameters": {"type": "object"},
                    }
                ],
            }
        ]
    )
    assert namespaced
    assert tool_output_call_ids("nope") == set()
    assert tool_output_call_ids([{"type": "function_call_output", "call_id": "c1"}, "x"]) == {"c1"}


def test_server_main_and_cli_list_stop_restart_doctor(monkeypatch, tmp_path, capsys):
    launched = {}

    def fake_run_app(app, **kwargs):
        launched.update(kwargs)
        launched["app"] = app

    monkeypatch.setattr("codex_shim.server.web.run_app", fake_run_app)
    settings = tmp_path / "models.json"
    settings.write_text("{}")
    server_main(["--settings", str(settings), "--host", "127.0.0.1", "--port", "8767"])
    assert launched["port"] == 8767

    monkeypatch.setattr(cli, "chatgpt_passthrough_available", lambda *a, **k: False)
    monkeypatch.setattr(cli, "cursor_passthrough_available", lambda *a, **k: False)
    monkeypatch.setattr(cli, "_load_models", lambda path: [])
    monkeypatch.setattr(cli, "_active_router", lambda *a, **k: None)
    assert cli.list_models(settings) == 1
    models = [
        ShimModel(
            slug="local-llama",
            model="llama",
            display_name="Llama",
            provider="openai",
            base_url="http://x",
            api_key="k",
        ),
        ShimModel(
            slug="no-key",
            model="x",
            display_name="X",
            provider="openai",
            base_url="http://x",
            api_key="",
        ),
    ]
    monkeypatch.setattr(cli, "_load_models", lambda path: models)
    monkeypatch.setattr(cli, "chatgpt_passthrough_available", lambda *a, **k: True)
    monkeypatch.setattr(cli, "chatgpt_passthrough_display_names", lambda: {"codex-gpt-5-5": "GPT"})
    monkeypatch.setattr(cli, "cursor_passthrough_available", lambda *a, **k: True)
    monkeypatch.setattr(
        cli,
        "cursor_catalog_models",
        lambda: [SimpleNamespace(catalog_slug="cursor-auto", display_name="Auto", upstream_id="auto")],
    )
    monkeypatch.setattr(cli, "_active_router", lambda *a, **k: SimpleNamespace(slug="auto", display_name="Router"))
    assert cli.list_models(settings) == 0
    out = capsys.readouterr().out
    assert "local-llama" in out
    assert "missing API key" in out

    missing = tmp_path / "missing-settings.json"
    checks = cli._doctor_settings(missing)
    assert checks[0].status == "WARN"
    bad = tmp_path / "bad-settings.json"
    bad.write_text("{")
    monkeypatch.setattr(
        cli,
        "_load_models",
        lambda path: (_ for _ in ()).throw(SystemExit("broken")),
    )
    checks = cli._doctor_settings(bad)
    assert checks[0].status == "FAIL"

    monkeypatch.setattr(cli, "_load_models", lambda path: models)
    monkeypatch.setattr("codex_shim.cli.router_module.load_router_config", lambda path: None)
    ok_settings = tmp_path / "ok.json"
    ok_settings.write_text("{}")
    checks = cli._doctor_settings(ok_settings)
    assert any("auto router configured: false" in check.message for check in checks)

    monkeypatch.setattr(cli, "_terminate_pid", lambda pid: None)
    waits = {"n": 0}

    def wait_exit(pid, timeout):
        del pid, timeout
        waits["n"] += 1
        return waits["n"] >= 2

    monkeypatch.setattr(cli, "_wait_for_pid_exit", wait_exit)
    monkeypatch.setattr(cli, "_wait_for_port_free", lambda *a, **k: True)
    monkeypatch.setattr(cli.os, "name", "posix")
    monkeypatch.setattr(cli.os, "killpg", lambda *a, **k: (_ for _ in ()).throw(OSError("no pg")))
    monkeypatch.setattr(cli.os, "kill", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_SHUTDOWN_TERM_WAIT_S", 0)
    monkeypatch.setattr(cli, "_SHUTDOWN_KILL_WAIT_S", 0)
    monkeypatch.setattr(cli, "_PORT_FREE_WAIT_S", 0)
    assert cli._stop_pid(42, 8767) is True

    monkeypatch.setattr(cli.os, "name", "posix")
    monkeypatch.setattr(cli, "DEFAULT_PORT", 8765)
    monkeypatch.setattr(cli, "_listener_pid", lambda port: 9)
    monkeypatch.setattr(cli, "_stop_pid", lambda pid, port: False)
    assert cli.restart(settings, 8767) == 1

    monkeypatch.setattr(
        cli,
        "refresh_opencode_go_settings",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no key")),
    )
    assert cli.refresh_opencode_go(settings, "E", "http://x", "chat", 1.0) == 1
    monkeypatch.setattr(
        cli,
        "refresh_opencode_go_settings",
        lambda *a, **k: SimpleNamespace(
            models=[{"slug": "ocgo-a", "model": "a", "provider": "openai", "opencode_go_endpoint": "chat"}],
            skipped=[("b", 500, 404)],
            settings_path=settings,
        ),
    )
    monkeypatch.setattr(cli, "_refresh_published_catalog", lambda *a, **k: None)
    assert cli.refresh_opencode_go(settings, "E", "http://x", "chat", 1.0) == 0
    out = capsys.readouterr().out
    assert "Skipped 1" in out
    assert "ocgo-a" in out

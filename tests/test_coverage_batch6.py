from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from aiohttp import web
from aiohttp.test_utils import make_mocked_request

from codex_shim import cli
from codex_shim.continuation_policy import is_previous_response_id_upstream_error, is_previous_response_id_upstream_event
from codex_shim.cursor_bridge import (
    BridgeError,
    CursorBridgeSession,
    cursor_bridge_registry,
    parse_bridge_tool_from_shell,
)
from codex_shim.cursor_passthrough import _format_function_started, _format_glob_result, _format_shell_started, build_cursor_prompt
from codex_shim.mcp_search import parse_mcp_tool_reference, responses_tools_need_tool_search
from codex_shim.naming import _format_token, format_cursor_display_name
from codex_shim.router import _content_text, _latest_from_input, _latest_from_messages
from codex_shim.server import (
    AnthropicMessagesStreamState,
    ResponsesStreamState,
    ShimServer,
)
from codex_shim.settings import load_passthrough_error_fallback
from codex_shim.tool_translate import apply_function_call_ids, parse_tool_arguments, responses_function_call_ids
from codex_shim.translate import (
    _advertised_chat_tool_names,
    _function_call_to_custom_item,
    _tool_choice_targets_hosted_codex_tool,
    chat_completion_to_anthropic_message,
    chat_to_responses_request,
    coerce_custom_responses_tools,
    function_call_item_from_chat_tool,
    prepare_codex_byok_responses_body,
    prepare_openai_responses_tool_schemas,
    resolve_namespaced_tool_name,
    responses_to_anthropic,
    responses_to_chat,
    responses_tool_type_map,
    split_namespaced_tool_chat_name,
    tighten_responses_tool_schemas,
)


class _FakeStream:
    def __init__(self) -> None:
        self.chunks: list[bytes] = []

    async def write(self, data: bytes) -> None:
        self.chunks.append(data)


def _settings(tmp_path: Path) -> Path:
    path = tmp_path / "models.json"
    path.write_text("{}")
    return path


def test_translate_remaining_helpers():
    assert resolve_namespaced_tool_name("mapped", {"mapped": ("ns", "tool")}) == ("ns", "tool")
    assert split_namespaced_tool_chat_name("mcp__exa.search") == (None, "mcp__exa.search")
    parsed = parse_mcp_tool_reference("mcp__exa.search")
    assert parsed == ("exa", "search")
    assert split_namespaced_tool_chat_name("a.b") == ("a", "b")
    assert responses_tool_type_map("nope") == {}
    assert responses_tool_type_map(["skip", {"type": "namespace", "name": "", "tools": [{"type": "function", "name": "x"}]}]) == {}
    assert _advertised_chat_tool_names(["skip", {"function": {"name": "shell"}}, {"name": "shell"}]) == ["shell"]
    assert _tool_choice_targets_hosted_codex_tool("nope") is False
    assert _tool_choice_targets_hosted_codex_tool({"type": "image_generation"}) is True
    assert _tool_choice_targets_hosted_codex_tool({"name": "web_search"}) is True
    assert prepare_codex_byok_responses_body({"tools": [{"type": "web_search"}]}, {"user-agent": "curl"}) == {
        "tools": [{"type": "web_search"}]
    }
    tightened = tighten_responses_tool_schemas(
        ["skip", {"parameters": {"type": "object", "properties": {"a": {"type": "string"}}}, "function": {"parameters": {"type": "object"}}}]
    )
    assert tightened[0] == "skip"
    prepared = prepare_openai_responses_tool_schemas({"tools": "nope"})
    assert prepared["tools"] == "nope"
    coerced = coerce_custom_responses_tools(["skip", {"type": "custom", "name": "apply_patch"}, {"type": "namespace", "tools": [{"type": "custom", "name": "p"}]}])
    assert coerced
    custom = _function_call_to_custom_item({"arguments": "{}"}, in_progress=True)
    assert custom["input"] == ""
    custom2 = _function_call_to_custom_item({"name": "apply_patch", "arguments": '{"input":"x"}'}, in_progress=False)
    assert custom2["name"] == "apply_patch"
    item = function_call_item_from_chat_tool(
        {"id": "call_1", "function": {"name": "web_search", "arguments": '{"query":"q"}'}},
        tool_types={"web_search": "web_search"},
    )
    assert item["type"] == "web_search_call"
    patch = function_call_item_from_chat_tool(
        {"id": "call_2", "function": {"name": "apply_patch", "arguments": "diff"}},
        tool_types={"apply_patch": "apply_patch"},
    )
    assert patch["type"] == "custom_tool_call"
    chat = responses_to_chat(
        {
            "model": "x",
            "input": [
                {"type": "reasoning", "summary": [{"text": "think"}], "encrypted_content": "blob"},
                {"role": "user", "content": "hi"},
            ],
        },
        "up",
    )
    assert chat["messages"]
    dangling = responses_to_chat(
        {"model": "x", "input": [{"type": "reasoning", "summary": [{"text": "only"}]}]},
        "up",
    )
    assert any(m.get("role") == "assistant" for m in dangling["messages"])
    converted = chat_to_responses_request({"messages": [], "temperature": 0.2, "max_tokens": 3, "tools": []}, "up", max_tokens=9)
    assert converted["temperature"] == 0.2
    anthropic = responses_to_anthropic(
        {
            "model": "x",
            "input": [
                {"type": "reasoning", "summary": [{"text": "think"}]},
            ],
        },
        "up",
        32,
    )
    assert anthropic["messages"]
    mapped = chat_completion_to_anthropic_message(
        {
            "choices": [
                {
                    "message": {
                        "reasoning_content": "r",
                        "content": "hello",
                        "tool_calls": ["skip", {"id": "c1", "function": {"name": "shell", "arguments": "{nope"}}],
                    },
                    "finish_reason": "length",
                }
            ]
        },
        "model",
    )
    assert mapped["stop_reason"] == "max_tokens"
    empty = chat_completion_to_anthropic_message({"choices": [{"message": {}}]}, "model")
    assert empty["content"][0]["text"] == ""


def test_settings_router_naming_mcp_tools_continuation(tmp_path):
    missing = tmp_path / "none.json"
    assert load_passthrough_error_fallback(missing) == {}
    bad = tmp_path / "bad.json"
    bad.write_text("{")
    assert load_passthrough_error_fallback(bad) == {}
    not_obj = tmp_path / "list.json"
    not_obj.write_text("[]")
    assert load_passthrough_error_fallback(not_obj) == {}
    empty_raw = tmp_path / "empty.json"
    empty_raw.write_text(json.dumps({"passthrough_error_fallback": ""}))
    assert load_passthrough_error_fallback(empty_raw) == {}
    as_str = tmp_path / "str.json"
    as_str.write_text(json.dumps({"passthrough_error_fallback": "local-llama"}))
    mapped = load_passthrough_error_fallback(as_str)
    assert "gpt-5.4-mini" in mapped
    weird = tmp_path / "weird.json"
    weird.write_text(json.dumps({"passthrough_error_fallback": 12}))
    assert load_passthrough_error_fallback(weird) == {}

    assert _content_text(None) == ""
    assert _content_text("x") == "x"
    assert "a" in _content_text([{"text": "a"}, "b", 12])
    assert _content_text({"text": "c"}) == "c"
    assert _content_text(9) == "9"
    assert _latest_from_input(" hi ") == "hi"
    assert _latest_from_input(12) == ""
    assert _latest_from_input([12, {"role": "user", "content": "ask"}, {"type": "input_text", "text": "tail"}]) == "tail"
    assert _latest_from_messages("nope") == ""
    assert _latest_from_messages([{"role": "user", "content": "q"}]) == "q"

    assert _format_token("v2") == "V2"
    assert _format_token("ABC") == "ABC"
    assert format_cursor_display_name("") == "Cursor -"
    assert format_cursor_display_name("Cursor - Auto").startswith("Cursor")

    assert responses_tools_need_tool_search("nope") is False
    assert responses_tools_need_tool_search(["skip", {"type": "tool_search"}]) is True
    assert responses_tools_need_tool_search([{"function": {"name": "mcp__exa__search"}}]) is True
    assert parse_mcp_tool_reference("plain") is None

    assert responses_function_call_ids("fc_call_9")[0].startswith("fc_")
    assert apply_function_call_ids({"type": "message"})["type"] == "message"
    assert apply_function_call_ids({"type": "function_call", "id": "abc"})["id"].startswith("fc_")
    assert parse_tool_arguments("{")["_raw"] == "{"
    assert parse_tool_arguments("[1]")["_value"] == [1]
    assert parse_tool_arguments({"a": 1}) == {"a": 1}
    assert parse_tool_arguments(12) == {}

    assert is_previous_response_id_upstream_error("x", code="previous_response_not_found") is True
    assert is_previous_response_id_upstream_event({"type": "other"}) is False
    assert is_previous_response_id_upstream_event({"type": "error", "detail": "previous_response_not_found"}) is True
    assert is_previous_response_id_upstream_event({"type": "error", "error": "plain"}) is False

    wait = _format_shell_started({"args": {"command": 'curl http://127.0.0.1:8765/_cursor_bridge/v1/wait -d \'{"job_id":"j1"}\''}})
    assert "wait" in wait
    poll = _format_shell_started({"args": {"command": "curl http://127.0.0.1/_cursor_bridge/v1/poll"}})
    assert "poll" in poll
    invoke = _format_shell_started({"args": {"command": 'curl /_cursor_bridge/v1/invoke -d \'{"tool":"shell"}\''}})
    assert "shell" in invoke or "invoke" in invoke or "Codex" in invoke
    assert parse_bridge_tool_from_shell("echo hi") is None
    assert _format_function_started({"arguments": "abc"})
    assert _format_glob_result({"result": {"success": {"files": ["a.py"] * 25}}})
    assert _format_glob_result({"result": {"success": {"totalFiles": 3}}}) == "3 file(s) matched"
    prompt = build_cursor_prompt(
        {
            "model": "cursor-auto",
            "input": [
                {
                    "type": "function_call",
                    "call_id": "c1",
                    "name": "shell",
                    "arguments": "{}",
                },
                {"role": "assistant", "content": "working"},
            ],
        }
    )
    assert "[ASSISTANT]" in prompt or "shell" in prompt


def test_install_logrotate_windows_and_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(cli.os, "name", "nt")
    assert cli.install_logrotate() == 1
    monkeypatch.setattr(cli.os, "name", "posix")
    monkeypatch.setattr(cli.shutil, "which", lambda name: None)
    assert cli.install_logrotate() == 1
    monkeypatch.setattr(cli.shutil, "which", lambda name: "/usr/bin/logrotate")
    monkeypatch.setattr(cli, "LOGROTATE_CONF_PATH", tmp_path / "logrotate.conf")
    monkeypatch.setattr(cli, "LOGROTATE_STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(cli, "LOGROTATE_STATE_PATH", tmp_path / "state" / "status")
    monkeypatch.setattr(cli, "LOGROTATE_SERVICE_UNIT", tmp_path / "units" / "svc")
    monkeypatch.setattr(cli, "LOGROTATE_TIMER_UNIT", tmp_path / "units" / "timer")
    monkeypatch.setattr(cli, "SERVICE_LOG_PATH", tmp_path / "shim.log")
    monkeypatch.setattr(cli.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=1))
    assert cli.install_logrotate() == 1
    monkeypatch.setattr(
        cli.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=0, stdout="", stderr=""),
    )
    (tmp_path / "shim.log").write_bytes(b"x" * 10)
    assert cli.install_logrotate(force_rotate=True) == 0


async def test_chatgpt_fallback_cursor_unauth_and_ws_target(monkeypatch, tmp_path):
    missing = tmp_path / "missing-auth.json"
    monkeypatch.setattr("codex_shim.server.DEFAULT_CODEX_AUTH", missing)
    shim = ShimServer(_settings(tmp_path))
    request = make_mocked_request("POST", "/v1/responses")

    async def fallback(*a, **k):
        return web.Response(text="byok", status=200)

    monkeypatch.setattr(ShimServer, "_maybe_passthrough_byok_fallback", fallback)
    ok = await shim._chatgpt_passthrough(request, {"model": "codex-gpt-5-6-luna", "input": "hi"})
    assert ok.status == 200

    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"tokens": {}}))
    monkeypatch.setattr("codex_shim.server.DEFAULT_CODEX_AUTH", empty)
    ok2 = await shim._chatgpt_passthrough(request, {"model": "codex-gpt-5-6-luna", "input": "hi"})
    assert ok2.status == 200

    monkeypatch.setattr("codex_shim.server.cursor_passthrough_available", lambda: False)
    with pytest.raises(web.HTTPUnauthorized):
        await shim._cursor_passthrough(request, {"model": "cursor-auto", "input": "hi"})

    class Target:
        kind = "chatgpt"
        requested_slug = "codex-gpt-5-6-luna"
        upstream_model = "gpt-5.6"
        response_model_override = None
        access_token = "tok"
        account_id = "acct"
        route = None

    async def resolve(self, payload):
        del payload
        return Target()

    async def handle(self, request, passthrough, payload, target):
        del request, passthrough, payload, target
        return True

    monkeypatch.setattr(ShimServer, "_resolve_ws_passthrough_target", resolve)
    monkeypatch.setattr(ShimServer, "_handle_ws_passthrough_response_create", handle)
    monkeypatch.setattr("codex_shim.server.ws_passthrough_enabled", lambda: True)

    class AsyncSend:
        async def send_str(self, payload):
            del payload

    await shim._handle_response_create_websocket(request, AsyncSend(), {"model": "codex-gpt-5-6-luna"})


async def test_anthropic_fail_interrupt_and_bridge_poll_timeout():
    stream = _FakeStream()
    anthropic = AnthropicMessagesStreamState("claude")
    anthropic.reasoning_open = True
    anthropic.reasoning_index = 0
    anthropic.text_open = True
    anthropic.text_index = 1
    anthropic.tool_calls[0] = {
        "open": True,
        "closed": False,
        "name": "shell",
        "id": "c1",
        "index": 0,
        "block_index": 2,
        "arguments": "{}",
        "emitted": 2,
    }
    await anthropic.fail(stream, "boom", code="upstream_error")
    await anthropic.fail(stream, "again")

    state = ResponsesStreamState("cursor")
    state.reasoning_blocks[("cursor_tool", 0)] = {
        "closed": False,
        "text": "working",
        "id": "rsn_1",
        "output_index": 0,
        "item_id": "rsn_1",
    }
    state.reasoning_blocks[("other", 1)] = {"closed": False, "text": "x", "id": "rsn_x", "output_index": 1}
    state.reasoning_blocks[("cursor_tool", 2)] = {
        "closed": True,
        "text": "done",
        "id": "rsn_2",
        "output_index": 2,
    }
    await state.interrupt_cursor_tool_activities(stream, "\ninterrupted")

    session = CursorBridgeSession.create(allowed_tools=frozenset({"shell"}), tool_types={}, tool_resolve={})
    session.attach_collector(SimpleNamespace(append_function_call=lambda **kwargs: None))
    await cursor_bridge_registry.register(session)
    try:
        session.created_at = 0
        with pytest.raises(BridgeError, match="expired"):
            await session.invoke(tool="shell", arguments={"command": "echo"})
        session.created_at = time.time()
        session.ttl_s = 10**6
        with pytest.raises(BridgeError, match="JSON object"):
            await session.invoke(tool="shell", arguments="nope")
        accepted = await session.invoke(tool="shell", arguments={"command": "echo"})
        assert accepted["ok"] is True
        polled = await session.poll_jobs(timeout_s=0.05)
        assert polled["pending"] >= 1
    finally:
        cursor_bridge_registry.close(session.bridge_id)

from __future__ import annotations

import json

import pytest
from aiohttp.test_utils import TestClient, TestServer

from codex_shim.cursor_bridge import (
    BRIDGE_SUFFIX_TAG,
    BridgeToolNotAllowedError,
    CursorBridgeSession,
    bridge_allowed_tools,
    bridge_tool_specs,
    build_bridge_suffix,
    cursor_bridge_registry,
    is_bridge_denied_tool,
    is_bridge_shell_command,
    parse_bridge_tool_from_shell,
)
from codex_shim.cursor_passthrough import (
    CursorResponseCollector,
    build_cursor_prompt,
    format_cursor_tool_started_markdown,
    resolve_cursor_workspace,
)
from codex_shim.server import ResponsesStreamState, ShimServer
from tests.test_server import _sse_events


def _goal_tools_body() -> dict:
    return {
        "model": "cursor-composer-2-5",
        "input": [{"role": "user", "content": "Mark goal complete"}],
        "tools": [
            {
                "type": "namespace",
                "name": "goals",
                "tools": [
                    {
                        "type": "function",
                        "name": "update_goal",
                        "description": "Update goal status",
                        "parameters": {
                            "type": "object",
                            "properties": {"status": {"type": "string"}},
                            "required": ["status"],
                        },
                    },
                    {"type": "function", "name": "create_goal", "description": "Create a goal"},
                    {"type": "function", "name": "get_goal", "description": "Read goal state"},
                ],
            }
        ],
    }


def _mixed_codex_tools_body() -> dict:
    return {
        "model": "cursor-grok-4-5-high",
        "tools": [
            {"type": "function", "name": "exec_command", "description": "Run shell"},
            {"type": "function", "name": "write_stdin", "description": "Write stdin"},
            {"type": "apply_patch"},
            {"type": "function", "name": "list_mcp_resources", "description": "MCP resources"},
            {"type": "function", "name": "tool_search", "description": "Search deferred MCP"},
            {
                "type": "function",
                "name": "update_plan",
                "description": "Update the plan",
                "parameters": {"type": "object", "properties": {"plan": {"type": "string"}}},
            },
            {
                "type": "function",
                "name": "create_goal",
                "description": "Create a goal with objective",
                "parameters": {
                    "type": "object",
                    "properties": {"objective": {"type": "string", "description": "Goal text"}},
                    "required": ["objective"],
                },
            },
            {
                "type": "namespace",
                "name": "collaboration",
                "description": "Tools for spawning and managing sub-agents.",
                "tools": [
                    {
                        "type": "function",
                        "name": "spawn_agent",
                        "description": "Spawns an agent to work on the specified task.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "task_name": {"type": "string", "description": "Task name"},
                                "message": {"type": "string", "description": "Initial task"},
                            },
                            "required": ["task_name", "message"],
                        },
                    },
                    {
                        "type": "function",
                        "name": "wait_agent",
                        "description": "Wait for a mailbox update from any live agent.",
                        "parameters": {
                            "type": "object",
                            "properties": {"timeout_ms": {"type": "number"}},
                        },
                    },
                    {
                        "type": "function",
                        "name": "send_message",
                        "description": "Send a message to an existing agent.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "target": {"type": "string"},
                                "message": {"type": "string"},
                            },
                            "required": ["target", "message"],
                        },
                    },
                    {"type": "function", "name": "list_agents", "description": "List live agents"},
                    {"type": "function", "name": "followup_task", "description": "Follow up a child"},
                    {"type": "function", "name": "interrupt_agent", "description": "Interrupt a child"},
                ],
            },
            {
                "type": "namespace",
                "name": "mcp__codebase_memory_mcp",
                "tools": [{"type": "function", "name": "search_graph"}],
            },
        ],
    }


def test_bridge_allowed_tools_from_request():
    allowed = bridge_allowed_tools(_goal_tools_body())
    assert "goals_update_goal" in allowed
    assert "goals_create_goal" in allowed
    assert "goals_get_goal" in allowed


def test_bridge_denylist_excludes_cursor_like_tools():
    assert is_bridge_denied_tool(chat_name="exec_command")
    assert is_bridge_denied_tool(chat_name="apply_patch", tool_type="apply_patch")
    assert is_bridge_denied_tool(chat_name="list_mcp_resources")
    assert is_bridge_denied_tool(chat_name="mcp__exa__web_search_exa")
    assert is_bridge_denied_tool(chat_name="search_graph", namespace="mcp__codebase_memory_mcp")
    assert not is_bridge_denied_tool(chat_name="collaboration_spawn_agent", namespace="collaboration")
    assert not is_bridge_denied_tool(chat_name="create_goal")

    allowed = bridge_allowed_tools(_mixed_codex_tools_body())
    assert "collaboration_spawn_agent" in allowed
    assert "collaboration_wait_agent" in allowed
    assert "collaboration_send_message" in allowed
    assert "collaboration_list_agents" in allowed
    assert "collaboration_followup_task" in allowed
    assert "collaboration_interrupt_agent" in allowed
    assert "create_goal" in allowed
    assert "update_plan" in allowed
    assert "exec_command" not in allowed
    assert "write_stdin" not in allowed
    assert "apply_patch" not in allowed
    assert "list_mcp_resources" not in allowed
    assert "tool_search" not in allowed
    assert not any(name.startswith("mcp__") for name in allowed)


def test_bridge_tool_specs_include_descriptions():
    specs = bridge_tool_specs(_mixed_codex_tools_body())
    by_name = {spec.chat_name: spec for spec in specs}
    spawn = by_name["collaboration_spawn_agent"]
    assert spawn.namespace == "collaboration"
    assert "Spawns an agent" in spawn.description
    assert spawn.parameters.get("required") == ["task_name", "message"]
    assert "task_name" in (spawn.parameters.get("properties") or {})


def test_build_bridge_suffix_appends_after_stable_prompt():
    body = _goal_tools_body()
    prefix = build_cursor_prompt(body)
    specs = bridge_tool_specs(body)
    session = CursorBridgeSession.create(
        allowed_tools=bridge_allowed_tools(body),
        tool_types={},
        tool_resolve={},
        tool_specs=specs,
    )
    suffix = build_bridge_suffix(session, 8765, workspace="/tmp/ws")
    full = prefix + "\n\n" + suffix
    assert prefix in full
    assert full.startswith(prefix)
    assert BRIDGE_SUFFIX_TAG in suffix
    assert session.bridge_id in suffix
    assert "http://127.0.0.1:8765/_cursor_bridge/v1/invoke" in suffix
    assert "goals.update_goal" in suffix or "goals_update_goal" in suffix
    assert "Update goal status" in suffix
    assert "Parameters schema:" in suffix
    assert "Codex workspace path: /tmp/ws" in suffix
    assert "create_goal requires" in suffix
    assert "Sub-agent protocol" in suffix


def test_build_bridge_suffix_lists_collaboration_before_denylisted_noise():
    body = _mixed_codex_tools_body()
    specs = bridge_tool_specs(body)
    session = CursorBridgeSession.create(
        allowed_tools=frozenset(spec.chat_name for spec in specs),
        tool_types={},
        tool_resolve={},
        tool_specs=specs,
    )
    suffix = build_bridge_suffix(session, 8765)
    assert "collaboration.spawn_agent" in suffix
    assert "wait_agent" in suffix
    assert "send_message" in suffix
    assert "exec_command" not in suffix
    assert "list_mcp_resources" not in suffix
    assert suffix.index("collaboration.spawn_agent") < suffix.index("update_plan")


def test_resolve_cursor_workspace_from_prompt():
    prompt = "Work in workspace `/tmp/codex-bridge-smoke-abc123` only."
    assert resolve_cursor_workspace(prompt=prompt) == "/tmp/codex-bridge-smoke-abc123"


def test_resolve_cursor_workspace_from_metadata():
    body = {"metadata": {"cwd": "/tmp/from-metadata"}}
    assert resolve_cursor_workspace(body) == "/tmp/from-metadata"


def test_smoke_cursor_bridge_tmux_script_is_valid_bash():
    import subprocess
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "scripts" / "smoke_cursor_bridge_tmux.sh"
    assert script.is_file()
    assert script.stat().st_mode & 0o111
    subprocess.run(["bash", "-n", str(script)], check=True)


def test_bridge_shell_recognizer():
    curl = (
        "curl -sS -X POST 'http://127.0.0.1:8765/_cursor_bridge/v1/invoke' "
        "-H 'Content-Type: application/json' "
        '-d \'{"bridge":"abc123","tool":"update_goal","arguments":{"status":"complete"}}\''
    )
    assert is_bridge_shell_command(curl)
    assert parse_bridge_tool_from_shell(curl) == "update_goal"
    assert not is_bridge_shell_command("git status")


def test_bridge_shell_markdown_is_compact():
    curl = (
        "curl -sS -X POST 'http://127.0.0.1:8765/_cursor_bridge/v1/invoke' "
        '-d \'{"bridge":"x","tool":"goals_update_goal","arguments":{}}\''
    )
    md = format_cursor_tool_started_markdown(
        {"tool_call": {"shellToolCall": {"args": {"command": curl}}}}
    )
    assert "→ Codex tool:" in md
    assert "curl" not in md


@pytest.mark.asyncio
async def test_bridge_invoke_rejects_disallowed_tool():
    body = _goal_tools_body()
    session = CursorBridgeSession.create(
        allowed_tools=bridge_allowed_tools(body),
        tool_types={},
        tool_resolve={},
        tool_specs=bridge_tool_specs(body),
    )
    collector = CursorResponseCollector()
    session.attach_collector(collector)
    with pytest.raises(BridgeToolNotAllowedError):
        await session.invoke(tool="exec_command", arguments={"cmd": "ls"})


@pytest.mark.asyncio
async def test_bridge_invoke_rejects_denylisted_even_if_present_in_body():
    body = _mixed_codex_tools_body()
    from codex_shim.translate import responses_tool_resolve_map, responses_tool_type_map

    tool_types = responses_tool_type_map(body["tools"])
    tool_resolve = responses_tool_resolve_map(body["tools"])
    session = CursorBridgeSession.create(
        allowed_tools=bridge_allowed_tools(body),
        tool_types=tool_types,
        tool_resolve=tool_resolve,
        tool_specs=bridge_tool_specs(body),
    )
    collector = CursorResponseCollector()
    session.attach_collector(collector)
    with pytest.raises(BridgeToolNotAllowedError):
        await session.invoke(tool="exec_command", arguments={"cmd": "ls"})
    result = await session.invoke(
        tool="spawn_agent",
        namespace="collaboration",
        arguments={"task_name": "child_1", "message": "do the thing"},
    )
    assert result["ok"] is True
    assert result["namespace"] == "collaboration"
    assert result["tool"] == "spawn_agent"


@pytest.mark.asyncio
async def test_bridge_invoke_collector_appends_function_call():
    body = _goal_tools_body()
    from codex_shim.translate import responses_tool_resolve_map, responses_tool_type_map

    tool_types = responses_tool_type_map(body["tools"])
    tool_resolve = responses_tool_resolve_map(body["tools"])
    session = CursorBridgeSession.create(
        allowed_tools=bridge_allowed_tools(body),
        tool_types=tool_types,
        tool_resolve=tool_resolve,
        tool_specs=bridge_tool_specs(body),
    )
    collector = CursorResponseCollector(tool_types=tool_types, tool_resolve=tool_resolve)
    session.attach_collector(collector)
    result = await session.invoke(
        tool="update_goal",
        arguments={"status": "complete", "goal_id": "g1"},
        namespace="goals",
    )
    assert result["ok"] is True
    assert result["codex_call_id"].startswith(f"call_{session.bridge_id}_")
    tool_items = [item for item in collector.output if item.get("type") == "function_call"]
    assert len(tool_items) == 1
    assert tool_items[0]["namespace"] == "goals"
    assert tool_items[0]["name"] == "update_goal"
    assert json.loads(tool_items[0]["arguments"]) == {"status": "complete", "goal_id": "g1"}


@pytest.mark.asyncio
async def test_bridge_emit_synthetic_function_call_streams_sse():
    body = _goal_tools_body()
    from codex_shim.translate import responses_tool_resolve_map, responses_tool_type_map

    tool_types = responses_tool_type_map(body["tools"])
    tool_resolve = responses_tool_resolve_map(body["tools"])

    class FakeResponse:
        def __init__(self) -> None:
            self.chunks: list[bytes] = []

        async def write(self, data: bytes) -> None:
            self.chunks.append(data)

    downstream = FakeResponse()
    state = ResponsesStreamState(
        "cursor-composer-2-5",
        tool_types=tool_types,
        tool_resolve=tool_resolve,
    )
    await state.start(downstream)
    await state.emit_synthetic_function_call(
        downstream,
        name="update_goal",
        arguments={"status": "complete"},
        call_id="call_bridge_test_1",
        namespace="goals",
        chat_name="goals_update_goal",
    )
    await state.finish(downstream, upstream_saw_done=True)

    events = _sse_events(b"".join(downstream.chunks).decode())
    added = [
        event
        for event in events
        if event.get("type") == "response.output_item.added"
        and (event.get("item") or {}).get("type") == "function_call"
    ]
    assert len(added) == 1
    assert added[0]["item"]["namespace"] == "goals"
    assert added[0]["item"]["name"] == "update_goal"
    assert any(event.get("type") == "response.function_call_arguments.done" for event in events)


@pytest.mark.asyncio
async def test_cursor_bridge_invoke_http_handler():
    body = _goal_tools_body()
    from codex_shim.translate import responses_tool_resolve_map, responses_tool_type_map

    tool_types = responses_tool_type_map(body["tools"])
    tool_resolve = responses_tool_resolve_map(body["tools"])
    session = CursorBridgeSession.create(
        allowed_tools=bridge_allowed_tools(body),
        tool_types=tool_types,
        tool_resolve=tool_resolve,
        tool_specs=bridge_tool_specs(body),
    )
    collector = CursorResponseCollector(tool_types=tool_types, tool_resolve=tool_resolve)
    session.attach_collector(collector)
    await cursor_bridge_registry.register(session)

    shim = ShimServer()
    client = TestClient(TestServer(shim.app()))
    await client.start_server()

    try:
        async with client.post(
            "/_cursor_bridge/v1/invoke",
            json={
                "bridge": session.bridge_id,
                "tool": "update_goal",
                "namespace": "goals",
                "arguments": {"status": "complete"},
            },
            headers={"Host": "127.0.0.1"},
        ) as resp:
            assert resp.status == 200
            payload = await resp.json()
            assert payload["ok"] is True
            assert payload["namespace"] == "goals"

        bad = await client.post(
            "/_cursor_bridge/v1/invoke",
            json={"bridge": session.bridge_id, "tool": "nope", "arguments": {}},
            headers={"Host": "127.0.0.1"},
        )
        assert bad.status == 400

        missing = await client.post(
            "/_cursor_bridge/v1/invoke",
            json={"bridge": "missing", "tool": "update_goal", "arguments": {}},
            headers={"Host": "127.0.0.1"},
        )
        assert missing.status == 404
    finally:
        cursor_bridge_registry.close(session.bridge_id)
        await client.close()

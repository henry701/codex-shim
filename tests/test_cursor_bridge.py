from __future__ import annotations

import asyncio
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


def test_smoke_bridge_adoption_script_is_valid_bash():
    import subprocess
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "scripts" / "smoke_bridge_adoption.sh"
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


@pytest.mark.asyncio
async def test_bridge_invoke_returns_job_id_and_wait_consumes_output():
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
    try:
        accepted = await session.invoke(
            tool="update_goal",
            arguments={"status": "complete"},
            namespace="goals",
        )
        assert accepted["ok"] is True
        assert accepted["status"] == "accepted"
        job_id = accepted["job_id"]
        call_id = accepted["codex_call_id"]
        assert job_id
        assert session.has_pending_jobs()

        n = cursor_bridge_registry.ingest_function_call_outputs(
            [
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": '{"status":"complete","goal_id":"g1"}',
                }
            ]
        )
        assert n == 1

        result = await session.wait_job(job_id, timeout_s=1.0)
        assert result["ok"] is True
        assert result["job_id"] == job_id
        assert result["status"] == "consumed"
        assert "complete" in str(result["output"])
        assert not session.has_pending_jobs()

        again = await session.wait_job(job_id, timeout_s=0.1)
        assert again["ok"] is False
        assert again["error"] == "unknown_job"
    finally:
        cursor_bridge_registry.close(session.bridge_id)


@pytest.mark.asyncio
async def test_bridge_poll_returns_ready_jobs_and_clears_them():
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
    try:
        a = await session.invoke(tool="create_goal", arguments={"objective": "x"}, namespace="goals")
        b = await session.invoke(tool="get_goal", arguments={}, namespace="goals")
        cursor_bridge_registry.ingest_function_call_outputs(
            [
                {"type": "function_call_output", "call_id": a["codex_call_id"], "output": "goal-created"},
                {"type": "function_call_output", "call_id": b["codex_call_id"], "output": "goal-state"},
            ]
        )
        polled = await session.poll_jobs(timeout_s=0.0)
        assert polled["ok"] is True
        assert len(polled["jobs"]) == 2
        ids = {job["job_id"] for job in polled["jobs"]}
        assert ids == {a["job_id"], b["job_id"]}
        assert not session.has_pending_jobs()

        empty = await session.poll_jobs(timeout_s=0.0)
        assert empty["jobs"] == []
    finally:
        cursor_bridge_registry.close(session.bridge_id)


@pytest.mark.asyncio
async def test_bridge_wait_http_handler_blocks_until_ingest():
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
            accepted = await resp.json()
        job_id = accepted["job_id"]
        call_id = accepted["codex_call_id"]

        async def _ingest_later() -> None:
            await asyncio.sleep(0.05)
            cursor_bridge_registry.ingest_function_call_outputs(
                [{"type": "function_call_output", "call_id": call_id, "output": "done-via-http"}]
            )

        ingest_task = asyncio.create_task(_ingest_later())
        async with client.post(
            "/_cursor_bridge/v1/wait",
            json={"bridge": session.bridge_id, "job_id": job_id, "timeout_ms": 2000},
            headers={"Host": "127.0.0.1"},
        ) as resp:
            assert resp.status == 200
            payload = await resp.json()
            assert payload["ok"] is True
            assert payload["output"] == "done-via-http"
        await ingest_task

        async with client.post(
            "/_cursor_bridge/v1/poll",
            json={"bridge": session.bridge_id, "timeout_ms": 0},
            headers={"Host": "127.0.0.1"},
        ) as resp:
            polled = await resp.json()
            assert polled["jobs"] == []
    finally:
        cursor_bridge_registry.close(session.bridge_id)
        await client.close()


@pytest.mark.asyncio
async def test_bridge_tool_output_followup_never_uses_delivery_stub():
    from pathlib import Path

    from codex_shim.cursor_bridge import (
        BRIDGE_DELIVERY_STUB_MARKER,
        decide_tool_output_followup,
        input_items_are_only_tool_outputs,
    )

    only_outputs = [
        {"type": "function_call_output", "call_id": "c1", "output": "goal-state"},
        {"type": "function_call_output", "call_id": "c2", "output": "agents"},
    ]
    desktop_shaped = [
        {"type": "reasoning", "summary": [{"type": "summary_text", "text": "…"}]},
        {"type": "function_call_output", "call_id": "c1", "output": "x"},
    ]
    with_user = [
        {"type": "function_call_output", "call_id": "c1", "output": "x"},
        {"role": "user", "content": "hi"},
    ]

    assert input_items_are_only_tool_outputs(only_outputs)
    assert input_items_are_only_tool_outputs(desktop_shaped)
    assert not input_items_are_only_tool_outputs(with_user)

    # Empty leftover must continue Cursor — never a stub completion.
    assert (
        decide_tool_output_followup(
            ingested=3,
            delivery_path=True,
            input_items=only_outputs,
            leftover="",
        )
        == "continue_cursor"
    )
    assert (
        decide_tool_output_followup(
            ingested=3,
            delivery_path=True,
            input_items=desktop_shaped,
            leftover="   ",
        )
        == "continue_cursor"
    )
    assert (
        decide_tool_output_followup(
            ingested=2,
            delivery_path=True,
            input_items=only_outputs,
            leftover="Real leftover from in-flight turn",
        )
        == "reuse_leftover"
    )
    assert (
        decide_tool_output_followup(
            ingested=2,
            delivery_path=True,
            input_items=with_user,
            leftover="",
        )
        == "noop"
    )
    assert (
        decide_tool_output_followup(
            ingested=2,
            delivery_path=False,
            input_items=only_outputs,
            leftover="",
        )
        == "noop"
    )
    # Guard: never reintroduce the Desktop-ending stub prose as an emitted string.
    root = Path(__file__).resolve().parents[1] / "codex_shim"
    banned = "Bridge delivered Codex tool results"
    for rel in ("server.py", "cursor_bridge.py", "cursor_passthrough.py"):
        assert banned not in (root / rel).read_text()


@pytest.mark.asyncio
async def test_bridge_ingest_is_idempotent_for_already_ready_jobs():
    """Desktop replays function_call_output in expanded history; ingest must not re-count."""
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
    try:
        accepted = await session.invoke(
            tool="update_goal",
            arguments={"status": "complete"},
            namespace="goals",
        )
        call_id = accepted["codex_call_id"]
        items = [
            {"type": "function_call_output", "call_id": call_id, "output": "first"},
        ]
        assert cursor_bridge_registry.ingest_function_call_outputs(items) == 1
        assert cursor_bridge_registry.ingest_function_call_outputs(items) == 0
        assert cursor_bridge_registry.ingest_function_call_outputs(items) == 0
        result = await session.wait_job(accepted["job_id"], timeout_s=1.0)
        assert result["ok"] is True
        assert result["output"] == "first"
    finally:
        cursor_bridge_registry.close(session.bridge_id)


@pytest.mark.asyncio
async def test_bridge_delivery_response_rejects_empty_message():
    shim = ShimServer()
    with pytest.raises(Exception) as excinfo:
        await shim._cursor_bridge_delivery_response(
            request=None,  # type: ignore[arg-type]
            raw_body={"stream": False},
            slug="cursor-composer-2-5",
            message="",
        )
    # aiohttp HTTPInternalServerError or similar
    assert "non-empty" in str(excinfo.value).lower() or getattr(excinfo.value, "status", None) == 500


@pytest.mark.asyncio
async def test_bridge_delivery_response_rejects_stub_marker_message():
    """Regression: synthetic 'Bridge delivered…' must never be emitted as assistant text."""
    from codex_shim.cursor_bridge import BRIDGE_DELIVERY_STUB_MARKER

    shim = ShimServer()
    stub = f"{BRIDGE_DELIVERY_STUB_MARKER} Codex tool results to the in-flight Cursor turn."
    with pytest.raises(Exception) as excinfo:
        await shim._cursor_bridge_delivery_response(
            request=None,  # type: ignore[arg-type]
            raw_body={"stream": False},
            slug="cursor-composer-2-5",
            message=stub,
        )
    assert "stub" in str(excinfo.value).lower() or getattr(excinfo.value, "status", None) == 500


@pytest.mark.asyncio
async def test_bridge_poll_reports_idle_when_session_has_no_jobs():
    """Live logs showed agents re-polling `jobs=0 pending=0` sessions in a loop."""
    body = _goal_tools_body()
    session = CursorBridgeSession.create(
        allowed_tools=bridge_allowed_tools(body),
        tool_types={},
        tool_resolve={},
        tool_specs=bridge_tool_specs(body),
    )
    session.attach_collector(CursorResponseCollector(tool_types={}, tool_resolve={}))
    await cursor_bridge_registry.register(session)
    try:
        idle = await session.poll_jobs(timeout_s=0.0)
        assert idle["ok"] is True
        assert idle["jobs"] == []
        assert idle["pending"] == 0
        assert idle["idle"] is True
        assert "stop polling" in idle["hint"].lower()

        accepted = await session.invoke(tool="get_goal", arguments={}, namespace="goals")
        busy = await session.poll_jobs(timeout_s=0.0)
        assert busy["pending"] == 1
        assert "idle" not in busy

        cursor_bridge_registry.ingest_function_call_outputs(
            [
                {
                    "type": "function_call_output",
                    "call_id": accepted["codex_call_id"],
                    "output": "goal-state",
                }
            ]
        )
        drained = await session.poll_jobs(timeout_s=0.0)
        assert len(drained["jobs"]) == 1
        assert "idle" not in drained
    finally:
        cursor_bridge_registry.close(session.bridge_id)


@pytest.mark.asyncio
async def test_registry_prunes_expired_sessions_with_unconsumed_jobs():
    """Turns that invoke 3 tools but wait on 1 leave `ready` jobs; sessions must still expire."""
    body = _goal_tools_body()
    session = CursorBridgeSession.create(
        allowed_tools=bridge_allowed_tools(body),
        tool_types={},
        tool_resolve={},
        tool_specs=bridge_tool_specs(body),
    )
    session.attach_collector(CursorResponseCollector(tool_types={}, tool_resolve={}))
    await cursor_bridge_registry.register(session)
    stale_id = session.bridge_id
    try:
        accepted = await session.invoke(tool="get_goal", arguments={}, namespace="goals")
        cursor_bridge_registry.ingest_function_call_outputs(
            [
                {
                    "type": "function_call_output",
                    "call_id": accepted["codex_call_id"],
                    "output": "never-consumed",
                }
            ]
        )
        # Ready-but-unconsumed keeps release_if_idle from reclaiming the session.
        cursor_bridge_registry.release_if_idle(stale_id)
        assert cursor_bridge_registry.get(stale_id) is not None

        session.created_at -= session.ttl_s + 1
        assert cursor_bridge_registry.prune_expired() == 1
        assert cursor_bridge_registry.get(stale_id) is None
    finally:
        cursor_bridge_registry.close(stale_id)


@pytest.mark.asyncio
async def test_bridge_routes_return_actionable_unknown_bridge_error():
    """A stale bridge id (shim restart / ended turn) must not read as a hard failure.

    A bare 404 caused reconnect storms and goals being marked blocked.
    """
    shim = ShimServer()
    client = TestClient(TestServer(shim.app()))
    await client.start_server()
    try:
        cases = [
            ("invoke", {"bridge": "gone", "tool": "get_goal", "arguments": {}}),
            ("wait", {"bridge": "gone", "job_id": "j1", "timeout_ms": 0}),
            ("poll", {"bridge": "gone", "timeout_ms": 0}),
        ]
        for action, body in cases:
            async with client.post(
                f"/_cursor_bridge/v1/{action}",
                json=body,
                headers={"Host": "127.0.0.1"},
            ) as resp:
                assert resp.status == 404
                payload = await resp.json()
            assert payload["ok"] is False
            assert payload["error"] == "unknown_bridge"
            assert payload["action"] == action
            assert payload["retryable"] is False
            hint = payload["hint"].lower()
            assert "do not retry" in hint
            assert "blocked" in hint
    finally:
        await client.close()


def test_bridge_suffix_documents_unknown_bridge_recovery():
    body = _goal_tools_body()
    session = CursorBridgeSession.create(
        allowed_tools=bridge_allowed_tools(body),
        tool_types={},
        tool_resolve={},
        tool_specs=bridge_tool_specs(body),
    )
    suffix = build_bridge_suffix(session, 8765)
    assert "unknown_bridge" in suffix
    assert "do not mark the goal blocked" in suffix


def test_build_bridge_suffix_documents_wait_and_poll():
    body = _goal_tools_body()
    session = CursorBridgeSession.create(
        allowed_tools=bridge_allowed_tools(body),
        tool_types={},
        tool_resolve={},
        tool_specs=bridge_tool_specs(body),
    )
    suffix = build_bridge_suffix(session, 8765)
    assert "/_cursor_bridge/v1/wait" in suffix
    assert "/_cursor_bridge/v1/poll" in suffix
    assert "job_id" in suffix


@pytest.mark.asyncio
async def test_registry_remembers_tool_name_after_job_is_consumed():
    body = _goal_tools_body()
    session = CursorBridgeSession.create(
        allowed_tools=bridge_allowed_tools(body),
        tool_types={},
        tool_resolve={},
        tool_specs=bridge_tool_specs(body),
    )
    session.attach_collector(CursorResponseCollector(tool_types={}, tool_resolve={}))
    await cursor_bridge_registry.register(session)
    try:
        accepted = await session.invoke(
            tool="create_goal", arguments={"objective": "x"}, namespace="goals"
        )
        call_id = accepted["codex_call_id"]
        cursor_bridge_registry.ingest_function_call_outputs(
            [{"type": "function_call_output", "call_id": call_id, "output": "made"}]
        )
        await session.wait_job(accepted["job_id"], timeout_s=1.0)
        # Job is consumed, but the name must survive for orphan repair.
        assert "create_goal" in str(cursor_bridge_registry.tool_name_for_call(call_id))
        assert cursor_bridge_registry.tool_name_for_call("call_missing") is None
    finally:
        cursor_bridge_registry.close(session.bridge_id)


class _FakeBridgeAgent:
    """Scripted cursor-agent that drives the real bridge protocol.

    Mirrors what the agent's curl calls do: emit text, invoke a Codex tool, block on
    the result, then keep talking. Records how many times it was spawned so tests can
    prove a follow-up turn continued this process instead of starting a new one.
    """

    def __init__(self) -> None:
        self.spawns = 0
        self.prompts: list[str] = []

    def __call__(self, prompt, model, *, workspace=None):
        self.spawns += 1
        self.prompts.append(prompt)
        return self._run(prompt)

    async def _run(self, prompt: str):
        yield {"type": "text_delta", "delta": "Inventorying the fork."}
        bridge_id = ""
        for line in prompt.splitlines():
            if '"bridge"' in line and ":" in line:
                bridge_id = line.split('"bridge"')[1].split('"')[1]
                break
        session = cursor_bridge_registry.get(bridge_id)
        assert session is not None, f"agent could not find bridge {bridge_id!r}"
        accepted = await session.invoke(
            tool="create_goal", arguments={"objective": "merge fork"}, namespace="goals"
        )
        result = await session.wait_job(accepted["job_id"], timeout_s=5.0)
        yield {"type": "text_delta", "delta": f"Codex said: {result.get('output')}."}
        yield {"type": "completed", "text": "done"}


async def _post_cursor_turn(client, payload: dict) -> str:
    resp = await client.post(
        "/v1/responses",
        json=payload,
        headers={"session-id": "adopt-session", "Host": "127.0.0.1"},
    )
    assert resp.status == 200
    return await resp.text()


@pytest.mark.asyncio
async def test_tool_output_turn_adopts_inflight_agent_instead_of_respawning(monkeypatch, tmp_path):
    """One cursor-agent must span the whole goal, across Codex tool round-trips.

    Respawning per turn rebuilds the prompt from scratch, which is how the agent lost
    its own reasoning and restarted its plan every turn.
    """
    import codex_shim.server as server_module

    agent = _FakeBridgeAgent()
    monkeypatch.setattr(server_module, "cursor_passthrough_available", lambda: True)
    monkeypatch.setattr(server_module, "iter_cursor_agent_events", agent)
    monkeypatch.setattr(server_module, "cursor_upstream_model", lambda slug: "composer-2.5")

    settings = tmp_path / "settings.json"
    settings.write_text("{}")
    client = TestClient(TestServer(ShimServer(settings).app()))
    await client.start_server()
    try:
        body = _goal_tools_body()
        body["stream"] = True
        first = await _post_cursor_turn(client, body)

        calls = [
            event
            for event in _sse_events(first)
            if event.get("type") == "response.output_item.added"
            and (event.get("item") or {}).get("type") == "function_call"
        ]
        assert len(calls) == 1, "first turn should hand Codex the tool call"
        call_id = calls[0]["item"]["call_id"]
        assert agent.spawns == 1

        follow_up = {
            "model": "cursor-composer-2-5",
            "stream": True,
            "tools": body["tools"],
            "input": [
                {"type": "function_call_output", "call_id": call_id, "output": "goal-created"}
            ],
        }
        second = await _post_cursor_turn(client, follow_up)

        assert agent.spawns == 1, "follow-up must continue the live agent, not spawn another"
        assert "Codex said: goal-created" in second
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_adopted_turn_falls_back_to_a_fresh_agent_when_none_is_live(monkeypatch, tmp_path):
    """With no agent to adopt, the follow-up must still produce a real turn."""
    import codex_shim.server as server_module

    async def fake_events(prompt, model, *, workspace=None):
        yield {"type": "text_delta", "delta": "fresh agent answer"}
        yield {"type": "completed", "text": "fresh agent answer"}

    monkeypatch.setattr(server_module, "cursor_passthrough_available", lambda: True)
    monkeypatch.setattr(server_module, "iter_cursor_agent_events", fake_events)
    monkeypatch.setattr(server_module, "cursor_upstream_model", lambda slug: "composer-2.5")

    settings = tmp_path / "settings.json"
    settings.write_text("{}")
    client = TestClient(TestServer(ShimServer(settings).app()))
    await client.start_server()
    try:
        text = await _post_cursor_turn(
            client,
            {
                "model": "cursor-composer-2-5",
                "stream": True,
                "input": [
                    {
                        "type": "function_call_output",
                        "call_id": "call_never_seen",
                        "output": "orphan",
                    }
                ],
            },
        )
        assert "fresh agent answer" in text
    finally:
        await client.close()


class _TwoRoundBridgeAgent(_FakeBridgeAgent):
    """Agent that invokes a second Codex tool after its first result arrives."""

    async def _run(self, prompt: str):
        bridge_id = ""
        for line in prompt.splitlines():
            if '"bridge"' in line and ":" in line:
                bridge_id = line.split('"bridge"')[1].split('"')[1]
                break
        session = cursor_bridge_registry.get(bridge_id)
        assert session is not None
        first = await session.invoke(
            tool="create_goal", arguments={"objective": "merge fork"}, namespace="goals"
        )
        created = await session.wait_job(first["job_id"], timeout_s=5.0)
        yield {"type": "text_delta", "delta": f"Created: {created.get('output')}."}
        second = await session.invoke(tool="get_goal", arguments={}, namespace="goals")
        state = await session.wait_job(second["job_id"], timeout_s=5.0)
        yield {"type": "text_delta", "delta": f"State: {state.get('output')}."}
        yield {"type": "completed", "text": "done"}


@pytest.mark.asyncio
async def test_adopted_agent_can_invoke_more_tools_on_later_turns(monkeypatch, tmp_path):
    """Early-complete closes the turn, not the session: the same agent keeps calling tools."""
    import codex_shim.server as server_module

    agent = _TwoRoundBridgeAgent()
    monkeypatch.setattr(server_module, "cursor_passthrough_available", lambda: True)
    monkeypatch.setattr(server_module, "iter_cursor_agent_events", agent)
    monkeypatch.setattr(server_module, "cursor_upstream_model", lambda slug: "composer-2.5")

    settings = tmp_path / "settings.json"
    settings.write_text("{}")
    client = TestClient(TestServer(ShimServer(settings).app()))
    await client.start_server()
    try:
        tools = _goal_tools_body()["tools"]

        def _call_ids(sse: str) -> list[str]:
            return [
                event["item"]["call_id"]
                for event in _sse_events(sse)
                if event.get("type") == "response.output_item.added"
                and (event.get("item") or {}).get("type") == "function_call"
            ]

        first = await _post_cursor_turn(
            client,
            {
                "model": "cursor-composer-2-5",
                "stream": True,
                "tools": tools,
                "input": [{"role": "user", "content": "merge the fork"}],
            },
        )
        call_one = _call_ids(first)
        assert len(call_one) == 1

        second = await _post_cursor_turn(
            client,
            {
                "model": "cursor-composer-2-5",
                "stream": True,
                "tools": tools,
                "input": [
                    {"type": "function_call_output", "call_id": call_one[0], "output": "goal-1"}
                ],
            },
        )
        assert "Created: goal-1" in second
        call_two = _call_ids(second)
        assert len(call_two) == 1, "adopted turn must be able to emit a new Codex tool call"

        third = await _post_cursor_turn(
            client,
            {
                "model": "cursor-composer-2-5",
                "stream": True,
                "tools": tools,
                "input": [
                    {"type": "function_call_output", "call_id": call_two[0], "output": "goal-state"}
                ],
            },
        )
        assert "State: goal-state" in third
        assert agent.spawns == 1, "three Codex turns must share one cursor-agent"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_cursor_thinking_reopens_after_tool_boundary(monkeypatch):
    """Each think→tool→think cycle must be a separate reasoning item for Desktop."""
    import codex_shim.server as server_module

    writes: list[dict] = []

    async def fake_write(_response, payload):
        writes.append(payload)

    monkeypatch.setattr(server_module, "_write_sse", fake_write)

    state = ResponsesStreamState("cursor-composer-2-5")
    response = object()
    await state.append_cursor_thinking_activity(response, "plan A")
    await state.close_cursor_thinking_activity(response)
    await state.open_cursor_tool_activity(response, "t1", "**cursor-agent · shell**\n\n> ls\n")
    await state.close_cursor_tool_activity(response, "t1")
    await state.append_cursor_thinking_activity(response, "plan B")
    await state.close_cursor_thinking_activity(response)

    reasoning_added = [
        event
        for event in writes
        if event.get("type") == "response.output_item.added"
        and (event.get("item") or {}).get("type") == "reasoning"
    ]
    assert len(reasoning_added) == 3  # think, tool, think
    texts = [
        state.reasoning_blocks[key]["text"]
        for key in sorted(
            state.reasoning_blocks,
            key=lambda k: state.reasoning_blocks[k]["output_index"],
        )
    ]
    assert any("plan A" in text for text in texts)
    assert any("plan B" in text for text in texts)
    assert any("shell" in text for text in texts)


@pytest.mark.asyncio
async def test_steer_follow_up_cancels_orphaned_agent_instead_of_adopting(monkeypatch, tmp_path):
    """Codex steer keeps history and sends new user text + interrupted tool output.

    The previous cursor-agent is still blocked on wait; adopting it would ignore the
    steer. Cancel it and let this turn respawn from the cached prompt.
    """
    import codex_shim.server as server_module

    agent = _FakeBridgeAgent()
    monkeypatch.setattr(server_module, "cursor_passthrough_available", lambda: True)
    monkeypatch.setattr(server_module, "iter_cursor_agent_events", agent)
    monkeypatch.setattr(server_module, "cursor_upstream_model", lambda slug: "composer-2.5")

    settings = tmp_path / "settings.json"
    settings.write_text("{}")
    client = TestClient(TestServer(ShimServer(settings).app()))
    await client.start_server()
    try:
        tools = _goal_tools_body()["tools"]
        first = await _post_cursor_turn(
            client,
            {
                "model": "cursor-composer-2-5",
                "stream": True,
                "tools": tools,
                "input": [{"role": "user", "content": "start goal"}],
            },
        )
        call_ids = [
            event["item"]["call_id"]
            for event in _sse_events(first)
            if event.get("type") == "response.output_item.added"
            and (event.get("item") or {}).get("type") == "function_call"
        ]
        assert len(call_ids) == 1
        assert agent.spawns == 1

        # Steer-shaped follow-up: interrupted tool output + new user text.
        second = await _post_cursor_turn(
            client,
            {
                "model": "cursor-composer-2-5",
                "stream": True,
                "tools": tools,
                "input": [
                    {
                        "type": "function_call_output",
                        "call_id": call_ids[0],
                        "output": '{"message":"Wait interrupted by new input.","timed_out":false}',
                    },
                    {"role": "user", "content": "stop waiting and continue"},
                ],
            },
        )
        assert agent.spawns == 2, "steer must spawn a fresh agent from history, not adopt"
        assert "Inventorying" in second or "Codex said" in second or len(second) > 0
    finally:
        await client.close()


def _attach_sleeping_agent(session: CursorBridgeSession) -> asyncio.Task[None]:
    async def sleeper() -> None:
        await asyncio.sleep(100)

    task = asyncio.create_task(sleeper())
    session._agent_task = task
    return task


@pytest.mark.asyncio
async def test_cancel_agent_on_disconnect_kills_when_turn_closed_without_handoff():
    """User cancel after early-complete used to skip kill because turn_closed was set."""
    session = CursorBridgeSession.create(
        allowed_tools=frozenset({"create_goal"}),
        tool_types={},
        tool_resolve={},
    )
    task = _attach_sleeping_agent(session)
    session.mark_turn_closed()

    session.cancel_agent_on_disconnect("user cancel")

    await asyncio.sleep(0)
    assert task.cancelled()


@pytest.mark.asyncio
async def test_cancel_agent_on_disconnect_preserves_agent_after_handoff_disconnect():
    session = CursorBridgeSession.create(
        allowed_tools=frozenset({"create_goal"}),
        tool_types={},
        tool_resolve={},
    )
    task = _attach_sleeping_agent(session)
    session.mark_turn_closed()
    session.mark_handoff_disconnect_expected()

    session.cancel_agent_on_disconnect("expected handoff hang-up")

    await asyncio.sleep(0)
    assert not task.cancelled()
    task.cancel()


@pytest.mark.asyncio
async def test_cancel_live_agents_for_session_kills_waiting_orphan():
    session = CursorBridgeSession.create(
        allowed_tools=frozenset({"create_goal"}),
        tool_types={},
        tool_resolve={},
    )
    session.session_key = "codex-session-1"
    task = _attach_sleeping_agent(session)
    await cursor_bridge_registry.register(session, session_key="codex-session-1")

    cancelled = cursor_bridge_registry.cancel_live_agents_for_session("codex-session-1")

    assert cancelled == 1
    await asyncio.sleep(0)
    assert task.cancelled()
    cursor_bridge_registry.close(session.bridge_id)


@pytest.mark.asyncio
async def test_fresh_turn_after_handoff_kills_orphan_agent(monkeypatch, tmp_path):
    """Cancel-without-tool-output must not leave cursor-agent blocked on bridge wait."""
    import codex_shim.server as server_module

    agent = _FakeBridgeAgent()
    monkeypatch.setattr(server_module, "cursor_passthrough_available", lambda: True)
    monkeypatch.setattr(server_module, "iter_cursor_agent_events", agent)
    monkeypatch.setattr(server_module, "cursor_upstream_model", lambda slug: "composer-2.5")

    settings = tmp_path / "settings.json"
    settings.write_text("{}")
    client = TestClient(TestServer(ShimServer(settings).app()))
    await client.start_server()
    try:
        tools = _goal_tools_body()["tools"]
        first = await _post_cursor_turn(
            client,
            {
                "model": "cursor-composer-2-5",
                "stream": True,
                "tools": tools,
                "input": [{"role": "user", "content": "start goal"}],
            },
        )
        call_ids = [
            event["item"]["call_id"]
            for event in _sse_events(first)
            if event.get("type") == "response.output_item.added"
            and (event.get("item") or {}).get("type") == "function_call"
        ]
        assert len(call_ids) == 1
        assert agent.spawns == 1

        await _post_cursor_turn(
            client,
            {
                "model": "cursor-composer-2-5",
                "stream": True,
                "tools": tools,
                "input": [{"role": "user", "content": "never mind, do something else"}],
            },
        )
        assert agent.spawns == 2, "fresh turn must cancel the waiting agent and respawn"
    finally:
        await client.close()


def test_replayed_history_tail_is_a_tool_output_delivery_not_a_steer():
    """After compaction Codex drops previous_response_id and replays history inline."""
    from codex_shim.cursor_bridge import input_items_deliver_tool_outputs

    post_compaction_replay = [
        {"type": "message", "role": "user", "content": "start goal"},
        {"type": "compaction", "summary": "…"},
        {"type": "message", "role": "assistant", "content": "working"},
        {"type": "reasoning", "summary": [{"type": "summary_text", "text": "…"}]},
        {"type": "function_call_output", "call_id": "c1", "output": "agents"},
        {"type": "function_call_output", "call_id": "c2", "output": "goal-state"},
    ]
    steer = [
        {"type": "function_call_output", "call_id": "c1", "output": "interrupted"},
        {"role": "user", "content": "stop waiting and continue"},
    ]

    assert input_items_deliver_tool_outputs(post_compaction_replay)
    assert not input_items_deliver_tool_outputs(steer)
    assert not input_items_deliver_tool_outputs([])


@pytest.mark.asyncio
async def test_post_compaction_history_replay_adopts_instead_of_respawning(monkeypatch, tmp_path):
    """Compaction stops Codex from sending previous_response_id, so the follow-up carries
    the whole conversation inline. Treating that as a steer cancelled the agent that owned
    the results and respawned it, which made the agent re-announce its plan every turn.
    """
    import codex_shim.server as server_module

    agent = _FakeBridgeAgent()
    monkeypatch.setattr(server_module, "cursor_passthrough_available", lambda: True)
    monkeypatch.setattr(server_module, "iter_cursor_agent_events", agent)
    monkeypatch.setattr(server_module, "cursor_upstream_model", lambda slug: "composer-2.5")

    settings = tmp_path / "settings.json"
    settings.write_text("{}")
    client = TestClient(TestServer(ShimServer(settings).app()))
    await client.start_server()
    try:
        body = _goal_tools_body()
        body["stream"] = True
        first = await _post_cursor_turn(client, body)

        calls = [
            event
            for event in _sse_events(first)
            if event.get("type") == "response.output_item.added"
            and (event.get("item") or {}).get("type") == "function_call"
        ]
        assert len(calls) == 1
        assert agent.spawns == 1

        follow_up = _goal_tools_body()
        follow_up["stream"] = True
        # No previous_response_id: post-compaction Codex replays history inline.
        follow_up["input"] = [
            {"type": "message", "role": "user", "content": "Mark goal complete"},
            {"type": "compaction", "summary": "earlier work summarized"},
            {"type": "reasoning", "summary": [{"type": "summary_text", "text": "…"}]},
            {
                "type": "function_call_output",
                "call_id": calls[0]["item"]["call_id"],
                "output": "goal-created",
            },
        ]
        second = await _post_cursor_turn(client, follow_up)

        assert agent.spawns == 1, "replayed history must adopt the live agent, not respawn"
        assert "Codex said: goal-created" in second
    finally:
        await client.close()

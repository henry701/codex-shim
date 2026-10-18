from __future__ import annotations

from codex_shim.responses_input_pipeline import (
    STALE_TOOL_OUTPUT_MESSAGE,
    UNKNOWN_FUNCTION_TOOL_NAME,
    align_responses_tool_call_ids,
    expand_cached_responses_input,
    prepare_responses_input_items,
    synthesize_missing_tool_outputs,
    synthesize_orphan_tool_calls,
)


class _FakeCache:
    def __init__(self, items: list[dict] | None, *, latest_items: list[dict] | None = None) -> None:
        self._items = items
        self._latest_items = latest_items

    def get(self, session_key: str, previous_response_id: str) -> list | None:
        return self._items

    def latest(self, session_key: str) -> list | None:
        return self._latest_items


def test_prepare_responses_input_items_skips_orphan_synthesis_when_disabled():
    from codex_shim.responses_input_pipeline import prepare_responses_input_items

    tool_output = {"type": "function_call_output", "call_id": "call_1", "output": "ok"}
    repaired, warnings = prepare_responses_input_items(
        cache=None,
        session_key="s",
        previous_response_id="resp_1",
        input_items=[tool_output],
        expand_enabled=False,
        context="turn",
        orphan_synthesis=False,
    )
    assert repaired == [tool_output]
    assert warnings == []


def test_synthesize_orphan_tool_calls_inserts_placeholder_before_output():
    input_items = [
        {"type": "function_call_output", "call_id": "call_orphan", "output": "x"},
        {"type": "function_call", "call_id": "call_ok", "name": "shell", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "call_ok", "output": "done"},
    ]
    repaired, warnings = synthesize_orphan_tool_calls(input_items)
    assert repaired[0]["type"] == "function_call"
    assert repaired[0]["call_id"] == "call_orphan"
    assert repaired[0]["name"] == UNKNOWN_FUNCTION_TOOL_NAME
    assert repaired[1]["call_id"] == "call_orphan"
    assert len(warnings) == 1
    assert "synthesized" in warnings[0]


def test_expand_cached_responses_input_falls_back_to_latest_session_snapshot():
    cached = [
        {"type": "message", "role": "user", "content": "real task"},
        {"type": "function_call", "call_id": "call_1", "name": "exec_command", "arguments": "{}"},
    ]
    delta = [
        {"type": "message", "role": "user", "content": "Continue"},
        {"type": "function_call_output", "call_id": "call_1", "output": "grep hit"},
    ]
    cache = _FakeCache(None, latest_items=cached)
    expanded = expand_cached_responses_input(
        cache=cache,
        session_key="sess",
        previous_response_id="resp_missing",
        delta_input=delta,
        expand_enabled=True,
        context="turn",
    )
    assert expanded[0]["content"] == "real task"
    assert expanded[1]["name"] == "exec_command"
    assert expanded[-1]["call_id"] == "call_1"
    assert expanded[-2]["content"] == "Continue"


def test_expand_cached_responses_input_prepends_cache_for_any_route():
    cached = [{"type": "message", "role": "user", "content": "prior"}]
    delta = [{"type": "function_call_output", "call_id": "call_x", "output": "tail"}]
    cache = _FakeCache(cached)
    expanded = expand_cached_responses_input(
        cache=cache,
        session_key="sess",
        previous_response_id="resp_prev",
        delta_input=delta,
        expand_enabled=True,
        context="turn",
    )
    assert len(expanded) == 2
    assert expanded[0]["content"] == "prior"


def test_prepare_responses_input_items_expands_then_synthesizes():
    cached = [
        {"type": "function_call", "call_id": "call_1", "name": "shell", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "call_1", "output": "ok"},
    ]
    delta = [{"type": "function_call_output", "call_id": "call_2", "output": "tail"}]
    cache = _FakeCache(cached)
    repaired, warnings = prepare_responses_input_items(
        cache=cache,
        session_key="sess",
        previous_response_id="resp_prev",
        input_items=delta,
        expand_enabled=True,
        context="compaction",
    )
    assert len(repaired) == 4
    assert repaired[2]["type"] == "function_call"
    assert repaired[2]["call_id"] == "call_2"
    assert repaired[3]["call_id"] == "call_2"
    assert any("synthesized" in warning for warning in warnings)
    assert any("replayed item id" in warning for warning in warnings)


def test_synthesize_orphan_tool_calls_uses_resolved_bridge_tool_name():
    """Bridge results arrive detached from their function_call; keep the real name."""
    input_items = [
        {"type": "function_call_output", "call_id": "call_bridge_1", "output": "goal"},
        {"type": "function_call_output", "call_id": "call_other", "output": "x"},
    ]
    names = {"call_bridge_1": "create_goal"}
    repaired, warnings = synthesize_orphan_tool_calls(
        input_items, name_resolver=names.get
    )
    assert repaired[0]["name"] == "create_goal"
    assert repaired[2]["name"] == UNKNOWN_FUNCTION_TOOL_NAME
    assert len(warnings) == 2


def test_synthesize_missing_tool_outputs_stubs_unanswered_function_call():
    input_items = [
        {
            "type": "function_call",
            "call_id": "call_stale",
            "name": "exec_command",
            "arguments": '{"cmd":"ps"}',
        },
        {"type": "message", "role": "user", "content": "continue the fork update"},
    ]
    repaired, warnings = synthesize_missing_tool_outputs(input_items)
    assert repaired[0]["type"] == "function_call"
    assert repaired[1]["type"] == "function_call_output"
    assert repaired[1]["call_id"] == "call_stale"
    assert repaired[1]["output"] == STALE_TOOL_OUTPUT_MESSAGE
    assert repaired[2]["role"] == "user"
    assert len(warnings) == 1
    assert "call_stale" in warnings[0]


def test_synthesize_missing_tool_outputs_leaves_paired_call_alone():
    input_items = [
        {"type": "function_call", "call_id": "call_ok", "name": "exec_command", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "call_ok", "output": "done"},
    ]
    repaired, warnings = synthesize_missing_tool_outputs(input_items)
    assert repaired == input_items
    assert warnings == []


def test_prepare_responses_input_items_stubs_unanswered_call_when_synthesis_enabled():
    repaired, warnings = prepare_responses_input_items(
        cache=None,
        session_key="s",
        previous_response_id=None,
        input_items=[
            {
                "type": "function_call",
                "call_id": "call_2b12cbfa-56f5-46a1-8ff9-6f1215aee5a2",
                "name": "exec_command",
                "arguments": "{}",
            }
        ],
        expand_enabled=False,
        context="turn",
    )
    assert repaired[1]["type"] == "function_call_output"
    assert repaired[1]["output"] == STALE_TOOL_OUTPUT_MESSAGE
    assert warnings


def test_align_responses_tool_call_ids_prefixes_bare_uuid_pair():
    uuid = "2b12cbfa-56f5-46a1-8ff9-6f1215aee5a2"
    aligned = align_responses_tool_call_ids(
        [
            {
                "type": "function_call",
                "call_id": uuid,
                "name": "exec_command",
                "arguments": "{}",
            },
            {"type": "function_call_output", "call_id": uuid, "output": "ps hit"},
        ]
    )
    assert aligned[0]["call_id"] == f"call_{uuid}"
    assert aligned[1]["call_id"] == f"call_{uuid}"
    assert aligned[1]["output"] == "ps hit"


def test_prepare_keeps_real_output_when_cache_uuid_and_call_prefix_differ():
    """GLM cache + ChatGPT rewrite of the call must still pair; do not stub or compact."""
    uuid = "2b12cbfa-56f5-46a1-8ff9-6f1215aee5a2"
    cache = _FakeCache(
        [
            {
                "type": "function_call",
                "call_id": f"call_{uuid}",
                "name": "exec_command",
                "arguments": "{}",
            },
            {"type": "function_call_output", "call_id": uuid, "output": "telegram-desktop"},
        ]
    )
    repaired, warnings = prepare_responses_input_items(
        cache=cache,
        session_key="sess",
        previous_response_id="resp_glm",
        input_items=[{"type": "message", "role": "user", "content": "continue on GPT"}],
        expand_enabled=True,
        context="turn",
    )
    outputs = [item for item in repaired if item.get("type") == "function_call_output"]
    assert len(outputs) == 1
    assert outputs[0]["call_id"] == f"call_{uuid}"
    assert outputs[0]["output"] == "telegram-desktop"
    assert warnings == ["cleared 1 replayed item id(s); rustponsesapi cannot look them up"]


def test_prepare_clears_server_minted_ids_that_require_persisted_lookup():
    server_msg = "msg_" + "ab" * 22
    server_rs = "rs_" + "cd" * 22
    server_fc = "fc_" + "ef" * 22
    server_ctc = "ctc_" + "12" * 22
    client_msg = "msg_01a13da5-4ffd-7830-9d9b-00f5efa06dd6"
    short_fc = "fc_7bf23739ba1d481fa26d0883350cc220"
    call_id = "call_900801da886f4c028988e9a0c8d9915a"
    cache = _FakeCache(
        [
            {"type": "message", "id": server_msg, "role": "assistant", "content": "kept"},
            {"type": "message", "id": client_msg, "role": "user", "content": "client"},
            {
                "type": "reasoning",
                "id": server_rs,
                "encrypted_content": "ciphertext",
                "summary": [],
            },
            {
                "type": "function_call",
                "id": server_fc,
                "call_id": call_id,
                "name": "exec",
                "arguments": "{}",
            },
            {
                "type": "function_call",
                "id": short_fc,
                "call_id": "call_7bf23739ba1d481fa26d0883350cc220",
                "name": "wait",
                "arguments": "{}",
            },
            {
                "type": "custom_tool_call",
                "id": server_ctc,
                "call_id": call_id,
                "name": "exec",
                "input": "{}",
            },
        ]
    )
    repaired, warnings = prepare_responses_input_items(
        cache=cache,
        session_key="sess",
        previous_response_id="resp_missing",
        input_items=[{"type": "message", "role": "user", "content": "answer"}],
        expand_enabled=True,
        context="turn",
    )
    by_type: dict[str, list] = {}
    for item in repaired:
        by_type.setdefault(item["type"], []).append(item)
    assert by_type["message"][0]["content"] == "kept"
    assert by_type["message"][1]["content"] == "client"
    assert all("id" not in item for item in by_type["message"])
    assert "id" not in by_type["reasoning"][0]
    assert by_type["reasoning"][0]["encrypted_content"] == "ciphertext"
    assert by_type["function_call"][0]["call_id"] == call_id
    assert by_type["function_call"][1]["call_id"] == "call_7bf23739ba1d481fa26d0883350cc220"
    for call in by_type["function_call"]:
        assert call["id"].startswith("fc_")
        assert "-" in call["id"]
        assert call["id"] not in {server_fc, short_fc}
    assert by_type["custom_tool_call"][0]["call_id"] == call_id
    assert by_type["custom_tool_call"][0]["id"].startswith("ctc_")
    assert "-" in by_type["custom_tool_call"][0]["id"]
    assert any("replayed item id" in warning for warning in warnings)


def test_native_delta_keeps_client_ids_and_drops_hex_ids():
    client_msg = "msg_01a13da5-4ffd-7830-9d9b-00f5efa06dd6"
    short_fc = "fc_7bf23739ba1d481fa26d0883350cc220"
    repaired, warnings = prepare_responses_input_items(
        cache=None,
        session_key="sess",
        previous_response_id="resp_live",
        input_items=[
            {"type": "message", "id": client_msg, "role": "user", "content": "hi"},
            {
                "type": "function_call",
                "id": short_fc,
                "call_id": "call_7bf23739ba1d481fa26d0883350cc220",
                "name": "wait",
                "arguments": "{}",
            },
        ],
        expand_enabled=False,
        context="turn",
        orphan_synthesis=False,
    )
    assert repaired[0]["id"] == client_msg
    assert repaired[1]["call_id"] == "call_7bf23739ba1d481fa26d0883350cc220"
    assert repaired[1]["id"].startswith("fc_")
    assert "-" in repaired[1]["id"]
    assert any("persisted item id" in warning for warning in warnings)

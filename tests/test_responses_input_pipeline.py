from __future__ import annotations

from codex_shim.responses_input_pipeline import (
    UNKNOWN_FUNCTION_TOOL_NAME,
    expand_cached_responses_input,
    prepare_responses_input_items,
    synthesize_orphan_tool_calls,
)


class _FakeCache:
    def __init__(self, items: list[dict] | None) -> None:
        self._items = items

    def get(self, session_key: str, previous_response_id: str) -> list | None:
        return self._items


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
    assert len(warnings) == 1


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

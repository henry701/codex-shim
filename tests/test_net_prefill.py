from __future__ import annotations

from types import SimpleNamespace

from codex_shim.net.prefill import (
    MAX_PREFILL_CONTINUES,
    ReplaySkipper,
    assistant_prefill_message,
    should_prefill_continue,
    skip_replayed_prefix,
    with_assistant_prefill,
)


def test_should_prefill_continue_on_silent_eof_without_finish_reason():
    assert should_prefill_continue(finish_reason=None, saw_done=False, continues=0) is True


def test_should_not_prefill_after_stop_or_tool_calls_even_without_done():
    assert should_prefill_continue(finish_reason="stop", saw_done=False, continues=0) is False
    assert should_prefill_continue(finish_reason="tool_calls", saw_done=False, continues=0) is False


def test_should_not_prefill_after_done_or_length_or_cap():
    assert should_prefill_continue(finish_reason=None, saw_done=True, continues=0) is False
    assert should_prefill_continue(finish_reason="length", saw_done=False, continues=0) is False
    assert (
        should_prefill_continue(
            finish_reason=None,
            saw_done=False,
            continues=MAX_PREFILL_CONTINUES,
        )
        is False
    )


def test_assistant_prefill_message_is_trailing_assistant_without_user_nudge():
    state = SimpleNamespace(
        message_text="The plan is",
        reasoning_blocks={"chat": {"text": "think"}},
        tool_calls={},
        mcp_tool_calls={},
        tool_search_calls={},
    )
    prefill = assistant_prefill_message(state)
    assert prefill is not None
    assert prefill["role"] == "assistant"
    assert prefill["content"] == "The plan is"
    assert prefill["reasoning_content"] == "think"
    assert "tool_calls" not in prefill


def test_assistant_prefill_includes_partial_tool_calls_as_valid_request_json():
    state = SimpleNamespace(
        message_text="",
        reasoning_blocks={},
        tool_calls={
            0: {
                "call_id": "call_1",
                "name": "exec_command",
                "arguments": '{"cmd": "ls',
            }
        },
        mcp_tool_calls={},
        tool_search_calls={},
    )
    prefill = assistant_prefill_message(state)
    assert prefill is not None
    assert prefill["role"] == "assistant"
    assert prefill["content"] == ""
    assert prefill["tool_calls"] == [
        {
            "id": "call_1",
            "type": "function",
            "index": 0,
            "function": {"name": "exec_command", "arguments": '{"cmd": "ls'},
        }
    ]


def test_assistant_prefill_message_none_when_empty():
    state = SimpleNamespace(
        message_text="",
        reasoning_blocks={},
        tool_calls={},
        mcp_tool_calls={},
        tool_search_calls={},
    )
    assert assistant_prefill_message(state) is None


def test_with_assistant_prefill_appends_assistant_and_never_a_user_nudge():
    messages = [{"role": "user", "content": "hi"}]
    prefill = {"role": "assistant", "content": "The plan is"}
    out = with_assistant_prefill(messages, prefill)
    assert [m["role"] for m in out] == ["user", "assistant"]
    assert out[-1] is prefill
    assert messages == [{"role": "user", "content": "hi"}]


def test_with_assistant_prefill_replaces_existing_trailing_assistant():
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "old"},
    ]
    prefill = {"role": "assistant", "content": "The plan is"}
    out = with_assistant_prefill(messages, prefill)
    assert len(out) == 2
    assert out[-1]["content"] == "The plan is"


def test_skip_replayed_prefix_strips_regenerated_assistant_text():
    emitted, skipped = skip_replayed_prefix("The plan is", "The plan is to ship", 0)
    assert emitted == " to ship"
    assert skipped == len("The plan is")


def test_skip_replayed_prefix_keeps_true_continuation():
    emitted, skipped = skip_replayed_prefix("The plan is", " to ship", 0)
    assert emitted == " to ship"
    assert skipped == 0


def test_replay_skipper_keeps_usage_only_chunk():
    skipper = ReplaySkipper(text_prefix="hello")
    event = {
        "choices": [{"delta": {}}],
        "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
    }
    assert skipper.filter_chunk(event) == event

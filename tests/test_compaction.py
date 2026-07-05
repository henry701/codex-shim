from __future__ import annotations

import pytest

from codex_shim.compaction import (
    CompactionTriggerError,
    apply_compaction_fallback_notice,
    compact_response_payload,
    compaction_output_item,
    compaction_summary_from_output,
    decode_shim_compaction_summary,
    drop_orphaned_tool_outputs,
    encode_shim_compaction_summary,
    is_orphan_tool_call_upstream_error,
    sanitize_compaction_input_items,
    strip_terminal_compaction_trigger,
)


def test_strip_terminal_compaction_trigger_removes_final_trigger():
    input_items = [
        {"role": "user", "content": "hello"},
        {"type": "compaction_trigger"},
    ]
    assert strip_terminal_compaction_trigger(input_items) == [
        {"role": "user", "content": "hello"},
    ]


def test_strip_terminal_compaction_trigger_rejects_non_terminal_trigger():
    input_items = [
        {"type": "compaction_trigger"},
        {"role": "user", "content": "hello"},
    ]
    with pytest.raises(CompactionTriggerError):
        strip_terminal_compaction_trigger(input_items)


def test_compaction_round_trip_summary_encoding():
    summary = "Task: finish compaction support for DeepSeek."
    encrypted = encode_shim_compaction_summary(summary)
    assert decode_shim_compaction_summary(encrypted) == summary


def test_apply_compaction_fallback_notice_prepends_agent_context():
    text = apply_compaction_fallback_notice("Keep going.", "Bad Request")
    assert "Remote native compaction failed (Bad Request)" in text
    assert "fallback summarization" in text
    assert text.endswith("Keep going.")


def test_apply_compaction_fallback_notice_includes_warnings_for_model():
    text = apply_compaction_fallback_notice(
        "Keep going.",
        "no compactable input remains after removing orphaned tool outputs",
        warnings=["dropped orphan function_call_output (call_id='call_x')"],
    )
    assert "Compaction warnings:" in text
    assert "dropped orphan function_call_output" in text
    assert "Keep going." in text


def test_drop_orphaned_tool_outputs_removes_unmatched_outputs():
    input_items = [
        {
            "type": "function_call_output",
            "call_id": "call_orphan",
            "output": "truncated",
        },
        {
            "type": "function_call",
            "name": "shell",
            "call_id": "call_ok",
            "arguments": "{}",
        },
        {
            "type": "function_call_output",
            "call_id": "call_ok",
            "output": "done",
        },
    ]
    cleaned, warnings = drop_orphaned_tool_outputs(input_items)
    assert len(cleaned) == 2
    assert cleaned[0]["type"] == "function_call"
    assert cleaned[1]["call_id"] == "call_ok"
    assert len(warnings) == 1
    assert "call_orphan" in warnings[0]


def test_drop_orphaned_tool_outputs_keeps_matched_custom_tool_round_trip():
    input_items = [
        {
            "type": "custom_tool_call",
            "call_id": "call_patch",
            "name": "apply_patch",
            "input": "{}",
        },
        {
            "type": "custom_tool_call_output",
            "call_id": "call_patch",
            "output": "patched",
        },
    ]
    cleaned, warnings = drop_orphaned_tool_outputs(input_items)
    assert cleaned == input_items
    assert warnings == []


def test_drop_orphaned_tool_outputs_orphan_only_matches_truncated_compaction_input():
    """Low-level orphan drop still removes detached outputs in isolation."""
    input_items = [
        {
            "type": "function_call_output",
            "call_id": "call_VF4XLxSPXoTfOfVgV0jDqbXq",
            "output": "truncated exec output",
        },
    ]
    cleaned, warnings = drop_orphaned_tool_outputs(input_items)
    assert cleaned == []
    assert len(warnings) == 1
    assert "function_call_output" in warnings[0]
    assert "call_VF4XLxSPXoTfOfVgV0jDqbXq" in warnings[0]


def test_sanitize_compaction_input_items_preserves_detached_orphan_only_batch():
    input_items = [
        {
            "type": "function_call_output",
            "call_id": "call_VF4XLxSPXoTfOfVgV0jDqbXq",
            "output": "truncated exec output",
        },
    ]
    cleaned, warnings, audit = sanitize_compaction_input_items(input_items)
    assert len(cleaned) == 1
    assert cleaned[0]["call_id"] == "call_VF4XLxSPXoTfOfVgV0jDqbXq"
    assert len(audit.preserved) == 1
    assert any("preserved" in warning for warning in warnings)


def test_sanitize_compaction_input_items_preserves_tail_outputs_after_expanded_history():
    input_items = [
        {"type": "message", "role": "user", "content": "run"},
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "shell",
            "arguments": "{}",
        },
        {"type": "function_call_output", "call_id": "call_1", "output": "ok"},
        {"type": "function_call_output", "call_id": "call_2", "output": "tail"},
    ]
    cleaned, warnings, audit = sanitize_compaction_input_items(input_items)
    assert [item.get("call_id") for item in cleaned if item.get("type") == "function_call_output"] == [
        "call_1",
        "call_2",
    ]
    assert len(audit.preserved) == 1
    assert audit.preserved[0][0].call_id == "call_2"
    assert any("preserved" in warning for warning in warnings)


def test_drop_orphaned_tool_outputs_drops_custom_tool_output_without_call():
    input_items = [
        {
            "type": "custom_tool_call_output",
            "call_id": "call_missing",
            "output": "orphan patch result",
        },
    ]
    cleaned, warnings = drop_orphaned_tool_outputs(input_items)
    assert cleaned == []
    assert len(warnings) == 1
    assert "custom_tool_call_output" in warnings[0]


def test_drop_orphaned_tool_outputs_drops_output_with_missing_call_id():
    input_items = [
        {
            "type": "function_call_output",
            "output": "no call_id field",
        },
    ]
    cleaned, warnings = drop_orphaned_tool_outputs(input_items)
    assert cleaned == []
    assert len(warnings) == 1
    assert "missing call_id" in warnings[0]


def test_is_orphan_tool_call_upstream_error_detects_chatgpt_message():
    message = "No tool call found for function call output with call_id call_VF4XLxSPXoTfOfVgV0jDqbXq."
    assert is_orphan_tool_call_upstream_error(message)
    assert not is_orphan_tool_call_upstream_error("Bad Request")


def test_compact_response_payload_emits_compaction_item():
    payload = compact_response_payload("deepseek-v4-pro", "Keep implementing compaction.")
    assert payload["status"] == "completed"
    assert len(payload["output"]) == 1
    item = payload["output"][0]
    assert item["type"] == "compaction"
    assert isinstance(item["encrypted_content"], str)
    assert decode_shim_compaction_summary(item["encrypted_content"]) == "Keep implementing compaction."


def test_compaction_summary_from_output_reads_message_and_compaction_items():
    item = compaction_output_item("Stored summary")
    output = [
        {
            "type": "message",
            "content": [{"type": "output_text", "text": "Visible summary"}],
        },
        item,
    ]
    assert compaction_summary_from_output(output) == "Visible summary\nStored summary"


def test_responses_to_chat_maps_compaction_item_to_user_context():
    from codex_shim.translate import responses_to_chat

    item = compaction_output_item("Task: keep going with DeepSeek compaction.")
    body = {
        "model": "slug",
        "input": [
            {"role": "user", "content": "latest ask"},
            item,
        ],
    }
    out = responses_to_chat(body, "upstream")
    assert out["messages"][-1]["role"] == "user"
    assert "Compacted conversation state:" in out["messages"][-1]["content"]
    assert "Task: keep going with DeepSeek compaction." in out["messages"][-1]["content"]

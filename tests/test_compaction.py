from __future__ import annotations

import pytest

from codex_shim.compaction import (
    CompactionTriggerError,
    compact_response_payload,
    compaction_output_item,
    compaction_summary_from_output,
    decode_shim_compaction_summary,
    encode_shim_compaction_summary,
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

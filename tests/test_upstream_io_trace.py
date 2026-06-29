from __future__ import annotations

from codex_shim.upstream_io_trace import summarize_upstream_body


def test_summarize_upstream_body_lists_all_input_item_types():
    body = {
        "model": "codex-gpt-5-5",
        "stream": True,
        "input": [
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "one"}]},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "two"}]},
            {"type": "compaction_trigger"},
        ],
    }
    summary = summarize_upstream_body(body)
    assert summary["input_items"] == 3
    assert summary["input_item_types"][-1] == "compaction_trigger"
    assert "chars=3" in summary["input_item_types"][0]

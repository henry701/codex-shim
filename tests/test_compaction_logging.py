from __future__ import annotations

from codex_shim.compaction.input_audit import CompactionInputItemRef, CompactionSanitizationAudit
from codex_shim.compaction.logging import log_compaction_sanitization


def test_log_compaction_sanitization_emits_drop_and_preserve_lines(capsys):
    audit = CompactionSanitizationAudit(incoming_items=3, outgoing_items=2)
    audit.dropped.append(
        (
            CompactionInputItemRef(index=0, item_type="function_call_output", call_id="call_bad"),
            "no in-batch function_call/custom_tool_call for call_id",
        )
    )
    audit.preserved.append(
        (
            CompactionInputItemRef(index=2, item_type="function_call_output", call_id="call_tail"),
            "detached compaction tail after last in-batch tool call",
        )
    )
    log_compaction_sanitization(audit)
    captured = capsys.readouterr().out
    assert "[compaction] sanitize-summary" in captured
    assert "sanitize DROP" in captured
    assert "call_id=call_bad" in captured
    assert "sanitize PRESERVE" in captured
    assert "call_id=call_tail" in captured

from __future__ import annotations

from codex_shim.compaction.input_audit import CompactionInputItemRef, CompactionSanitizationAudit
from codex_shim.compaction.logging import log_compaction_sanitization


def test_log_compaction_sanitization_emits_drop_preserve_and_synthesis_lines(capsys):
    audit = CompactionSanitizationAudit(incoming_items=3, outgoing_items=3)
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
    audit.synthesized.append(
        "synthesized function_call before orphan index=1 type=function_call_output call_id=call_x"
    )
    log_compaction_sanitization(audit)
    captured = capsys.readouterr().out
    assert "[compaction] sanitize-summary" in captured
    assert "sanitize DROP" in captured
    assert "call_id=call_bad" in captured
    assert "sanitize PRESERVE" in captured
    assert "call_id=call_tail" in captured
    assert "sanitize SYNTH" in captured
    assert "call_id=call_x" in captured

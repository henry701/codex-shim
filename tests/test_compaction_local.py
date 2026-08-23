from __future__ import annotations

import asyncio

from codex_shim.compaction.config import CompactionSettings
from codex_shim.compaction.local import (
    deterministic_fallback_summary,
    select_summarization_span,
    serialize_conversation,
    summary_is_usable,
    verbatim_user_quotes,
)
from codex_shim.compaction.orchestrator import CompactionOrchestrator
from codex_shim.compaction.protocol import decode_shim_compaction_summary
from codex_shim.compaction.strategies.bodies import build_summarization_compact_body
from codex_shim.compaction.types import (
    CompactionAdapters,
    CompactionRequest,
    NativeAttemptResult,
    SummarizationAttemptResult,
)


def _msg(role: str, text: str) -> dict:
    return {
        "type": "message",
        "role": role,
        "content": [{"type": "input_text", "text": text}],
    }


def _call(index: int, *, output: str = "ok") -> list[dict]:
    call_id = f"c{index}"
    return [
        {
            "type": "function_call",
            "call_id": call_id,
            "name": "exec_command",
            "arguments": '{"cmd":"pwd"}',
        },
        {"type": "function_call_output", "call_id": call_id, "output": output},
    ]


def _ox_history(*, followup: bool = True) -> list[dict]:
    items: list[dict] = [
        _msg("developer", "You are Codex."),
        _msg("developer", "Follow AGENTS.md."),
        _msg("user", "Ship a local compaction module that never drops user work."),
    ]
    for index in range(30):
        items.extend(_call(index, output=f"edited src/file_{index}.py"))
    if followup:
        items.append(_msg("user", "keep going"))
    return items


def test_select_span_ignores_developer_preamble_and_keeps_first_user_plus_tools():
    span = select_summarization_span(
        _ox_history(),
        tail_turns=2,
        preserve_recent_tokens=8000,
    )
    blob = str(span.head)
    assert "Ship a local compaction module that never drops user work." in blob
    assert any(item.get("type") == "function_call" for item in span.head)
    assert not any(
        isinstance(item, dict)
        and item.get("role") == "developer"
        and "You are Codex." in str(item)
        and len(span.head) <= 2
        for item in span.head[:2]
    )


def test_select_span_never_excludes_the_only_user_goal():
    span = select_summarization_span(
        _ox_history(followup=False),
        tail_turns=2,
        preserve_recent_tokens=8000,
    )
    assert span.tail == []
    assert "Ship a local compaction module that never drops user work." in str(span.head)
    assert sum(1 for item in span.head if item.get("type") == "function_call") == 30


def test_select_span_can_exclude_last_user_turn_but_keeps_oldest_in_window():
    items = _ox_history(followup=True)
    span = select_summarization_span(
        items,
        tail_turns=1,
        preserve_recent_tokens=50_000,
    )
    assert "keep going" in str(span.tail)
    assert "Ship a local compaction module that never drops user work." in str(span.head)
    assert "keep going" not in str(span.head)
    assert any(item.get("type") == "function_call" for item in span.head)


def test_huge_last_turn_stays_in_head_when_it_exceeds_token_budget():
    items = [
        _msg("user", "First task: rewrite compaction."),
        *_call(0, output="small"),
        _msg("user", "Now dump a giant log."),
        *_call(1, output="x" * 80_000),
    ]
    span = select_summarization_span(
        items,
        tail_turns=1,
        preserve_recent_tokens=100,
    )
    assert "First task: rewrite compaction." in str(span.head)
    assert "Now dump a giant log." in str(span.head)
    assert span.tail == []


def test_serialize_conversation_quotes_user_and_tools():
    text = serialize_conversation(_ox_history(followup=False))
    assert "[user]:" in text.lower() or "[User]:" in text
    assert "never drops user work" in text
    assert "exec_command" in text
    assert "edited src/file_0.py" in text


def _long_session(*, prompts: int = 60) -> list[dict]:
    items: list[dict] = [
        {
            "type": "compaction",
            "summary": "older compacted work from before the recency window",
        },
        _msg("developer", "You are Codex."),
    ]
    for index in range(prompts):
        items.append(
            _msg("user", f"Prompt {index:03d}: do step {index} of the long-running task.")
        )
        items.extend(_call(index, output=f"did step {index} on src/file_{index}.py"))
    return items


def test_select_span_drops_stale_user_prompts_outside_recency_cap():
    items = _long_session(prompts=60)
    span = select_summarization_span(
        items,
        tail_turns=1,
        preserve_recent_tokens=50_000,
        max_recent_user_prompts=50,
    )
    head = str(span.head)
    tail = str(span.tail)
    assert "Prompt 000:" not in head
    assert "Prompt 009:" not in head
    assert "Prompt 010:" in head
    assert "Prompt 058:" in head
    assert "Prompt 059:" in tail
    assert "Prompt 059:" not in head
    assert "You are Codex." in head
    assert "older compacted work from before the recency window" in head
    assert any(item.get("type") == "function_call" for item in span.head)


def test_verbatim_user_quotes_cap_to_recent_prompts():
    short = verbatim_user_quotes(_ox_history())
    assert "never drops user work" in short
    assert "keep going" in short
    assert "You are Codex." not in short

    quotes = verbatim_user_quotes(_long_session(prompts=60), max_prompts=50)
    assert "Prompt 000:" not in quotes
    assert "Prompt 009:" not in quotes
    assert "Prompt 010:" in quotes
    assert "Prompt 059:" in quotes
    assert 'omitted_older="10"' in quotes


def test_deterministic_fallback_goal_is_recent_not_first():
    summary = deterministic_fallback_summary(
        _long_session(prompts=60),
        previous_summary=None,
        reason="summarizer returned an empty summary",
    )
    assert "## Goal" in summary
    assert "Prompt 059:" in summary
    assert "Prompt 000:" not in summary
    assert "Prompt 010:" in summary
    assert "exec_command" in summary


def test_summary_is_usable_requires_recent_not_first_user_snippet():
    items = _long_session(prompts=60)
    filler = "progress notes " * 80
    with_stale = f"## Goal\n- Prompt 000: do step 0 of the long-running task.\n\n{filler}"
    with_recent = f"## Goal\n- Prompt 059: do step 59 of the long-running task.\n\n{filler}"
    assert summary_is_usable(with_stale, items=items) is False
    assert summary_is_usable(with_recent, items=items) is True
    summary = deterministic_fallback_summary(
        _ox_history(),
        previous_summary=None,
        reason="summarizer returned an empty summary",
    )
    assert "## Goal" in summary
    assert "never drops user work" in summary
    assert "exec_command" in summary
    assert "file_0.py" in summary


def test_summary_is_usable_rejects_generic_short_handoff():
    items = _ox_history()
    junk = (
        "## Goal\n- Continue the conversation\n\n## Constraints & Preferences\n- (none)\n\n"
        "## Progress\n### Done\n- (none)\n### In Progress\n- (none)\n### Blocked\n- (none)\n\n"
        "## Key Decisions\n- (none)\n\n## Next Steps\n- (none)\n\n"
        "## Critical Context\n- (none)\n\n## Relevant Files\n- (none)\n"
    )
    assert summary_is_usable(junk, items=items) is False
    good = deterministic_fallback_summary(items, previous_summary=None, reason="test")
    assert summary_is_usable(good, items=items) is True


def test_summary_is_usable_requires_recent_user_snippet():
    items = _ox_history()
    padded = (
        "## Goal\n- Continue the conversation\n\n"
        + ("filler " * 200)
        + "\n## Progress\n### Done\n- tools ran\n"
    )
    assert summary_is_usable(padded, items=items) is False
    with_goal = (
        "## Goal\n"
        "- Ship a local compaction module that never drops user work.\n\n"
        + ("progress notes " * 80)
    )
    assert summary_is_usable(with_goal, items=items) is True


def test_summarization_body_embeds_verbatim_user_quotes():
    from codex_shim.compaction.pipeline import prepare_compaction_input

    prepared = prepare_compaction_input(_ox_history(), CompactionSettings(tail_turns=2))
    body = build_summarization_compact_body(
        prepared,
        body={"model": "nous-stealth-ox-alpha"},
        upstream_model="stealth-ox-alpha",
        requested_slug="nous-stealth-ox-alpha",
        settings=CompactionSettings(),
    )
    blob = str(body["input"])
    assert "never drops user work" in blob
    assert "keep going" in blob


def _adapters(
    *,
    summary: str = "",
    tertiary: str = "",
) -> CompactionAdapters:
    async def native_byok(request, prepared):
        return NativeAttemptResult(native_message="native compact HTTP 502")

    async def summarization_byok(request, prepared, native_message):
        return SummarizationAttemptResult(summary=summary)

    async def tertiary_byok(request, prepared, native_message, slug):
        return SummarizationAttemptResult(summary=tertiary)

    async def unused(*args, **kwargs):
        raise AssertionError("unexpected adapter")

    return CompactionAdapters(
        native_chatgpt=unused,
        native_cursor=unused,
        native_byok=native_byok,
        summarization_chatgpt=unused,
        summarization_cursor=unused,
        summarization_byok=summarization_byok,
        tertiary_byok=tertiary_byok,
        acquire_chatgpt_lock=unused,
    )


def test_orchestrator_uses_local_fallback_when_llm_summary_is_unusable():
    items = _ox_history()
    request = CompactionRequest(
        http_request=object(),
        body={"model": "nous-stealth-ox-alpha"},
        stripped_input=items,
        requested_slug="nous-stealth-ox-alpha",
        provider="byok",
        skip_native=True,
        preset_native_message="native compact HTTP 502",
        settings=CompactionSettings(fallback_enabled=True),
    )
    orchestrator = CompactionOrchestrator(
        _adapters(summary="ok", tertiary=""),
    )

    result = asyncio.run(orchestrator.run(request))
    decoded = decode_shim_compaction_summary(result.item.get("encrypted_content")) or ""
    assert "never drops user work" in decoded
    assert result.phase == "local_fallback"

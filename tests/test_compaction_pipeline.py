from __future__ import annotations

import asyncio

from codex_shim.compaction.config import (
    CompactionSettings,
    compaction_prompt_cache_key,
    effective_compaction_output_token_reserve,
)
from codex_shim.compaction.model_resolver import CompactionModelResolver
from codex_shim.compaction.pipeline import (
    STILL_OVER_BUDGET_WARNING,
    collapse_consecutive_duplicate_user_messages,
    estimate_input_tokens,
    extract_previous_summary,
    prepare_compaction_input,
    rewrite_tool_outputs_for_context_window,
    truncate_tool_output_chars,
)
from codex_shim.compaction.prompts import (
    build_summarization_user_prompt,
    stable_summarization_instruction_prefix,
)
from codex_shim.compaction.protocol import encode_shim_compaction_summary


def _tool_heavy_history(*, output_chars: int) -> list[dict]:
    text = "x" * output_chars
    return [
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "run tools"}]},
        {"type": "function_call", "call_id": "c1", "name": "exec_command", "arguments": "{}"},
        {
            "type": "function_call_output",
            "call_id": "c1",
            "output": [{"type": "input_text", "text": text}],
        },
    ]


def test_prepare_compaction_input_synthesizes_detached_orphan_only_tail():
    items = [
        {
            "type": "function_call_output",
            "call_id": "call_tail",
            "output": "truncated tool output",
        },
    ]
    prepared = prepare_compaction_input(items, CompactionSettings())
    assert len(prepared.native_input) == 2
    assert prepared.native_input[0]["type"] == "function_call"
    assert prepared.native_input[1]["call_id"] == "call_tail"
    assert any("synthesized" in warning for warning in prepared.warnings)


def test_prepare_compaction_input_synthesizes_orphans_and_splits_summarization_input():
    settings = CompactionSettings(tail_turns=1, tool_output_max_chars=10)
    items = [
        {"type": "function_call_output", "call_id": "orphan", "output": "x" * 20},
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "old"}]},
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "new"}]},
    ]
    prepared = prepare_compaction_input(items, settings, compaction_model_context_window=128_000)
    assert len(prepared.native_input) == 4
    assert any("synthesized" in warning for warning in prepared.warnings)
    assert len(prepared.summarization_input) == 3


def _msg(role: str, text: str) -> dict:
    return {
        "type": "message",
        "role": role,
        "content": [{"type": "input_text", "text": text}],
    }


def _tool_round(index: int, *, output: str = "ok") -> list[dict]:
    call_id = f"call_{index}"
    return [
        {
            "type": "function_call",
            "call_id": call_id,
            "name": "exec_command",
            "arguments": '{"cmd":"ls"}',
        },
        {
            "type": "function_call_output",
            "call_id": call_id,
            "output": output,
        },
    ]


def test_developer_preamble_is_not_a_user_turn_and_does_not_drop_work():
    """Codex prefixes developer instructions; those must not eat tail_turns."""
    items: list[dict] = [
        _msg("developer", "You are Codex."),
        _msg("developer", "Follow the repo AGENTS.md."),
        _msg("user", "Implement local compaction that never drops the task."),
    ]
    for index in range(40):
        items.extend(_tool_round(index, output=f"changed file_{index}.py"))
    items.append(_msg("user", "continue"))

    prepared = prepare_compaction_input(items, CompactionSettings(tail_turns=2))
    blob = str(prepared.summarization_input)
    assert "Implement local compaction that never drops the task." in blob
    assert any(
        isinstance(item, dict) and item.get("type") == "function_call"
        for item in prepared.summarization_input
    )
    assert prepared.stats["summarization_items"] > 4


def test_single_user_turn_with_long_tool_trace_stays_in_summarization_input():
    items: list[dict] = [
        _msg("developer", "system preamble"),
        _msg("developer", "more preamble"),
        _msg("user", "Port the compaction module from OpenCode/Pi/Hermes."),
    ]
    for index in range(25):
        items.extend(_tool_round(index))

    prepared = prepare_compaction_input(items, CompactionSettings(tail_turns=2))
    blob = str(prepared.summarization_input)
    assert "Port the compaction module from OpenCode/Pi/Hermes." in blob
    assert sum(1 for item in prepared.summarization_input if item.get("type") == "function_call") == 25


def test_extract_previous_summary_reads_shim_compaction_item():
    summary = "Prior task state"
    items = [
        {
            "type": "compaction",
            "encrypted_content": encode_shim_compaction_summary(summary),
        }
    ]
    assert extract_previous_summary(items) == summary


def test_rewrite_tool_outputs_for_context_window_replaces_oldest_output():
    items = [
        {"type": "function_call", "call_id": "c1", "name": "shell", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "c1", "output": "a" * 4000},
        {"type": "function_call", "call_id": "c2", "name": "shell", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "c2", "output": "b" * 4000},
    ]
    rewritten, count = rewrite_tool_outputs_for_context_window(items, token_budget=50)
    assert count >= 1
    assert any(
        item.get("output") == "Output exceeded the available model context and was truncated"
        for item in rewritten
        if item.get("type") == "function_call_output"
    )


def test_truncates_codex_content_item_tool_output():
    huge = "x" * 5000
    items = [
        {
            "type": "function_call_output",
            "call_id": "c1",
            "output": [{"type": "input_text", "text": huge}],
        }
    ]
    truncated, count, chars_removed = truncate_tool_output_chars(items, max_chars=2000)
    assert count == 1
    assert chars_removed == 3000
    assert isinstance(truncated[0]["output"], str)
    assert len(truncated[0]["output"]) < len(huge)


def test_same_model_context_skips_trim_when_history_fits():
    items = _tool_heavy_history(output_chars=80_000)
    settings = CompactionSettings()
    prepared = prepare_compaction_input(
        items,
        settings,
        compaction_model_context_window=128_000,
    )
    assert prepared.stats["truncated_tool_outputs"] == 0
    assert prepared.stats["rewritten_tool_outputs"] == 0
    reserve = effective_compaction_output_token_reserve(settings)
    assert prepared.stats["compaction_input_token_budget"] == 128_000 - reserve
    assert (
        prepared.stats["estimated_input_tokens"]
        <= prepared.stats["compaction_input_token_budget"]
    )


def test_one_million_context_override_skips_trim_for_moderate_history():
    items = _tool_heavy_history(output_chars=200_000)
    settings = CompactionSettings(model="big-1m", override_current_model=True)
    prepared = prepare_compaction_input(
        items,
        settings,
        compaction_model_context_window=1_000_000,
    )
    assert prepared.stats["truncated_tool_outputs"] == 0
    assert prepared.stats["rewritten_tool_outputs"] == 0
    assert prepared.stats["compaction_input_token_budget"] == (
        1_000_000 - effective_compaction_output_token_reserve(settings)
    )


def test_same_model_truncates_only_when_over_budget():
    items = _tool_heavy_history(output_chars=520_000)
    settings = CompactionSettings(tool_output_max_chars=2000)
    prepared = prepare_compaction_input(
        items,
        settings,
        compaction_model_context_window=128_000,
    )
    assert prepared.stats["truncated_tool_outputs"] >= 1
    assert prepared.stats["chars_truncated_from_tool_outputs"] > 1000


def test_same_model_prunes_when_still_over_budget_after_truncation():
    items = [
        {"type": "function_call", "call_id": "c1", "name": "shell", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "c1", "output": "a" * 50_000},
        {"type": "function_call", "call_id": "c2", "name": "shell", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "c2", "output": "b" * 50_000},
    ]
    settings = CompactionSettings(
        tool_output_max_chars=2000,
        compaction_output_token_reserve=127_000,
    )
    prepared = prepare_compaction_input(
        items,
        settings,
        compaction_model_context_window=128_000,
    )
    assert prepared.stats["compaction_input_token_budget"] == 1_000
    assert prepared.stats["truncated_tool_outputs"] >= 1
    assert prepared.stats["rewritten_tool_outputs"] >= 1


def test_warns_when_still_over_budget_after_truncation_and_pruning():
    huge_user = "u" * 500_000
    items = [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": huge_user}],
        },
        {"type": "function_call", "call_id": "c1", "name": "shell", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "c1", "output": "t" * 4000},
    ]
    settings = CompactionSettings(tool_output_max_chars=2000)
    prepared = prepare_compaction_input(
        items,
        settings,
        compaction_model_context_window=20_000,
    )
    assert any(STILL_OVER_BUDGET_WARNING in warning for warning in prepared.warnings)
    assert (
        prepared.stats["estimated_input_tokens_after_pruning"]
        > prepared.stats["compaction_input_token_budget"]
    )


def test_no_budget_skips_all_shim_side_trimming():
    items = _tool_heavy_history(output_chars=500_000)
    prepared = prepare_compaction_input(
        items,
        CompactionSettings(),
        compaction_model_context_window=None,
    )
    assert prepared.stats["truncated_tool_outputs"] == 0
    assert prepared.stats["rewritten_tool_outputs"] == 0
    assert "compaction_input_token_budget" not in prepared.stats


def test_build_summarization_user_prompt_anchors_previous_summary():
    prompt = build_summarization_user_prompt(previous_summary="Old goal", recent_user_turns_excluded=2)
    assert "<previous-summary>" in prompt
    assert "Old goal" in prompt
    assert "most recent 2 user turn(s)" in prompt


def test_stable_summarization_instruction_prefix_is_stable():
    first = stable_summarization_instruction_prefix()
    second = stable_summarization_instruction_prefix()
    assert first == second
    assert len(first) > 500


def test_compaction_model_resolver_prefers_configured_model():
    settings = CompactionSettings(model="gpt-5.4-mini", override_current_model=True)

    async def _run():
        return await CompactionModelResolver(settings).resolve(
            requested_slug="codex-gpt-5-5",
            body={"model": "codex-gpt-5-5"},
        )

    resolved = asyncio.run(_run())
    assert resolved.summarization_slug == "gpt-5.4-mini"
    assert resolved.native_slug == "codex-gpt-5-5"


def test_compaction_model_resolver_async_route_fn_resolves_tertiary():
    settings = CompactionSettings()

    class Route:
        slug = "or-free-router"
        api_key = "secret"

    async def route_fn(body):
        assert body["model"] == "or-free-router"
        return Route()

    async def _run():
        return await CompactionModelResolver(
            settings,
            route_fn=route_fn,
            has_credentials_fn=lambda route: bool(route.api_key),
        ).resolve(
            requested_slug="codex-gpt-5-4-mini",
            body={"model": "codex-gpt-5-4-mini"},
            passthrough_fallback_slug="or-free-router",
        )

    resolved = asyncio.run(_run())
    assert resolved.tertiary_slug == "or-free-router"
    assert resolved.tertiary_skip_reason is None


def test_compaction_model_resolver_async_route_fn_no_credentials():
    settings = CompactionSettings()

    class Route:
        slug = "or-free-router"
        api_key = ""

    async def route_fn(body):
        return Route()

    async def _run():
        return await CompactionModelResolver(
            settings,
            route_fn=route_fn,
            has_credentials_fn=lambda route: bool(route.api_key),
        ).resolve(
            requested_slug="codex-gpt-5-4-mini",
            body={},
            passthrough_fallback_slug="or-free-router",
        )

    resolved = asyncio.run(_run())
    assert resolved.tertiary_slug is None
    assert resolved.tertiary_skip_reason == "no_credentials"
    assert resolved.tertiary_configured_slug == "or-free-router"


def test_compaction_model_resolver_not_configured_skip_reason():
    async def _run():
        return await CompactionModelResolver(CompactionSettings()).resolve(
            requested_slug="codex-gpt-5-4-mini",
            body={},
        )

    resolved = asyncio.run(_run())
    assert resolved.tertiary_slug is None
    assert resolved.tertiary_skip_reason == "not_configured"


def test_collapse_consecutive_duplicate_user_messages():
    dup = {"type": "message", "role": "user", "content": "same"}
    items = [dup, dup, dup, {"type": "message", "role": "user", "content": "other"}, dup]
    collapsed, count = collapse_consecutive_duplicate_user_messages(items)
    assert count == 2
    assert len(collapsed) == 3
    assert collapsed[0]["content"] == "same"
    assert collapsed[1]["content"] == "other"
    assert collapsed[2]["content"] == "same"


def test_prepare_compaction_input_dedupes_consecutive_user_messages():
    dup = {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "x"}]}
    items = [dup, dup, dup]
    prepared = prepare_compaction_input(items, CompactionSettings())
    assert prepared.stats["deduped_user_messages"] == 2
    assert len(prepared.native_input) == 1


def test_compaction_prompt_cache_key_versioned():
    settings = CompactionSettings(prompt_cache_key_version="v2")
    assert compaction_prompt_cache_key(settings) == "codex-shim-compact:v2"


def test_estimate_input_tokens_scales_with_payload_size():
    small = _tool_heavy_history(output_chars=4_000)
    large = _tool_heavy_history(output_chars=400_000)
    assert estimate_input_tokens(large) > estimate_input_tokens(small)

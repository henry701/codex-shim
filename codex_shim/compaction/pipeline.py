from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import Any

from .codex_reference import CONTEXT_WINDOW_TRUNCATED_OUTPUT_MESSAGE, TOOL_OUTPUT_REWRITE_TYPES
from .config import CompactionSettings
from .context import compute_compaction_input_token_budget
from .input_audit import CompactionSanitizationAudit
from .logging import (
    log_compaction_budget_fit,
    log_compaction_input_snapshot,
    log_compaction_sanitization,
)
from .protocol import decode_shim_compaction_summary, sanitize_compaction_input_items

STILL_OVER_BUDGET_WARNING = (
    "compaction input still exceeds estimated token budget after truncation and pruning; "
    "continuing with fallback compaction"
)


def _approx_token_count(text: str) -> int:
    return max(1, len(text) // 4)


def _estimate_item_chars(item: Any) -> int:
    try:
        return len(json.dumps(item, default=str))
    except Exception:
        return len(str(item))


def estimate_input_tokens(items: list[Any], *, instructions_chars: int = 0) -> int:
    total = _approx_token_count(" " * instructions_chars) if instructions_chars else 0
    for item in items:
        total += max(1, _estimate_item_chars(item) // 4)
    return total


def _is_user_turn_boundary(item: dict[str, Any]) -> bool:
    if item.get("type") != "message":
        return False
    role = item.get("role")
    return role in {"user", "developer"}


def _user_turn_start_indices(items: list[Any]) -> list[int]:
    indices: list[int] = []
    for index, raw in enumerate(items):
        if isinstance(raw, dict) and _is_user_turn_boundary(raw):
            indices.append(index)
    return indices


def extract_previous_summary(items: list[Any]) -> str | None:
    for raw in reversed(items):
        if not isinstance(raw, dict):
            continue
        if raw.get("type") not in {"compaction", "compaction_summary"}:
            continue
        decoded = decode_shim_compaction_summary(raw.get("encrypted_content"))
        if decoded:
            return decoded
    return None


def _tool_output_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for block in value:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type in {"input_text", "text", "output_text"}:
                parts.append(str(block.get("text") or block.get("content") or ""))
        return "\n".join(parts)
    return ""


def _truncate_output_value(value: Any, max_chars: int) -> tuple[Any, int]:
    if max_chars <= 0:
        return value, 0
    if isinstance(value, str):
        text = value
        original = value
    elif isinstance(value, list):
        text = _tool_output_text(value)
        original = value
    else:
        return value, 0
    if len(text) <= max_chars:
        return original, 0
    omitted = len(text) - max_chars
    truncated = f"{text[:max_chars]}\n[truncated for compaction: {omitted} chars omitted]"
    return truncated, omitted


def truncate_tool_output_chars(items: list[Any], max_chars: int) -> tuple[list[Any], int, int]:
    if max_chars <= 0:
        return items, 0, 0
    rewritten = copy.deepcopy(items)
    count = 0
    chars_removed = 0
    for item in rewritten:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type in {"function_call_output", "custom_tool_call_output"}:
            output, removed = _truncate_output_value(item.get("output"), max_chars)
            if removed > 0:
                item["output"] = output
                count += 1
                chars_removed += removed
    return rewritten, count, chars_removed


def _rewrite_tool_output_item(item: dict[str, Any]) -> dict[str, Any] | None:
    item_type = item.get("type")
    if item_type not in TOOL_OUTPUT_REWRITE_TYPES:
        return None
    rewritten = copy.deepcopy(item)
    if item_type in {"function_call_output", "custom_tool_call_output"}:
        rewritten["output"] = CONTEXT_WINDOW_TRUNCATED_OUTPUT_MESSAGE
        return rewritten
    if item_type == "tool_search_output":
        rewritten["tools"] = []
        return rewritten
    return None


def rewrite_tool_outputs_for_context_window(
    items: list[Any],
    *,
    token_budget: int,
    instructions_chars: int = 0,
) -> tuple[list[Any], int]:
    """Mirror Codex trim_function_call_history: rewrite oldest eligible outputs from the end."""
    if token_budget <= 0:
        return items, 0
    working = copy.deepcopy(items)
    rewritten = 0
    while estimate_input_tokens(working, instructions_chars=instructions_chars) > token_budget:
        replaced = False
        for index in range(len(working) - 1, -1, -1):
            raw = working[index]
            if not isinstance(raw, dict):
                continue
            candidate = _rewrite_tool_output_item(raw)
            if candidate is None:
                continue
            if candidate == raw:
                continue
            working[index] = candidate
            rewritten += 1
            replaced = True
            break
        if not replaced:
            break
    return working, rewritten


def exclude_recent_user_turns(
    items: list[Any],
    *,
    tail_turns: int,
) -> tuple[list[Any], list[Any], int]:
    """Split for summarization input only; client retains user messages post-compact."""
    if tail_turns <= 0 or not items:
        return items, [], 0
    turn_starts = _user_turn_start_indices(items)
    if len(turn_starts) <= tail_turns:
        return [], items, len(turn_starts)
    split_at = turn_starts[-tail_turns]
    return items[:split_at], items[split_at:], tail_turns


@dataclass
class PreparedInput:
    native_input: list[Any]
    summarization_input: list[Any]
    previous_summary: str | None
    warnings: list[str] = field(default_factory=list)
    excluded_user_turns: int = 0
    stats: dict[str, int] = field(default_factory=dict)
    sanitization_audit: CompactionSanitizationAudit | None = None


def _fit_compaction_input_to_budget(
    working: list[Any],
    *,
    input_budget: int,
    instructions_chars: int,
    settings: CompactionSettings,
    stats: dict[str, int],
    warnings: list[str],
) -> list[Any]:
    estimated = estimate_input_tokens(working, instructions_chars=instructions_chars)
    stats["estimated_input_tokens"] = estimated
    if estimated <= input_budget:
        log_compaction_budget_fit(estimated=estimated, budget=input_budget)
        return working

    if settings.tool_output_max_chars > 0:
        working, truncated, chars_removed = truncate_tool_output_chars(
            working,
            settings.tool_output_max_chars,
        )
        stats["truncated_tool_outputs"] = truncated
        stats["chars_truncated_from_tool_outputs"] = chars_removed
        estimated = estimate_input_tokens(working, instructions_chars=instructions_chars)
        stats["estimated_input_tokens_after_truncation"] = estimated

    if estimated > input_budget:
        working, rewritten = rewrite_tool_outputs_for_context_window(
            working,
            token_budget=input_budget,
            instructions_chars=instructions_chars,
        )
        stats["rewritten_tool_outputs"] = rewritten
        if rewritten:
            warnings.append(
                f"rewrote {rewritten} tool output(s) to fit compaction input token budget"
            )
        estimated = estimate_input_tokens(working, instructions_chars=instructions_chars)
        stats["estimated_input_tokens_after_pruning"] = estimated

    if estimated > input_budget:
        warnings.append(STILL_OVER_BUDGET_WARNING)
        print(
            f"[warn] compaction: {STILL_OVER_BUDGET_WARNING} "
            f"(estimated={estimated}, budget={input_budget})",
            flush=True,
        )

    log_compaction_budget_fit(
        estimated=stats.get(
            "estimated_input_tokens_after_pruning",
            stats.get("estimated_input_tokens_after_truncation", estimated),
        ),
        budget=input_budget,
        truncated=stats.get("truncated_tool_outputs", 0),
        chars_removed=stats.get("chars_truncated_from_tool_outputs", 0),
        rewritten=stats.get("rewritten_tool_outputs", 0),
        still_over=estimated > input_budget,
    )

    return working


def prepare_compaction_input(
    stripped_input: list[Any],
    settings: CompactionSettings,
    *,
    client_instructions: str | None = None,
    compaction_model_context_window: int | None = None,
) -> PreparedInput:
    warnings: list[str] = []
    stats: dict[str, int] = {
        "input_items": len(stripped_input),
        "rewritten_tool_outputs": 0,
        "truncated_tool_outputs": 0,
        "chars_truncated_from_tool_outputs": 0,
    }

    log_compaction_input_snapshot("pre-sanitize", stripped_input)

    cleaned, orphan_warnings, sanitization_audit = sanitize_compaction_input_items(stripped_input)
    log_compaction_sanitization(sanitization_audit)
    warnings.extend(orphan_warnings)
    stats["sanitization_dropped"] = len(sanitization_audit.dropped)
    stats["sanitization_preserved"] = len(sanitization_audit.preserved)

    instructions_chars = len((client_instructions or "").strip())
    working = cleaned

    log_compaction_input_snapshot("post-sanitize", working)

    input_budget = compute_compaction_input_token_budget(
        compaction_model_context_window,
        settings,
    )
    if input_budget is not None:
        stats["compaction_input_token_budget"] = input_budget
        stats["compaction_model_context_window"] = compaction_model_context_window or 0
        working = _fit_compaction_input_to_budget(
            working,
            input_budget=input_budget,
            instructions_chars=instructions_chars,
            settings=settings,
            stats=stats,
            warnings=warnings,
        )
    else:
        stats["estimated_input_tokens"] = estimate_input_tokens(
            working,
            instructions_chars=instructions_chars,
        )

    previous_summary = extract_previous_summary(working)

    head, tail, excluded = exclude_recent_user_turns(
        working,
        tail_turns=settings.tail_turns,
    )
    summarization_input = head if head else working
    stats["summarization_items"] = len(summarization_input)
    stats["excluded_tail_items"] = len(tail)

    return PreparedInput(
        native_input=working,
        summarization_input=summarization_input,
        previous_summary=previous_summary,
        warnings=warnings,
        excluded_user_turns=excluded,
        stats=stats,
        sanitization_audit=sanitization_audit,
    )

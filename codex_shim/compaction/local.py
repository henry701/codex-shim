from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

TOOL_CALL_TYPES = frozenset(
    {"function_call", "function_call", "custom_tool_call", "custom_tool_call"}
)
TOOL_OUTPUT_TYPES = frozenset(
    {
        "function_call_output",
        "function_call_output",
        "custom_tool_call_output",
        "custom_tool_call_output",
    }
)
COMPACTION_ITEM_TYPES = frozenset({"compaction", "compaction_summary", "compaction_trigger"})
SERIALIZE_TOOL_OUTPUT_MAX_CHARS = 2000
# Nous/Ox LLM stubs were ~400-600 chars for ~200 items. Scale the floor
# with history size so a small local fallback is not rejected.
MIN_SUMMARY_CHARS_WITH_TOOLS = 800
MIN_SUMMARY_CHARS_FLOOR = 240
SUMMARY_CHARS_PER_ITEM = 4
# Require a contiguous slice of a recent user message. Skip if shorter.
MIN_USER_SNIPPET_CHARS = 24
USER_SNIPPET_CHARS = 64
DEFAULT_MAX_RECENT_USER_PROMPTS = 50
FILE_RE = re.compile(
    r"(?:^|[\s`\"'(=])((?:[A-Za-z0-9_.-]+/)+\.?[A-Za-z0-9_.-]+(?:\.[A-Za-z0-9]+)?)"
)


@dataclass(frozen=True)
class SummarizationSpan:
    head: list[Any]
    tail: list[Any]
    excluded_user_turns: int


def item_text(item: Any) -> str:
    if not isinstance(item, dict):
        return str(item or "")
    chunks: list[str] = []
    content = item.get("content")
    if isinstance(content, str):
        chunks.append(content)
    elif isinstance(content, list):
        for block in content:
            if isinstance(block, dict):
                chunks.append(str(block.get("text") or block.get("content") or ""))
            elif isinstance(block, str):
                chunks.append(block)
    for key in ("text", "output", "summary", "arguments", "input"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            chunks.append(value)
        elif isinstance(value, list):
            for block in value:
                if isinstance(block, dict):
                    chunks.append(str(block.get("text") or block.get("content") or ""))
    return "\n".join(part for part in chunks if part).strip()


def is_real_user_turn(item: Any) -> bool:
    """OpenCode/Pi/Hermes: a turn starts at a real user message, never developer/system."""
    if not isinstance(item, dict):
        return False
    item_type = item.get("type")
    if item_type in COMPACTION_ITEM_TYPES:
        return False
    if item.get("role") != "user":
        return False
    if item_type not in {None, "message"}:
        return False
    text = item_text(item)
    if not text:
        return False
    if "compacted conversation" in text.lower() and len(text) < 240:
        return False
    return True


def has_tool_work(items: list[Any]) -> bool:
    return any(
        isinstance(item, dict) and item.get("type") in TOOL_CALL_TYPES | TOOL_OUTPUT_TYPES
        for item in items
    )


def estimate_item_tokens(item: Any) -> int:
    try:
        payload = json.dumps(item, default=str)
    except Exception:
        payload = str(item)
    return max(1, len(payload) // 4)


def estimate_items_tokens(items: list[Any]) -> int:
    return sum(estimate_item_tokens(item) for item in items)


def user_turn_start_indices(items: list[Any]) -> list[int]:
    return [index for index, item in enumerate(items) if is_real_user_turn(item)]


def recent_user_texts(items: list[Any], *, max_prompts: int = DEFAULT_MAX_RECENT_USER_PROMPTS) -> list[str]:
    texts = [item_text(item).strip() for item in items if is_real_user_turn(item)]
    texts = [text for text in texts if text]
    if max_prompts <= 0:
        return texts
    return texts[-max_prompts:]


def _is_compaction_item(item: Any) -> bool:
    return isinstance(item, dict) and item.get("type") in COMPACTION_ITEM_TYPES


def _keep_from_stale_prefix(item: Any) -> bool:
    """Older user turns go stale; developer/system preamble and prior summaries do not."""
    if _is_compaction_item(item):
        return True
    if not isinstance(item, dict):
        return False
    if item.get("type") not in {None, "message"}:
        return False
    return item.get("role") in {"developer", "system"}


def _exclude_tail_within(
    items: list[Any],
    *,
    tail_turns: int,
    preserve_recent_tokens: int,
) -> tuple[list[Any], list[Any], int]:
    if tail_turns <= 0 or not items:
        return list(items), [], 0
    starts = user_turn_start_indices(items)
    if len(starts) <= 1:
        return list(items), [], 0
    budget = max(0, preserve_recent_tokens)
    split_at: int | None = None
    excluded = 0
    tail_tokens = 0
    # Keep the oldest user turn *in this window* in the head so a 1–2 turn
    # session cannot collapse to developer preamble + dropped tools.
    for index in range(len(starts) - 1, 0, -1):
        start = starts[index]
        end = starts[index + 1] if index + 1 < len(starts) else len(items)
        turn_tokens = estimate_items_tokens(items[start:end])
        if excluded >= tail_turns:
            break
        if budget > 0 and excluded == 0 and turn_tokens > budget:
            break
        if budget > 0 and excluded > 0 and tail_tokens + turn_tokens > budget:
            break
        split_at = start
        tail_tokens += turn_tokens
        excluded += 1
    if split_at is None:
        return list(items), [], 0
    head = list(items[:split_at])
    tail = list(items[split_at:])
    if has_tool_work(items) and not has_tool_work(head):
        return list(items), [], 0
    if not any(is_real_user_turn(item) for item in head):
        return list(items), [], 0
    return head, tail, excluded


def select_summarization_span(
    items: list[Any],
    *,
    tail_turns: int,
    preserve_recent_tokens: int = 8000,
    max_recent_user_prompts: int = DEFAULT_MAX_RECENT_USER_PROMPTS,
) -> SummarizationSpan:
    """Recency-biased split: summarize recent work, drop stale user prompts.

    Safety nets:
    - developer/system preamble is never a turn boundary
    - only the last ``max_recent_user_prompts`` user turns are in play
      (older prompts are stale; previous compaction summaries still carry them)
    - a token-budgeted suffix of those turns stays out of the summarizer
    - if excluding the tail would drop all tool work, exclude nothing
    """
    if not items:
        return SummarizationSpan([], [], 0)

    starts = user_turn_start_indices(items)
    window_from = 0
    if max_recent_user_prompts > 0 and len(starts) > max_recent_user_prompts:
        window_from = starts[-max_recent_user_prompts]

    leading = [item for item in items[:window_from] if _keep_from_stale_prefix(item)]
    window = list(items[window_from:])
    head, tail, excluded = _exclude_tail_within(
        window,
        tail_turns=tail_turns,
        preserve_recent_tokens=preserve_recent_tokens,
    )
    return SummarizationSpan(leading + head, tail, excluded)


def serialize_conversation(
    items: list[Any],
    *,
    tool_output_max_chars: int = SERIALIZE_TOOL_OUTPUT_MAX_CHARS,
) -> str:
    """OpenCode/Pi transcript: labeled history, not a live conversation to continue."""
    lines: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        role = item.get("role")
        if item_type in COMPACTION_ITEM_TYPES:
            summary = item_text(item)
            if summary:
                lines.append(f"[previous-summary]: {summary}")
            continue
        if item_type in {None, "message"} and role:
            text = item_text(item)
            if text:
                lines.append(f"[{role}]: {text}")
            continue
        if item_type in TOOL_CALL_TYPES:
            name = item.get("name") or item.get("tool") or "tool"
            call_id = item.get("call_id") or item.get("id") or ""
            args = item.get("arguments") or item.get("input") or ""
            if not isinstance(args, str):
                try:
                    args = json.dumps(args, default=str)
                except Exception:
                    args = str(args)
            prefix = f"[tool_call] {name}"
            if call_id:
                prefix += f" call_id={call_id}"
            lines.append(f"{prefix} {args}".rstrip())
            continue
        if item_type in TOOL_OUTPUT_TYPES:
            call_id = item.get("call_id") or ""
            output = item_text(item)
            if len(output) > tool_output_max_chars:
                omitted = len(output) - tool_output_max_chars
                output = f"{output[:tool_output_max_chars]}\n[truncated {omitted} chars]"
            suffix = f" call_id={call_id}" if call_id else ""
            lines.append(f"[tool_result]{suffix} {output}".rstrip())
            continue
        leftover = item_text(item)
        if leftover:
            lines.append(f"[{item_type or 'item'}]: {leftover}")
    return "\n".join(lines)


def verbatim_user_quotes(
    items: list[Any],
    *,
    max_prompts: int = DEFAULT_MAX_RECENT_USER_PROMPTS,
) -> str:
    """Hermes: user instructions are quoted, never paraphrased away."""
    quotes = recent_user_texts(items, max_prompts=max_prompts)
    if not quotes:
        return ""
    omitted = 0
    if max_prompts > 0:
        total = len(recent_user_texts(items, max_prompts=0))
        omitted = max(0, total - len(quotes))
    attrs = f' n="{len(quotes)}"'
    if omitted:
        attrs += f' omitted_older="{omitted}"'
    numbered = "\n".join(f"{index}. {quote}" for index, quote in enumerate(quotes, start=1))
    return f"<user-messages-verbatim{attrs}>\n{numbered}\n</user-messages-verbatim>"


def _most_recent_goal_text(
    items: list[Any],
    *,
    max_prompts: int = DEFAULT_MAX_RECENT_USER_PROMPTS,
) -> str:
    texts = recent_user_texts(items, max_prompts=max_prompts)
    for text in reversed(texts):
        first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
        if len(first_line) >= MIN_USER_SNIPPET_CHARS:
            return text.strip()
    return texts[-1] if texts else ""


def _user_task_snippet(text: str) -> str:
    first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    if len(first_line) < MIN_USER_SNIPPET_CHARS:
        return ""
    return first_line[:USER_SNIPPET_CHARS].casefold()


def summary_is_usable(
    summary: str,
    *,
    items: list[Any],
    previous_summary: str | None = None,
) -> bool:
    """Accept an LLM handoff only if it still carries the task.

    Not an NLP classifier. Empty, missing ## Goal, tiny vs a tool-heavy
    history, or missing a contiguous slice of a recent user message →
    unusable; use the deterministic local fallback.
    """
    text = (summary or "").strip()
    if not text:
        return False
    if not re.search(r"##\s*Goal\b", text, flags=re.I):
        return False
    if has_tool_work(items):
        min_chars = min(
            MIN_SUMMARY_CHARS_WITH_TOOLS,
            max(MIN_SUMMARY_CHARS_FLOOR, SUMMARY_CHARS_PER_ITEM * len(items)),
        )
        if len(text) < min_chars:
            return False
    snippet = ""
    for user_text in reversed(recent_user_texts(items)):
        snippet = _user_task_snippet(user_text)
        if snippet:
            break
    if not snippet:
        return True
    haystack = f"{text}\n{previous_summary or ''}".casefold()
    return snippet in haystack


def _collect_tool_names(items: list[Any], *, limit: int = 12) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, dict) or item.get("type") not in TOOL_CALL_TYPES:
            continue
        name = str(item.get("name") or item.get("tool") or "tool")
        if name in seen:
            continue
        seen.add(name)
        names.append(name)
        if len(names) >= limit:
            break
    return names


def _collect_file_paths(items: list[Any], *, limit: int = 16) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    for item in items:
        for match in FILE_RE.findall(item_text(item)):
            path = match.strip(".,;:)")
            if path in seen or len(path) < 4:
                continue
            seen.add(path)
            found.append(path)
            if len(found) >= limit:
                return found
    return found


def deterministic_fallback_summary(
    items: list[Any],
    *,
    previous_summary: str | None = None,
    reason: str = "summarizer unavailable",
) -> str:
    """Pi/Hermes last resort: structured handoff reconstructed from raw items."""
    users = recent_user_texts(items)
    goal = _most_recent_goal_text(items) or "(none recoverable)"
    tools = _collect_tool_names(items)
    files = _collect_file_paths(items)
    prev = (previous_summary or "").strip()
    progress_lines = [f"- called {name}" for name in tools] or ["- (none recoverable from items)"]
    later = [text for text in users[-3:] if text.strip() != goal]
    next_steps = [f"- {text}" for text in later] or [
        "- Continue from the protected recent messages after this summary"
    ]
    context_bits = [f"- Local fallback because {reason}."]
    quotes = verbatim_user_quotes(items)
    if quotes:
        context_bits.append(quotes)
    if prev:
        clipped = prev if len(prev) <= 2400 else prev[:2400].rstrip() + "\n...[previous summary truncated]"
        context_bits.append(f"- Previous summary snapshot:\n{clipped}")
    files_lines = [f"- {path}" for path in files] or ["- (none)"]
    return "\n".join(
        [
            "## Goal",
            f"- {goal}",
            "",
            "## Constraints & Preferences",
            "- Recovered locally without an LLM summarizer. Prefer live files and git state over omitted details.",
            "",
            "## Progress",
            "### Done",
            *progress_lines,
            "### In Progress",
            "- (unknown from deterministic fallback)",
            "### Blocked",
            "- (none recoverable)",
            "",
            "## Key Decisions",
            "- (none recoverable from deterministic fallback)",
            "",
            "## Next Steps",
            *next_steps,
            "",
            "## Critical Context",
            *context_bits,
            "",
            "## Relevant Files",
            *files_lines,
        ]
    )

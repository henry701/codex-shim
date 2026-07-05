from __future__ import annotations

from pathlib import Path

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

SUMMARY_TEMPLATE_V1 = """Output exactly the Markdown structure shown inside <template> and keep the section order unchanged. Do not include the <template> tags in your response.
<template>
## Goal
- [single-sentence task summary]

## Constraints & Preferences
- [user constraints, preferences, specs, or "(none)"]

## Progress
### Done
- [completed work or "(none)"]

### In Progress
- [current work or "(none)"]

### Blocked
- [blockers or "(none)"]

## Key Decisions
- [decision and why, or "(none)"]

## Next Steps
- [ordered next actions or "(none)"]

## Critical Context
- [important technical facts, errors, open questions, or "(none)"]

## Relevant Files
- [file or directory path: why it matters, or "(none)"]
</template>

Rules:
- Keep every section, even when empty.
- Use terse bullets, not prose paragraphs.
- Preserve exact file paths, commands, error strings, and identifiers when known.
- Do not mention the summary process or that context was compacted."""

COMPACTION_PROMPT_VERSION = "v1"


def _load_compaction_system_prompt() -> str:
    path = _PROMPTS_DIR / "compaction_system_v1.txt"
    return path.read_text(encoding="utf-8").strip()


COMPACTION_SYSTEM_PROMPT_V1 = _load_compaction_system_prompt()


def default_native_compact_instructions() -> str:
    return (
        "Compact the conversation into a concise state handoff for the next Codex turn. "
        "Preserve the active task, user requirements, important file paths, commands already run, "
        "tool results, decisions, blockers, and the latest state. Omit filler and repeated text."
    )


def native_compact_instructions(client_instructions: str | None) -> str:
    text = (client_instructions or "").strip()
    if text:
        return text
    return default_native_compact_instructions()


def summarization_system_instructions() -> str:
    return COMPACTION_SYSTEM_PROMPT_V1


def build_summarization_user_prompt(
    *,
    previous_summary: str | None = None,
    extra_context: list[str] | None = None,
    recent_user_turns_excluded: int = 0,
) -> str:
    parts: list[str] = []
    if previous_summary:
        parts.append(
            "Update the anchored summary below using the conversation history above.\n"
            "Preserve still-true details, remove stale details, and merge in the new facts.\n"
            f"<previous-summary>\n{previous_summary}\n</previous-summary>"
        )
    else:
        parts.append("Create a new anchored summary from the conversation history.")
    parts.append(SUMMARY_TEMPLATE_V1)
    if recent_user_turns_excluded > 0:
        parts.append(
            f"Note: the most recent {recent_user_turns_excluded} user turn(s) are preserved "
            "verbatim in the thread by the client; focus the summary on earlier context."
        )
    for line in extra_context or []:
        text = line.strip()
        if text:
            parts.append(text)
    return "\n\n".join(parts)


def stable_summarization_instruction_prefix() -> str:
    """Stable prefix for prompt caching (system + template without dynamic content)."""
    return f"{COMPACTION_SYSTEM_PROMPT_V1}\n\n{SUMMARY_TEMPLATE_V1}"

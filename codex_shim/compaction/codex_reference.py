"""Constants and notes mirroring Codex CLI compaction behavior."""

from __future__ import annotations

# codex-rs/core/src/compact_remote.rs
CONTEXT_WINDOW_TRUNCATED_OUTPUT_MESSAGE = (
    "Output exceeded the available model context and was truncated"
)

# codex-rs/core/src/compact_remote_v2.rs
RETAINED_MESSAGE_TOKEN_BUDGET = 64_000

TOOL_OUTPUT_REWRITE_TYPES = frozenset(
    {"function_call_output", "custom_tool_call_output", "tool_search_output"}
)

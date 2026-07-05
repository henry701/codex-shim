from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from .config import CompactionSettings


@dataclass
class CompactionRequest:
    http_request: Any
    body: dict[str, Any]
    stripped_input: list[Any]
    requested_slug: str
    provider: Literal["chatgpt", "cursor", "byok"]
    session_key: str = ""
    transport: Literal["v2", "legacy_compact"] = "v2"
    upstream_model: str | None = None
    route: Any = None
    tool_types: dict[str, str] | None = None
    tool_resolve: dict[str, tuple[str | None, str]] | None = None
    settings: CompactionSettings = field(default_factory=CompactionSettings)
    skip_native: bool = False
    preset_native_message: str = ""
    passthrough_fallback_slug: str | None = None
    compaction_model_context_window: int | None = None


@dataclass
class CompactionResult:
    item: dict[str, Any]
    usage: dict[str, Any] | None = None
    response_slug: str = ""
    warnings: list[str] = field(default_factory=list)
    native_error: str | None = None
    phase: Literal["native", "summarization", "tertiary", "sanitization_only"] = "native"
    legacy_payload: dict[str, Any] | None = None


@dataclass
class NativeAttemptResult:
    item: dict[str, Any] | None = None
    usage: dict[str, Any] | None = None
    error_response: Any = None
    native_status: int | None = None
    native_message: str = ""
    legacy_payload: dict[str, Any] | None = None


@dataclass
class SummarizationAttemptResult:
    summary: str = ""
    usage: dict[str, Any] | None = None
    error_response: Any = None


@dataclass
class CompactionAdapters:
    """Transport adapters injected from ShimServer."""

    native_chatgpt: Any
    native_cursor: Any
    native_byok: Any
    summarization_chatgpt: Any
    summarization_cursor: Any
    summarization_byok: Any
    tertiary_byok: Any
    acquire_chatgpt_lock: Any
    release_chatgpt_lock: Any = None

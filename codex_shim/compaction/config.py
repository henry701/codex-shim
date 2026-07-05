from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from ..settings import DEFAULT_SETTINGS

# Estimated tokens for compaction instructions, tools, and summarization user template.
DEFAULT_COMPACTION_INSTRUCTION_TOKEN_OVERHEAD = 8192
DEFAULT_COMPACTION_OUTPUT_TOKEN_RESERVE = 20_000


@dataclass(frozen=True)
class CompactionSettings:
    model: str | None = None
    override_current_model: bool = False
    fallback_enabled: bool = True
    tail_turns: int = 2
    preserve_recent_tokens: int = 8000
    tool_output_max_chars: int = 2000
    prune_tool_outputs: bool = False
    summary_max_output_tokens: int = 4096
    prompt_cache_key_version: str = "v1"
    prompt_cache_key_per_session: bool = False
    tertiary_fallback_slug: str | None = None
    compaction_output_token_reserve: int | None = None
    use_client_instructions_for_native: bool = True


def effective_compaction_output_token_reserve(settings: CompactionSettings) -> int:
    if settings.compaction_output_token_reserve is not None:
        return max(0, settings.compaction_output_token_reserve)
    return DEFAULT_COMPACTION_OUTPUT_TOKEN_RESERVE


def _coerce_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on"}
    return default


def _coerce_int(value: Any, default: int) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return default


def _coerce_optional_positive_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, str) and value.strip().isdigit():
        parsed = int(value.strip())
        return parsed if parsed > 0 else None
    return None


def _load_output_token_reserve(raw: dict[str, Any]) -> int | None:
    if "compaction_output_token_reserve" in raw:
        return _coerce_optional_positive_int(raw.get("compaction_output_token_reserve"))
    if "context_window_token_budget" in raw:
        legacy = _coerce_optional_positive_int(raw.get("context_window_token_budget"))
        if legacy is not None:
            return legacy
    return None


def load_compaction_settings(path: Path | None = None) -> CompactionSettings:
    settings_path = Path(path or DEFAULT_SETTINGS).expanduser()
    if not settings_path.exists():
        return CompactionSettings()
    try:
        data = json.loads(settings_path.read_text())
    except (OSError, json.JSONDecodeError):
        return CompactionSettings()
    if not isinstance(data, dict):
        return CompactionSettings()
    raw = data.get("compaction")
    if not isinstance(raw, dict):
        return CompactionSettings()
    model = raw.get("model")
    tertiary = raw.get("tertiary_fallback_slug")
    return CompactionSettings(
        model=str(model).strip() if isinstance(model, str) and model.strip() else None,
        override_current_model=_coerce_bool(raw.get("override_current_model"), False),
        fallback_enabled=_coerce_bool(raw.get("fallback_enabled"), True),
        tail_turns=max(0, _coerce_int(raw.get("tail_turns"), 2)),
        preserve_recent_tokens=max(0, _coerce_int(raw.get("preserve_recent_tokens"), 8000)),
        tool_output_max_chars=max(0, _coerce_int(raw.get("tool_output_max_chars"), 2000)),
        prune_tool_outputs=_coerce_bool(raw.get("prune_tool_outputs"), False),
        summary_max_output_tokens=max(256, _coerce_int(raw.get("summary_max_output_tokens"), 4096)),
        prompt_cache_key_version=str(raw.get("prompt_cache_key_version") or "v1"),
        prompt_cache_key_per_session=_coerce_bool(raw.get("prompt_cache_key_per_session"), False),
        tertiary_fallback_slug=(
            str(tertiary).strip() if isinstance(tertiary, str) and tertiary.strip() else None
        ),
        compaction_output_token_reserve=_load_output_token_reserve(raw),
        use_client_instructions_for_native=_coerce_bool(raw.get("use_client_instructions_for_native"), True),
    )


def compaction_prompt_cache_key(settings: CompactionSettings, session_key: str | None = None) -> str:
    base = f"codex-shim-compact:{settings.prompt_cache_key_version}"
    if settings.prompt_cache_key_per_session and session_key:
        return f"{base}:{session_key}"
    return base

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..settings import (
    ShimModel,
    is_chatgpt_passthrough_slug,
    load_chatgpt_passthrough_catalog_models,
)
from .config import CompactionSettings, effective_compaction_output_token_reserve

DEFAULT_COMPACTION_CATALOG_PATH = Path.home() / ".codex" / "custom_model_catalog.json"


def compaction_budget_slug(settings: CompactionSettings, requested_slug: str) -> str:
    configured = settings.model
    if configured and settings.override_current_model:
        return configured
    return requested_slug


def _context_from_catalog_entry(entry: dict[str, Any]) -> int | None:
    for key in ("max_context_window", "context_window"):
        value = entry.get(key)
        if isinstance(value, int) and value > 0:
            return value
    return None


def _load_desktop_catalog_models(catalog_path: Path | None = None) -> list[dict[str, Any]]:
    path = Path(catalog_path or DEFAULT_COMPACTION_CATALOG_PATH).expanduser()
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(data, dict):
        return []
    models = data.get("models")
    if not isinstance(models, list):
        return []
    return [dict(model) for model in models if isinstance(model, dict)]


def context_window_tokens_for_slug(
    slug: str,
    *,
    byok_models: list[ShimModel],
    catalog_path: Path | None = None,
) -> int | None:
    for entry in _load_desktop_catalog_models(catalog_path):
        if str(entry.get("slug") or "") == slug:
            context = _context_from_catalog_entry(entry)
            if context is not None:
                return context

    for entry in load_chatgpt_passthrough_catalog_models(catalog_path):
        if str(entry.get("slug") or "") == slug:
            context = _context_from_catalog_entry(entry)
            if context is not None:
                return context

    for model in byok_models:
        if model.slug == slug and model.max_context_limit and model.max_context_limit > 0:
            return model.max_context_limit

    if is_chatgpt_passthrough_slug(slug, catalog_path):
        return 400_000
    if slug.startswith("cursor-"):
        return 200_000
    return None


def compute_compaction_input_token_budget(
    compaction_model_context_window: int | None,
    settings: CompactionSettings,
) -> int | None:
    if compaction_model_context_window is None or compaction_model_context_window <= 0:
        return None
    reserve = effective_compaction_output_token_reserve(settings)
    return max(0, compaction_model_context_window - reserve)

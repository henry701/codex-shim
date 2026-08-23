from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import router as router_module
from .catalog_context import (
    CATALOG_TIER_BYOK,
    CATALOG_TIER_CHATGPT,
    CATALOG_TIER_CURSOR,
    CATALOG_TIER_ROUTER,
    CatalogContextSettings,
    apply_catalog_context_to_entry,
    load_catalog_context_settings,
)
from .catalog_slugs import CHATGPT_CATALOG_SLUG
from .upstream_compat import supports_parallel_tool_calls_in_catalog
from .settings import (
    OPENAI_PROVIDER_ID,
    ShimModel,
    available_model_slugs,
    chatgpt_passthrough_available,
    default_model_slug,
    load_chatgpt_passthrough_catalog_models,
    usable_byok_models,
)
from .cursor_passthrough import cursor_passthrough_available, cursor_passthrough_entries
from .discover import context_limit_for_discovered_model
from .naming import catalog_display_name


PLAN_TIERS = ["free", "plus", "pro", "team", "business", "enterprise"]

# Codex Desktop sorts picker entries by ascending priority, not JSON array order.
CATALOG_PRIORITY_BASE = {
    CATALOG_TIER_BYOK: 100,
    CATALOG_TIER_CHATGPT: 5_000,
    CATALOG_TIER_CURSOR: 10_000,
    CATALOG_TIER_ROUTER: 20_000,
}


def sort_catalog_entries(
    entries: list[dict[str, Any]],
    *,
    slug_key: str = "slug",
) -> list[dict[str, Any]]:
    return sorted(entries, key=lambda entry: str(entry.get(slug_key) or ""))


def assign_catalog_display_priorities(
    tiered_entries: list[tuple[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Assign ascending priorities within each catalog tier, sorted by slug."""
    by_tier: dict[str, list[dict[str, Any]]] = {
        CATALOG_TIER_BYOK: [],
        CATALOG_TIER_CHATGPT: [],
        CATALOG_TIER_CURSOR: [],
        CATALOG_TIER_ROUTER: [],
    }
    for tier, entry in tiered_entries:
        bucket = by_tier.get(tier)
        if bucket is None:
            raise ValueError(f"unknown catalog tier: {tier}")
        bucket.append(entry)

    ordered: list[dict[str, Any]] = []
    for tier in (
        CATALOG_TIER_BYOK,
        CATALOG_TIER_CHATGPT,
        CATALOG_TIER_CURSOR,
        CATALOG_TIER_ROUTER,
    ):
        base = CATALOG_PRIORITY_BASE[tier]
        for rank, entry in enumerate(sort_catalog_entries(by_tier[tier])):
            entry["priority"] = base + rank
            ordered.append(entry)
    return ordered


def _reasoning_catalog_fields(model: ShimModel) -> dict[str, Any]:
    if model.supports_reasoning_summaries:
        return {
            "default_reasoning_summary": "auto",
            "supports_reasoning_summaries": True,
        }
    return {
        "default_reasoning_summary": "none",
        "reasoning_summary_format": "none",
        "supports_reasoning_summaries": False,
    }


_REASONING_LEVEL_DESCRIPTIONS = {
    "minimal": "Minimal reasoning",
    "low": "Faster, lighter reasoning",
    "medium": "Balanced speed and reasoning",
    "high": "Deeper reasoning",
    "xhigh": "Maximum reasoning where supported",
    "max": "Maximum reasoning depth for the hardest problems",
    "ultra": "Maximum reasoning with automatic task delegation",
}
# Catalog JSON (`custom_model_catalog.json`) only — never discovery/raw.
# CLI and Desktop both parse model_catalog_json as ModelsResponse
# { models: Vec<ModelInfo> }. ModelInfo.supported_reasoning_levels is a
# required Vec<ReasoningEffortPreset> with no #[serde(default)]; omitting
# the key fails config.toml: missing field `supported_reasoning_levels`.
# Empty [] deserializes, but Desktop's effort picker wants real rows.
# When upstream listed no variants, write low/medium/high on this file.
_CATALOG_FILE_FALLBACK_REASONING_LEVELS = (
    {"effort": "low", "description": _REASONING_LEVEL_DESCRIPTIONS["low"]},
    {"effort": "medium", "description": _REASONING_LEVEL_DESCRIPTIONS["medium"]},
    {"effort": "high", "description": _REASONING_LEVEL_DESCRIPTIONS["high"]},
)
# Desktop serde for InputModality is a closed enum: text | image | audio.
# models.dev (and some ChatGPT copy-through rows) also emit `video` and other
# values. Writing those into custom_model_catalog.json makes Desktop fail:
#   failed to parse model_catalog_json ... unknown variant `video`,
#   expected one of `text`, `image`, `audio`
# Strip at every catalog write path, including ChatGPT passthrough finalize.
# Never copy models.dev modalities through verbatim.
_DESKTOP_INPUT_MODALITIES = ("text", "image", "audio")
_DESKTOP_INPUT_MODALITY_SET = frozenset(_DESKTOP_INPUT_MODALITIES)


def _model_raw(model: ShimModel) -> dict[str, Any]:
    raw = model.raw
    return raw if isinstance(raw, dict) else {}


def _supported_reasoning_levels(model: ShimModel) -> list[dict[str, str]]:
    # Upstream only. Catalog-file fallback is applied later; discovery/raw
    # must not invent low/medium/high for models with no effort variants.
    efforts = _model_raw(model).get("reasoning_efforts")
    if not isinstance(efforts, (list, tuple)) or not efforts:
        return []
    levels: list[dict[str, str]] = []
    seen: set[str] = set()
    for effort in efforts:
        name = str(effort or "").strip().lower()
        if not name or name in seen:
            continue
        seen.add(name)
        levels.append(
            {
                "effort": name,
                "description": _REASONING_LEVEL_DESCRIPTIONS.get(name, name),
            }
        )
    return levels


def _catalog_file_reasoning_levels(levels: Any) -> list[dict[str, str]]:
    if isinstance(levels, list) and levels:
        return levels
    return [dict(level) for level in _CATALOG_FILE_FALLBACK_REASONING_LEVELS]


def _catalog_file_default_reasoning_level(levels: list[dict[str, str]], existing: Any) -> str:
    # Catalog file contract: always write a default, and it must be one of the
    # listed efforts. CLI ModelInfo.default_reasoning_level is Option (missing
    # deserializes as None); Desktop's picker still needs a selected row.
    efforts = [
        str(level.get("effort") or "").strip().lower()
        for level in levels
        if isinstance(level, dict) and str(level.get("effort") or "").strip()
    ]
    current = str(existing or "").strip().lower()
    if current and current in efforts:
        return current
    for preferred in ("medium", "high", "low"):
        if preferred in efforts:
            return preferred
    return efforts[0] if efforts else "medium"


def _desktop_input_modalities(raw: Any) -> list[str]:
    items: list[str] = []
    seen: set[str] = set()
    if isinstance(raw, (list, tuple)):
        for item in raw:
            name = str(item or "").strip().lower()
            if name in _DESKTOP_INPUT_MODALITY_SET and name not in seen:
                seen.add(name)
                items.append(name)
    if "text" not in seen:
        items.insert(0, "text")
    return items


def _input_modalities(model: ShimModel) -> list[str]:
    raw = _model_raw(model).get("input_modalities")
    if isinstance(raw, (list, tuple)) and raw:
        return _desktop_input_modalities(raw)
    return ["text"] if model.no_image_support else ["text", "image"]


def _catalog_description(model: ShimModel, display_name: str) -> str:
    description = str(_model_raw(model).get("upstream_description") or "").strip()
    if description:
        return description
    return f"{display_name} via local Codex shim."


def _finalize_catalog_entry(entry: dict[str, Any], *, tier: str, context_settings: CatalogContextSettings | None) -> dict:
    finalized = apply_catalog_context_to_entry(entry, tier=tier, settings=context_settings)
    if "input_modalities" in finalized:
        finalized["input_modalities"] = _desktop_input_modalities(finalized.get("input_modalities"))
    finalized["supported_reasoning_levels"] = _catalog_file_reasoning_levels(
        finalized.get("supported_reasoning_levels")
    )
    finalized["default_reasoning_level"] = _catalog_file_default_reasoning_level(
        finalized["supported_reasoning_levels"],
        finalized.get("default_reasoning_level"),
    )
    return finalized


def catalog_entry(model: ShimModel, *, context_settings: CatalogContextSettings | None = None) -> dict:
    context = model.max_context_limit or _default_context(model)
    compact = max(8_000, int(context * 0.8))
    truncation = min(64_000, max(8_000, int(context * 0.32)))
    reasoning_levels = _catalog_file_reasoning_levels(_supported_reasoning_levels(model))
    display_name = catalog_display_name(model)
    modalities = _input_modalities(model)
    no_image = "image" not in modalities
    entry = {
        "slug": model.slug,
        "display_name": display_name,
        "description": _catalog_description(model, display_name),
        "context_window": context,
        "max_context_window": context,
        "auto_compact_token_limit": compact,
        "truncation_policy": {"mode": "tokens", "limit": truncation},
        **_reasoning_catalog_fields(model),
        "default_verbosity": "low",
        "support_verbosity": False,
        "apply_patch_tool_type": "freeform",
        "web_search_tool_type": "text_and_image",
        "supports_search_tool": True,
        "supports_parallel_tool_calls": supports_parallel_tool_calls_in_catalog(model),
        "experimental_supported_tools": [],
        "input_modalities": modalities,
        "supports_image_detail_original": not no_image,
        "shell_type": "shell_command",
        "visibility": "list",
        "minimal_client_version": "0.0.1",
        "supported_in_api": True,
        "availability_nux": None,
        "upgrade": None,
        "prefer_websockets": False,
        "available_in_plans": PLAN_TIERS,
        "base_instructions": "You are a coding agent running in Codex through a local BYOK shim.",
        "model_messages": {
            "instructions_template": (
                "You are Codex running on {model_name} through a local all-model shim. "
                "Be a helpful, direct coding collaborator."
            ),
            "instructions_variables": {"model_name": display_name},
        },
        "default_reasoning_level": _reasoning_effort(
            model, [level["effort"] for level in reasoning_levels]
        ),
        "supported_reasoning_levels": reasoning_levels,
    }
    return _finalize_catalog_entry(entry, tier=CATALOG_TIER_BYOK, context_settings=context_settings)


def chatgpt_passthrough_entries(*, context_settings: CatalogContextSettings | None = None) -> list[dict]:
    """Catalog entries for GPT models routed through ChatGPT passthrough."""
    entries: list[dict] = []
    for raw in load_chatgpt_passthrough_catalog_models():
        entry = {key: value for key, value in raw.items() if not str(key).startswith("_")}
        entry["visibility"] = "list"
        entry.setdefault("available_in_plans", PLAN_TIERS)
        entry.setdefault("minimal_client_version", "0.0.1")
        entry.setdefault("supported_in_api", True)
        entry["prefer_websockets"] = True
        if entry.get("slug") == CHATGPT_CATALOG_SLUG:
            entry["isDefault"] = True
        entries.append(
            _finalize_catalog_entry(entry, tier=CATALOG_TIER_CHATGPT, context_settings=context_settings)
        )
    return entries


def chatgpt_passthrough_entry() -> dict:
    """Catalog entry for the default GPT-5.5 ChatGPT passthrough model."""
    for entry in chatgpt_passthrough_entries():
        if entry.get("slug") == CHATGPT_CATALOG_SLUG:
            return entry
    return chatgpt_passthrough_entries()[0]


def write_catalog(
    models: list[ShimModel],
    path: Path,
    router_config=None,
    *,
    settings_data: dict[str, Any] | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    context_settings = load_catalog_context_settings(settings_data)
    tiered_entries: list[tuple[str, dict[str, Any]]] = []
    if router_config is not None and router_module.router_is_active(router_config, available_model_slugs(models)):
        tiered_entries.append(
            (
                CATALOG_TIER_ROUTER,
                _finalize_catalog_entry(
                    router_module.router_catalog_entry(router_config),
                    tier=CATALOG_TIER_ROUTER,
                    context_settings=context_settings,
                ),
            ),
        )
    if chatgpt_passthrough_available():
        tiered_entries.extend(
            (CATALOG_TIER_CHATGPT, entry)
            for entry in chatgpt_passthrough_entries(context_settings=context_settings)
        )
    if cursor_passthrough_available():
        cursor_entries = cursor_passthrough_entries()
        if cursor_entries and not chatgpt_passthrough_available():
            cursor_entries[0]["isDefault"] = True
        tiered_entries.extend(
            (
                CATALOG_TIER_CURSOR,
                _finalize_catalog_entry(entry, tier=CATALOG_TIER_CURSOR, context_settings=context_settings),
            )
            for entry in cursor_entries
        )
    tiered_entries.extend(
        (CATALOG_TIER_BYOK, catalog_entry(model, context_settings=context_settings))
        for model in usable_byok_models(models)
    )
    entries = assign_catalog_display_priorities(tiered_entries)
    _validate_catalog_file_models(entries)
    payload = {"models": sort_catalog_entries(entries)}
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")
    return path


def _validate_catalog_file_models(models: list[dict[str, Any]]) -> None:
    """Refuse to write a catalog Desktop/CLI serde will reject or that has a broken picker.

    CLI ``ModelInfo.default_reasoning_level`` is ``Option`` (missing is None).
    ``supported_reasoning_levels`` is a required ``Vec`` with no ``#[serde(default)]``.
    Empty ``[]`` parses, but Desktop's effort picker wants real rows, and a
    missing ``default_reasoning_level`` leaves the UI without a selection.
    Always emit both on this file; default must be one of the listed efforts.
    """
    for entry in models:
        slug = entry.get("slug") or "<missing-slug>"
        levels = entry.get("supported_reasoning_levels")
        if not isinstance(levels, list) or not levels:
            raise ValueError(f"{slug}: catalog supported_reasoning_levels must be a non-empty list")
        efforts: list[str] = []
        for level in levels:
            if not isinstance(level, dict):
                raise ValueError(f"{slug}: catalog reasoning level must be an object")
            effort = str(level.get("effort") or "").strip()
            if not effort or "description" not in level:
                raise ValueError(f"{slug}: catalog reasoning level needs effort and description")
            efforts.append(effort)
        default = entry.get("default_reasoning_level")
        if not default:
            raise ValueError(f"{slug}: catalog default_reasoning_level is required on this file")
        if str(default) not in efforts:
            raise ValueError(
                f"{slug}: catalog default_reasoning_level {default!r} is not in {efforts}"
            )
        modalities = entry.get("input_modalities")
        if isinstance(modalities, list):
            unknown = [item for item in modalities if item not in _DESKTOP_INPUT_MODALITY_SET]
            if unknown:
                raise ValueError(f"{slug}: catalog input_modalities not in Desktop enum: {unknown}")


def write_config(models: list[ShimModel], path: Path, catalog_path: Path, port: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        default_slug = default_model_slug(models)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    text = f'''# Generated by codex-shim. This file is opt-in and is not ~/.codex/config.toml.
model = "{_toml_escape(default_slug)}"
model_provider = "{OPENAI_PROVIDER_ID}"
openai_base_url = "http://127.0.0.1:{port}/v1"
model_catalog_json = "{_toml_escape(str(catalog_path))}"

[features]
tool_search_always_defer_mcp_tools = true
'''
    path.write_text(text)
    return path


def codex_config_overrides(catalog_path: Path, default_slug: str, port: int) -> list[str]:
    return [
        f'model="{_toml_escape(default_slug)}"',
        f'model_provider="{OPENAI_PROVIDER_ID}"',
        f'openai_base_url="http://127.0.0.1:{port}/v1"',
        f'model_catalog_json="{_toml_escape(str(catalog_path))}"',
        "features.tool_search_always_defer_mcp_tools=true",
    ]


def _default_context(model: ShimModel) -> int:
    inferred = context_limit_for_discovered_model(model.model) or context_limit_for_discovered_model(
        model.slug
    )
    if inferred:
        return inferred
    lower = f"{model.model} {model.display_name}".lower()
    if "claude" in lower:
        return 200_000
    if "gpt-5" in lower:
        return 400_000
    if "gemini" in lower:
        return 1_000_000
    return 128_000


def _guess_reasoning_effort(model: ShimModel) -> str:
    lower = model.display_name.lower()
    if "xhigh" in lower or "x-high" in lower:
        return "xhigh"
    if "high" in lower:
        return "high"
    if "medium" in lower:
        return "medium"
    if "low" in lower:
        return "low"
    return "medium"


def _reasoning_effort(model: ShimModel, levels: list[str] | None = None) -> str:
    available = [str(level).strip().lower() for level in (levels or []) if str(level).strip()]
    if not available:
        available = [item["effort"] for item in _supported_reasoning_levels(model)]
    guessed = _guess_reasoning_effort(model)
    if guessed in available:
        return guessed
    if "medium" in available:
        return "medium"
    if "high" in available:
        return "high"
    return available[0] if available else "medium"


def _toml_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')

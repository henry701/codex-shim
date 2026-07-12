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


def _finalize_catalog_entry(entry: dict[str, Any], *, tier: str, context_settings: CatalogContextSettings | None) -> dict:
    return apply_catalog_context_to_entry(entry, tier=tier, settings=context_settings)


def catalog_entry(model: ShimModel, *, context_settings: CatalogContextSettings | None = None) -> dict:
    context = model.max_context_limit or _default_context(model)
    compact = max(8_000, int(context * 0.8))
    truncation = min(64_000, max(8_000, int(context * 0.32)))
    reasoning = _reasoning_effort(model)
    display_name = catalog_display_name(model)
    entry = {
        "slug": model.slug,
        "display_name": display_name,
        "description": f"{display_name} via local Codex shim.",
        "context_window": context,
        "max_context_window": context,
        "auto_compact_token_limit": compact,
        "truncation_policy": {"mode": "tokens", "limit": truncation},
        "default_reasoning_level": reasoning,
        "supported_reasoning_levels": [
            {"effort": "low", "description": "Faster, lighter reasoning"},
            {"effort": "medium", "description": "Balanced speed and reasoning"},
            {"effort": "high", "description": "Deeper reasoning"},
            {"effort": "xhigh", "description": "Maximum reasoning where supported"},
        ],
        **_reasoning_catalog_fields(model),
        "default_verbosity": "low",
        "support_verbosity": False,
        "apply_patch_tool_type": "freeform",
        "web_search_tool_type": "text_and_image",
        "supports_search_tool": True,
        "supports_parallel_tool_calls": supports_parallel_tool_calls_in_catalog(model),
        "experimental_supported_tools": [],
        "input_modalities": ["text"] if model.no_image_support else ["text", "image"],
        "supports_image_detail_original": not model.no_image_support,
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
    payload = {"models": sort_catalog_entries(entries)}
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")
    return path


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
    lower = f"{model.model} {model.display_name}".lower()
    if "claude" in lower:
        return 200_000
    if "gpt-5" in lower:
        return 400_000
    if "gemini" in lower:
        return 1_000_000
    return 128_000


def _reasoning_effort(model: ShimModel) -> str:
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


def _toml_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')

"""Apply catalog context_window overrides from ~/.codex-shim/models.json.

Catalog tiers (``apply_to_tiers`` / discovery grouping)
-------------------------------------------------------
These tiers are assigned when ``write_catalog`` builds
``~/.codex/custom_model_catalog.json``. They are **not** OpenAI plan tiers.

+------------+-----------------------------------------------+---------------------+
| Tier       | Models                                        | Typical discovery   |
+============+===============================================+=====================+
| ``chatgpt``| ChatGPT subscription passthrough              | ``codex login``     |
|            | (``codex-gpt-*`` / backend ``/models`` cache)  | models cache        |
+------------+-----------------------------------------------+---------------------+
| ``cursor`` | Cursor subscription passthrough               | Cursor auth bridge  |
|            | (``cursor-*``). Context is usually controlled |                     |
|            | by cursor-agent itself — leave out of         |                     |
|            | ``apply_to_tiers`` unless you know you need it.|                     |
+------------+-----------------------------------------------+---------------------+
| ``byok``   | Local / OpenRouter / Zen / NVIDIA / etc.      | ``discover`` flags  |
|            | Prefer per-model ``max_context_limit`` in     | + ``models[]``      |
|            | ``models.json`` instead of the global modifier.|                     |
+------------+-----------------------------------------------+---------------------+
| ``router`` | Shim auto-router entry                        | ``router`` block    |
+------------+-----------------------------------------------+---------------------+

``modifier`` only applies to tiers listed in ``apply_to_tiers`` (default:
``["chatgpt"]``). Exact ``overrides`` and glob ``override_patterns`` still match
by slug across all tiers.

JSON configs may include ``$comment`` / ``$comments`` keys; they are ignored.
"""

from __future__ import annotations

import fnmatch
import math
from dataclasses import dataclass, field
from typing import Any, Mapping

# Catalog discovery tiers used by write_catalog / apply_to_tiers.
CATALOG_TIER_BYOK = "byok"
CATALOG_TIER_CHATGPT = "chatgpt"
CATALOG_TIER_CURSOR = "cursor"
CATALOG_TIER_ROUTER = "router"

ALL_CATALOG_TIERS: frozenset[str] = frozenset(
    {
        CATALOG_TIER_BYOK,
        CATALOG_TIER_CHATGPT,
        CATALOG_TIER_CURSOR,
        CATALOG_TIER_ROUTER,
    }
)

CATALOG_TIER_DESCRIPTIONS: dict[str, str] = {
    CATALOG_TIER_CHATGPT: (
        "ChatGPT subscription passthrough models (codex-gpt-*); from Codex backend /models cache."
    ),
    CATALOG_TIER_CURSOR: (
        "Cursor subscription passthrough; cursor-agent usually owns context — omit from apply_to_tiers."
    ),
    CATALOG_TIER_BYOK: (
        "Bring-your-own-key / local / discovered providers; prefer per-model max_context_limit."
    ),
    CATALOG_TIER_ROUTER: "Shim auto-router catalog entry.",
}

# Global modifier applies to ChatGPT passthrough by default. Cursor is omitted
# because cursor-agent enforces its own context limits.
DEFAULT_MODIFIER_TIERS = frozenset({CATALOG_TIER_CHATGPT})

_COMMENT_KEYS = frozenset({"$comment", "$comments", "comment", "comments"})


@dataclass(frozen=True)
class CatalogContextOverride:
    context_window: int | None = None
    max_context_window: int | None = None
    auto_compact_token_limit: int | None = None
    truncation_limit: int | None = None


@dataclass(frozen=True)
class CatalogContextSettings:
    modifier: float | None = None
    apply_to_tiers: frozenset[str] = DEFAULT_MODIFIER_TIERS
    overrides: dict[str, CatalogContextOverride] = field(default_factory=dict)
    override_patterns: dict[str, CatalogContextOverride] = field(default_factory=dict)


def load_catalog_context_settings(settings_data: Mapping[str, Any] | None) -> CatalogContextSettings | None:
    if not isinstance(settings_data, dict):
        return None
    raw = settings_data.get("catalog_context")
    if not isinstance(raw, dict):
        return None

    modifier = _positive_float(raw.get("modifier"))
    tiers_raw = raw.get("apply_to_tiers", raw.get("modifier_tiers"))
    apply_to_tiers = _parse_tiers(tiers_raw)

    overrides = _parse_override_map(raw.get("overrides"))
    patterns = _parse_override_map(raw.get("override_patterns", raw.get("patterns")))

    if modifier is None and not overrides and not patterns:
        return None
    return CatalogContextSettings(
        modifier=modifier,
        apply_to_tiers=apply_to_tiers,
        overrides=overrides,
        override_patterns=patterns,
    )


def apply_catalog_context_to_entry(
    entry: dict[str, Any],
    *,
    tier: str,
    settings: CatalogContextSettings | None,
) -> dict[str, Any]:
    if settings is None:
        return entry

    slug = str(entry.get("slug") or "")
    override = _resolve_override(slug, settings)
    context = _base_context(entry)
    if context is None:
        return entry

    if override is not None:
        if override.context_window is not None:
            context = override.context_window
        elif override.max_context_window is not None:
            context = override.max_context_window
    elif settings.modifier is not None and tier in settings.apply_to_tiers:
        context = max(1, int(math.floor(context * settings.modifier)))
    else:
        # No slug override and modifier does not apply to this tier.
        return entry

    return _write_context_fields(entry, context, override)


def known_catalog_tiers() -> frozenset[str]:
    """Return the closed set of catalog discovery tiers accepted by ``apply_to_tiers``."""
    return ALL_CATALOG_TIERS


def _resolve_override(slug: str, settings: CatalogContextSettings) -> CatalogContextOverride | None:
    if slug in settings.overrides:
        return settings.overrides[slug]
    for pattern, override in settings.override_patterns.items():
        if fnmatch.fnmatchcase(slug, pattern):
            return override
    return None


def _base_context(entry: Mapping[str, Any]) -> int | None:
    for key in ("context_window", "max_context_window"):
        value = entry.get(key)
        if isinstance(value, int) and value > 0:
            return value
    return None


def _write_context_fields(
    entry: dict[str, Any],
    context: int,
    override: CatalogContextOverride | None,
) -> dict[str, Any]:
    updated = dict(entry)
    updated["context_window"] = context
    updated["max_context_window"] = (
        override.max_context_window if override and override.max_context_window is not None else context
    )

    compact = override.auto_compact_token_limit if override and override.auto_compact_token_limit is not None else max(
        8_000, int(context * 0.8)
    )
    trunc = override.truncation_limit if override and override.truncation_limit is not None else min(
        64_000, max(8_000, int(context * 0.32))
    )
    updated["auto_compact_token_limit"] = compact
    policy = updated.get("truncation_policy")
    if isinstance(policy, dict):
        updated["truncation_policy"] = {**policy, "mode": policy.get("mode") or "tokens", "limit": trunc}
    else:
        updated["truncation_policy"] = {"mode": "tokens", "limit": trunc}
    return updated


def _parse_tiers(raw: Any) -> frozenset[str]:
    if raw is None:
        return DEFAULT_MODIFIER_TIERS
    if isinstance(raw, str):
        tiers = {raw.strip()} if raw.strip() else set()
    elif isinstance(raw, list):
        tiers = {str(item).strip() for item in raw if str(item).strip()}
    else:
        return DEFAULT_MODIFIER_TIERS
    if not tiers:
        return DEFAULT_MODIFIER_TIERS
    # Keep only known tiers so typos do not silently expand the modifier.
    known = tiers & ALL_CATALOG_TIERS
    return frozenset(known) if known else DEFAULT_MODIFIER_TIERS


def _parse_override_map(raw: Any) -> dict[str, CatalogContextOverride]:
    if not isinstance(raw, dict):
        return {}
    parsed: dict[str, CatalogContextOverride] = {}
    for key, value in raw.items():
        slug = str(key).strip()
        if not slug or slug in _COMMENT_KEYS or not isinstance(value, dict):
            continue
        override = _parse_override(value)
        if override is not None:
            parsed[slug] = override
    return parsed


def _parse_override(raw: dict[str, Any]) -> CatalogContextOverride | None:
    context = _positive_int(raw.get("context_window"))
    max_context = _positive_int(raw.get("max_context_window"))
    compact = _positive_int(raw.get("auto_compact_token_limit"))
    trunc = _positive_int(raw.get("truncation_limit", raw.get("truncation_policy_limit")))
    if context is None and max_context is None and compact is None and trunc is None:
        return None
    return CatalogContextOverride(
        context_window=context,
        max_context_window=max_context,
        auto_compact_token_limit=compact,
        truncation_limit=trunc,
    )


def _positive_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _positive_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None

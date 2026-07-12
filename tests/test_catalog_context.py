from __future__ import annotations

import json
from pathlib import Path

from codex_shim.catalog import write_catalog
from codex_shim.catalog_context import (
    ALL_CATALOG_TIERS,
    CATALOG_TIER_BYOK,
    CATALOG_TIER_CHATGPT,
    CATALOG_TIER_CURSOR,
    CATALOG_TIER_DESCRIPTIONS,
    CATALOG_TIER_ROUTER,
    CatalogContextOverride,
    CatalogContextSettings,
    DEFAULT_MODIFIER_TIERS,
    apply_catalog_context_to_entry,
    known_catalog_tiers,
    load_catalog_context_settings,
)
from codex_shim.settings import ShimModel


def test_known_catalog_tiers_enum_is_closed_and_documented():
    assert known_catalog_tiers() == ALL_CATALOG_TIERS
    assert ALL_CATALOG_TIERS == {
        CATALOG_TIER_BYOK,
        CATALOG_TIER_CHATGPT,
        CATALOG_TIER_CURSOR,
        CATALOG_TIER_ROUTER,
    }
    assert set(CATALOG_TIER_DESCRIPTIONS) == ALL_CATALOG_TIERS
    assert DEFAULT_MODIFIER_TIERS == frozenset({CATALOG_TIER_CHATGPT})


def test_load_catalog_context_settings_reads_modifier_and_overrides():
    settings = load_catalog_context_settings(
        {
            "catalog_context": {
                "$comment": "docs only",
                "modifier": 0.9,
                "apply_to_tiers": ["chatgpt"],
                "overrides": {"codex-gpt-5-6-terra": {"context_window": 240000}},
                "override_patterns": {"codex-gpt-5-6-*": {"context_window": 200000}},
            }
        }
    )
    assert settings is not None
    assert settings.modifier == 0.9
    assert settings.apply_to_tiers == frozenset({CATALOG_TIER_CHATGPT})
    assert settings.overrides["codex-gpt-5-6-terra"].context_window == 240000
    assert "$comment" not in settings.overrides


def test_load_ignores_comment_keys_in_override_maps():
    settings = load_catalog_context_settings(
        {
            "catalog_context": {
                "modifier": 0.9,
                "overrides": {
                    "$comment": {"context_window": 1},
                    "codex-gpt-5-6-terra": {"context_window": 240000, "$comment": "ignored"},
                },
            }
        }
    )
    assert settings is not None
    assert set(settings.overrides) == {"codex-gpt-5-6-terra"}


def test_load_defaults_apply_to_tiers_to_chatgpt_only():
    settings = load_catalog_context_settings({"catalog_context": {"modifier": 0.9}})
    assert settings is not None
    assert settings.apply_to_tiers == frozenset({CATALOG_TIER_CHATGPT})


def test_load_drops_unknown_tiers_and_falls_back_when_empty():
    settings = load_catalog_context_settings(
        {"catalog_context": {"modifier": 0.9, "apply_to_tiers": ["chatgpt", "not-a-tier"]}}
    )
    assert settings is not None
    assert settings.apply_to_tiers == frozenset({CATALOG_TIER_CHATGPT})

    fallback = load_catalog_context_settings(
        {"catalog_context": {"modifier": 0.9, "apply_to_tiers": ["nope", "also-nope"]}}
    )
    assert fallback is not None
    assert fallback.apply_to_tiers == DEFAULT_MODIFIER_TIERS


def test_exact_override_wins_over_pattern_and_modifier():
    settings = CatalogContextSettings(
        modifier=0.9,
        overrides={"codex-gpt-5-6-terra": CatalogContextOverride(context_window=240000)},
        override_patterns={"codex-gpt-5-6-*": CatalogContextOverride(context_window=200000)},
    )
    entry = {
        "slug": "codex-gpt-5-6-terra",
        "context_window": 372000,
        "max_context_window": 372000,
        "truncation_policy": {"mode": "tokens", "limit": 10000},
    }
    updated = apply_catalog_context_to_entry(entry, tier="chatgpt", settings=settings)
    assert updated["context_window"] == 240000
    assert updated["max_context_window"] == 240000
    assert updated["auto_compact_token_limit"] == 192000


def test_modifier_applies_to_chatgpt_tier_only_by_default():
    settings = load_catalog_context_settings({"catalog_context": {"modifier": 0.9}})
    assert settings is not None
    entry = {"slug": "codex-gpt-5-5", "context_window": 400000, "max_context_window": 400000}
    chatgpt = apply_catalog_context_to_entry(entry, tier=CATALOG_TIER_CHATGPT, settings=settings)
    cursor = apply_catalog_context_to_entry(entry, tier=CATALOG_TIER_CURSOR, settings=settings)
    byok = apply_catalog_context_to_entry(entry, tier=CATALOG_TIER_BYOK, settings=settings)
    router = apply_catalog_context_to_entry(entry, tier=CATALOG_TIER_ROUTER, settings=settings)
    assert chatgpt["context_window"] == 360000
    assert chatgpt["auto_compact_token_limit"] == 288000
    assert cursor["context_window"] == 400000
    assert byok["context_window"] == 400000
    assert router["context_window"] == 400000
    # Unmodified tiers keep original truncation / compact fields unset.
    assert "auto_compact_token_limit" not in cursor


def test_modifier_can_opt_into_cursor_tier_explicitly():
    settings = load_catalog_context_settings(
        {"catalog_context": {"modifier": 0.5, "apply_to_tiers": ["cursor"]}}
    )
    assert settings is not None
    entry = {"slug": "cursor-composer", "context_window": 200000, "max_context_window": 200000}
    cursor = apply_catalog_context_to_entry(entry, tier=CATALOG_TIER_CURSOR, settings=settings)
    chatgpt = apply_catalog_context_to_entry(entry, tier=CATALOG_TIER_CHATGPT, settings=settings)
    assert cursor["context_window"] == 100000
    assert chatgpt["context_window"] == 200000


def test_pattern_override_applies_across_tiers():
    settings = CatalogContextSettings(
        modifier=0.9,
        apply_to_tiers=frozenset({CATALOG_TIER_CHATGPT}),
        override_patterns={"local-*": CatalogContextOverride(context_window=50000)},
    )
    entry = {"slug": "local-llama", "context_window": 131072, "max_context_window": 131072}
    updated = apply_catalog_context_to_entry(entry, tier=CATALOG_TIER_BYOK, settings=settings)
    assert updated["context_window"] == 50000


def test_modifier_can_opt_into_router_tier_via_write_path():
    settings = load_catalog_context_settings(
        {"catalog_context": {"modifier": 0.5, "apply_to_tiers": ["router"]}}
    )
    assert settings is not None
    entry = {"slug": "auto", "context_window": 400000, "max_context_window": 400000}
    updated = apply_catalog_context_to_entry(entry, tier=CATALOG_TIER_ROUTER, settings=settings)
    assert updated["context_window"] == 200000


def test_write_catalog_applies_chatgpt_modifier_not_byok(tmp_path, monkeypatch):
    cache = tmp_path / "models-cache.json"
    cache.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "slug": "gpt-5.6-terra",
                        "display_name": "GPT-5.6 Terra",
                        "context_window": 372000,
                        "max_context_window": 372000,
                        "visibility": "list",
                    }
                ]
            }
        )
    )
    monkeypatch.setattr("codex_shim.settings.DEFAULT_CODEX_MODELS_CACHE", cache)
    monkeypatch.setattr("codex_shim.catalog.chatgpt_passthrough_available", lambda: True)
    monkeypatch.setattr("codex_shim.catalog.cursor_passthrough_available", lambda: False)

    byok = ShimModel(
        model="gemma",
        display_name="Local",
        slug="local-llama",
        provider="generic-chat-completion-api",
        base_url="http://127.0.0.1:28000/v1",
        api_key="local",
        max_context_limit=131072,
    )
    catalog_path = tmp_path / "catalog.json"
    write_catalog(
        [byok],
        catalog_path,
        settings_data={"catalog_context": {"modifier": 0.9, "apply_to_tiers": ["chatgpt"]}},
    )
    data = json.loads(catalog_path.read_text())
    by_slug = {entry["slug"]: entry for entry in data["models"]}
    assert by_slug["codex-gpt-5-6-terra"]["context_window"] == 334800
    assert by_slug["codex-gpt-5-6-terra"]["max_context_window"] == 334800
    assert by_slug["codex-gpt-5-6-terra"]["auto_compact_token_limit"] == 267840
    assert by_slug["local-llama"]["context_window"] == 131072

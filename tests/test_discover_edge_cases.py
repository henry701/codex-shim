from __future__ import annotations

import json
from pathlib import Path

import pytest

from codex_shim.catalog import write_catalog
from codex_shim.discover import (
    LocalModelRecord,
    _catalog_slug_for_model,
    _enrich_builtin_template,
    _resolved_api_key,
    discover_byok_models,
    discover_enabled,
    fetch_openrouter_free_model_ids,
    is_local_base_url,
    merge_discovered_models,
    refresh_local_explicit_models,
    OPENROUTER_FREE_TEMPLATE,
)
from codex_shim.settings import ModelSettings, ShimModel, load_chatgpt_passthrough_catalog_models

pytestmark = pytest.mark.enable_model_discovery


@pytest.fixture
def auth_missing(monkeypatch, tmp_path):
    monkeypatch.setattr("codex_shim.settings.DEFAULT_CODEX_AUTH", tmp_path / "missing-auth.json")


def _local_model(**overrides) -> ShimModel:
    base = dict(
        slug="local-llama",
        model="stale.gguf",
        display_name="Stale",
        provider="generic-chat-completion-api",
        base_url="http://127.0.0.1:28000/v1",
        api_key="local",
        raw={"discover": "local"},
    )
    base.update(overrides)
    return ShimModel(**base)


def test_discover_disabled_globally_skips_builtin_templates(monkeypatch):
    monkeypatch.setattr(
        "codex_shim.discover.discover_opencode_cli_ids",
        lambda prefix: ["big-pickle"] if prefix == "opencode" else ["openrouter/free"],
    )
    models = discover_byok_models([], settings_data={"discover": False})
    assert models == []


def test_discover_openrouter_alias_respects_openrouter_key(monkeypatch):
    monkeypatch.setattr(
        "codex_shim.discover.discover_opencode_cli_ids",
        lambda prefix: ["openrouter/free"] if prefix == "openrouter" else [],
    )
    enabled = discover_byok_models([], settings_data={"discover": {"openrouter": True}})
    disabled = discover_byok_models([], settings_data={"discover": {"openrouter": False}})
    assert any(model.slug == "or-openrouter-free" for model in enabled)
    assert not any(model.slug == "or-openrouter-free" for model in disabled)


def test_paid_zen_is_not_auto_discovered(monkeypatch):
    monkeypatch.setattr(
        "codex_shim.discover.fetch_zen_model_ids",
        lambda: ["kimi-k2.6", "deepseek-v4-flash", "minimax-m3-free"],
    )
    monkeypatch.setattr(
        "codex_shim.discover.discover_opencode_cli_ids",
        lambda prefix: ["big-pickle", "kimi-k2.6"] if prefix == "opencode" else [],
    )
    explicit = [
        ShimModel(
            slug="zen-kimi-k2-6",
            model="kimi-k2.6",
            display_name="Zen Kimi",
            provider="generic-chat-completion-api",
            base_url="https://opencode.ai/zen/v1",
            api_key="sk-opencode-test",
        )
    ]
    models = discover_byok_models(explicit)
    slugs = {model.slug for model in models}
    assert "zen-kimi-k2-6" in slugs
    assert "zen-deepseek-v4-flash" not in slugs
    assert "oc-free-minimax-m3-free" in slugs
    assert "oc-free-big-pickle" in slugs
    assert sum(model.model == "kimi-k2.6" for model in models) == 1


def test_local_refresh_keeps_explicit_when_endpoint_unreachable(monkeypatch):
    monkeypatch.setattr("codex_shim.discover.fetch_local_openai_models", lambda *_a, **_k: [])
    [kept] = refresh_local_explicit_models([_local_model()])
    assert kept.model == "stale.gguf"
    assert kept.display_name == "Stale"
    assert "discovered_local" not in kept.raw


def test_local_refresh_skipped_when_discover_false():
    model = _local_model(raw={"discover": False})
    [kept] = refresh_local_explicit_models([model])
    assert kept.model == "stale.gguf"


def test_local_refresh_expands_multiple_models(monkeypatch):
    monkeypatch.setattr(
        "codex_shim.discover.fetch_local_openai_models",
        lambda *_a, **_k: [
            LocalModelRecord("alpha.gguf", 8192),
            LocalModelRecord("beta.gguf", 16384),
        ],
    )
    refreshed = refresh_local_explicit_models([_local_model()])
    assert [model.slug for model in refreshed] == ["local-llama", "local-llama-beta-gguf"]
    assert refreshed[0].model == "alpha.gguf"
    assert refreshed[1].max_context_limit == 16384


def test_is_local_base_url_private_ranges():
    assert is_local_base_url("http://127.0.0.1:28000/v1")
    assert is_local_base_url("http://localhost:11434/v1")
    assert is_local_base_url("http://10.0.0.5:8080/v1")
    assert not is_local_base_url("https://api.openai.com/v1")


def test_catalog_slug_collision_gets_numeric_suffix():
    used: set[str] = set()
    first = _catalog_slug_for_model("vendor/model-a", "or", used, 0)
    second = _catalog_slug_for_model("vendor/model-a", "or", used, 1)
    assert first == "or-vendor-model-a"
    assert second == "or-vendor-model-a-1"


def test_openrouter_free_fallback_when_cli_empty(monkeypatch):
    monkeypatch.setattr("codex_shim.discover.discover_opencode_cli_ids", lambda *_a, **_k: [])
    assert fetch_openrouter_free_model_ids() == ["openrouter/free"]


def test_enrich_builtin_template_uses_explicit_openrouter_credentials():
    explicit = [
        ShimModel(
            slug="or-free-router",
            model="openrouter/free",
            display_name="OR Free",
            provider="generic-chat-completion-api",
            base_url="https://openrouter.ai/api/v1",
            api_key="or-key-from-config",
            extra_headers={"X-Custom": "1"},
        )
    ]
    enriched = _enrich_builtin_template(OPENROUTER_FREE_TEMPLATE, explicit)
    assert enriched.api_key == "or-key-from-config"
    assert enriched.extra_headers["X-Custom"] == "1"
    assert enriched.extra_headers["HTTP-Referer"] == "https://opencode.ai/"


def test_resolved_api_key_expands_env_placeholder(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "resolved-key")
    assert _resolved_api_key("${OPENROUTER_API_KEY}") == "resolved-key"
    assert _resolved_api_key("literal") == "literal"


def test_chatgpt_catalog_uses_codex_prefix_and_prefers_cache(monkeypatch, tmp_path):
    cache = tmp_path / "models_cache.json"
    cache.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "slug": "gpt-5.5",
                        "display_name": "GPT-5.5 Cached",
                        "context_window": 272000,
                    }
                ]
            }
        )
    )
    monkeypatch.setattr(
        "codex_shim.discover.discover_chatgpt_model_ids_from_openai_api",
        lambda: ["gpt-5.5", "gpt-5.2"],
    )
    entries = load_chatgpt_passthrough_catalog_models(cache)
    by_slug = {entry["slug"]: entry for entry in entries}
    assert by_slug["codex-gpt-5-5"]["display_name"] == "GPT-5.5 Cached"
    assert by_slug["codex-gpt-5-5"]["_upstream_model"] == "gpt-5.5"
    assert by_slug["codex-gpt-5-2"]["_upstream_model"] == "gpt-5.2"


def test_write_catalog_includes_discovered_zen_public(tmp_path, monkeypatch, auth_missing):
    monkeypatch.setattr(
        "codex_shim.discover.fetch_zen_model_ids",
        lambda: ["minimax-m3-free"],
    )
    monkeypatch.setattr("codex_shim.discover.discover_opencode_cli_ids", lambda *_a, **_k: [])
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"models": []}))
    models = ModelSettings(settings).load()
    catalog_path = tmp_path / "catalog.json"
    write_catalog(models, catalog_path)
    payload = json.loads(catalog_path.read_text())
    slugs = {entry["slug"] for entry in payload["models"]}
    assert "oc-free-minimax-m3-free" in slugs


def test_model_settings_load_explicit_skips_discovery(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "codex_shim.discover.discover_opencode_cli_ids",
        lambda prefix: ["openrouter/free"] if prefix == "openrouter" else [],
    )
    settings = tmp_path / "settings.json"
    settings.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "model": "gpt-5.5",
                        "display_name": "Only One",
                        "provider": "openai",
                        "base_url": "http://127.0.0.1:9999/v1",
                        "api_key": "x",
                    }
                ]
            }
        )
    )
    explicit = ModelSettings(settings).load_explicit()
    loaded = ModelSettings(settings).load()
    assert [model.slug for model in explicit] == ["gpt-5-5"]
    assert len(loaded) > len(explicit)


def test_merge_skips_discovered_slug_already_used_by_different_model():
    explicit = [
        ShimModel(
            slug="or-custom",
            model="openrouter/free",
            display_name="Custom",
            provider="generic-chat-completion-api",
            base_url="https://openrouter.ai/api/v1",
            api_key="k",
        )
    ]
    discovered = [
        ShimModel(
            slug="or-openrouter-free",
            model="openrouter/free",
            display_name="Dup model id",
            provider="generic-chat-completion-api",
            base_url="https://openrouter.ai/api/v1",
            api_key="k",
            raw={"discovered": True},
        )
    ]
    merged = merge_discovered_models(explicit, discovered)
    assert len(merged) == 1
    assert merged[0].slug == "or-custom"

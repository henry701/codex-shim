from __future__ import annotations

import json

import pytest

from codex_shim.discover import (
    LocalModelRecord,
    _catalog_slug_for_model,
    _parse_models_dev_opencode_free_ids,
    _parse_models_dev_opencode_paid_ids,
    discover_byok_models,
    discover_enabled,
    fetch_nvidia_integrate_model_ids,
    fetch_openrouter_free_model_ids,
    fetch_zen_model_ids,
    fetch_zen_paid_model_ids,
    fetch_zen_public_model_ids,
    is_openrouter_free_model,
    is_zen_public_model,
    merge_discovered_models,
    refresh_local_explicit_models,
)
from codex_shim.settings import ShimModel


def _zen_template() -> ShimModel:
    return ShimModel(
        slug="zen-big-pickle",
        model="big-pickle",
        display_name="OpenCode Zen — Big Pickle (free)",
        provider="generic-chat-completion-api",
        base_url="https://opencode.ai/zen/v1",
        api_key="public",
    )


pytestmark = pytest.mark.enable_model_discovery  # noqa: PT023


def test_is_zen_public_model():
    assert is_zen_public_model("big-pickle")
    assert is_zen_public_model("minimax-m3-free")
    assert not is_zen_public_model("kimi-k2.6")


def test_discover_enabled_defaults_true():
    assert discover_enabled(None, "zen", has_template=True) is True
    assert discover_enabled({"discover": {"zen": False}}, "zen", has_template=True) is False
    assert discover_enabled({"discover": False}, "zen", has_template=True) is False
    assert discover_enabled({"discover": {"zen_paid": False}}, "zen", has_template=True) is False


def test_merge_discovered_models_keeps_explicit_entries():
    explicit = [_zen_template()]
    discovered = [
        ShimModel(
            slug="zen-minimax-m2-5",
            model="minimax-m2.5",
            display_name="OpenCode Zen — MiniMax M2.5",
            provider="generic-chat-completion-api",
            base_url="https://opencode.ai/zen/v1",
            api_key="public",
            raw={"discovered": True},
        )
    ]
    merged = merge_discovered_models(explicit, discovered)
    assert [model.slug for model in merged] == ["zen-big-pickle", "zen-minimax-m2-5"]


def test_merge_discovered_models_sorts_by_slug():
    explicit = [
        ShimModel(
            slug="zulu",
            model="zulu",
            display_name="Zulu",
            provider="openai",
            base_url="http://example.invalid/v1",
            api_key="k",
        )
    ]
    discovered = [
        ShimModel(
            slug="alpha",
            model="alpha",
            display_name="Alpha",
            provider="openai",
            base_url="http://example.invalid/v1",
            api_key="k",
            raw={"discovered": True},
        )
    ]
    merged = merge_discovered_models(explicit, discovered)
    assert [model.slug for model in merged] == ["alpha", "zulu"]


def test_openrouter_free_router_uses_stable_slug():
    slug = _catalog_slug_for_model("openrouter/free", "or", set(), 0)
    assert slug == "or-free-router"


def test_discover_byok_models_adds_zen_public_models(monkeypatch):
    monkeypatch.setattr(
        "codex_shim.discover.fetch_models_dev_opencode_free_model_ids",
        lambda: ["big-pickle", "deepseek-v4-flash-free"],
    )
    models = discover_byok_models([_zen_template()])
    slugs = {model.slug for model in models}
    assert "zen-big-pickle" in slugs
    assert "oc-free-deepseek-v4-flash-free" in slugs
    assert "oc-free-minimax-m3-free" not in slugs
    assert "zen-kimi-k2-6" not in slugs


def test_parse_models_dev_opencode_paid_ids_skips_free_and_deprecated():
    payload = {
        "opencode": {
            "models": {
                "big-pickle": {
                    "id": "big-pickle",
                    "status": "active",
                    "cost": {"input": 0, "output": 0},
                },
                "kimi-k2.6": {
                    "id": "kimi-k2.6",
                    "status": "active",
                    "cost": {"input": 0.3, "output": 1.2},
                },
                "minimax-m3": {
                    "id": "minimax-m3",
                    "status": "deprecated",
                    "cost": {"input": 0.5, "output": 1.0},
                },
            }
        }
    }
    assert _parse_models_dev_opencode_paid_ids(payload) == ["kimi-k2.6"]


def test_fetch_zen_paid_model_ids_requires_api_key(monkeypatch):
    monkeypatch.setattr(
        "codex_shim.discover.fetch_models_dev_opencode_paid_model_ids",
        lambda: ["kimi-k2.6"],
    )
    assert fetch_zen_paid_model_ids(api_key="") == ["kimi-k2.6"]
    monkeypatch.setattr("codex_shim.discover.fetch_models_dev_opencode_paid_model_ids", lambda: [])
    monkeypatch.setattr(
        "codex_shim.discover.fetch_zen_model_ids",
        lambda api_key="": ["kimi-k2.6"] if api_key else [],
    )
    assert fetch_zen_paid_model_ids(api_key="sk-test") == ["kimi-k2.6"]


def test_parse_models_dev_opencode_free_ids_skips_deprecated_and_paid(monkeypatch):
    payload = {
        "opencode": {
            "models": {
                "big-pickle": {
                    "id": "big-pickle",
                    "status": "active",
                    "cost": {"input": 0, "output": 0},
                },
                "minimax-m3-free": {
                    "id": "minimax-m3-free",
                    "status": "deprecated",
                    "cost": {"input": 0, "output": 0},
                },
                "kimi-k2.6": {
                    "id": "kimi-k2.6",
                    "status": "active",
                    "cost": {"input": 0.3, "output": 1.2},
                },
            }
        }
    }
    assert _parse_models_dev_opencode_free_ids(payload) == ["big-pickle"]


def test_fetch_zen_public_model_ids_falls_back_to_opencode_cli(monkeypatch):
    monkeypatch.setattr("codex_shim.discover.fetch_models_dev_opencode_free_model_ids", lambda: [])
    monkeypatch.setattr(
        "codex_shim.discover.discover_opencode_cli_ids",
        lambda prefix: ["big-pickle", "mimo-v2.5-free", "kimi-k2.6"] if prefix == "opencode" else [],
    )
    assert fetch_zen_public_model_ids() == ["big-pickle", "mimo-v2.5-free"]


def test_refresh_local_explicit_models_uses_endpoint_name(monkeypatch):
    local = ShimModel(
        slug="local-llama",
        model="stale-name.gguf",
        display_name="Local Gemma 4 (llama.cpp)",
        provider="generic-chat-completion-api",
        base_url="http://127.0.0.1:28000/v1",
        api_key="local",
        raw={"discover": "local"},
    )
    monkeypatch.setattr(
        "codex_shim.discover.fetch_local_openai_models",
        lambda *_args, **_kwargs: [LocalModelRecord("qwen3-8b-q4.gguf", 65536)],
    )
    [refreshed] = refresh_local_explicit_models([local])
    assert refreshed.model == "qwen3-8b-q4.gguf"
    assert "Gemma" not in refreshed.display_name
    assert refreshed.display_name.startswith("Local —")
    assert refreshed.max_context_limit == 65536


def test_is_openrouter_free_model():
    assert is_openrouter_free_model("openrouter/free")
    assert is_openrouter_free_model("meta-llama/llama-3.3-70b-instruct:free")
    assert not is_openrouter_free_model("anthropic/claude-3.5-sonnet")


def test_fetch_openrouter_free_model_ids(monkeypatch):
    monkeypatch.setattr(
        "codex_shim.discover.discover_opencode_cli_ids",
        lambda prefix: [
            "openrouter/free",
            "meta-llama/llama-3.3-70b-instruct:free",
            "anthropic/claude-3.5-sonnet",
        ]
        if prefix == "openrouter"
        else [],
    )
    assert fetch_openrouter_free_model_ids() == [
        "meta-llama/llama-3.3-70b-instruct:free",
        "openrouter/free",
    ]


def test_fetch_nvidia_integrate_model_ids(monkeypatch):
    monkeypatch.setattr(
        "codex_shim.discover.discover_opencode_cli_ids",
        lambda prefix: [
            "meta/llama-3.3-70b-instruct",
            "black-forest-labs/flux_1-schnell",
            "nemotron-3-super-120b-a12b",
        ]
        if prefix == "nvidia"
        else [],
    )
    ids = fetch_nvidia_integrate_model_ids()
    assert "meta/llama-3.3-70b-instruct" in ids
    assert "nemotron-3-super-120b-a12b" in ids
    assert "black-forest-labs/flux_1-schnell" not in ids


def test_discover_byok_models_adds_openrouter_free_and_nvidia(monkeypatch):
    monkeypatch.setattr("codex_shim.discover.fetch_models_dev_opencode_free_model_ids", lambda: [])
    monkeypatch.setattr(
        "codex_shim.discover.discover_opencode_cli_ids",
        lambda prefix: {
            "openrouter": ["openrouter/free", "qwen/qwen3-coder:free"],
            "nvidia": ["meta/llama-3.3-70b-instruct"],
        }.get(prefix, []),
    )
    models = discover_byok_models([])
    slugs = {model.slug for model in models}
    assert "or-free-router" in slugs
    assert "or-qwen-qwen3-coder-free" in slugs
    assert "nvidia-meta-llama-3-3-70b-instruct" in slugs


def test_fetch_zen_model_ids_parses_openai_style_payload(monkeypatch):
    payload = json.dumps({"data": [{"id": "minimax-m2.5"}, {"id": "kimi-k2.6"}]}).encode()

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return payload

    monkeypatch.setattr("codex_shim.discover.urlopen", lambda *_args, **_kwargs: FakeResponse())
    assert fetch_zen_model_ids() == ["minimax-m2.5", "kimi-k2.6"]

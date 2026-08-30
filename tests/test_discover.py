from __future__ import annotations

import json

import pytest

from codex_shim.discover import (
    LocalModelRecord,
    NOUS_PORTAL_TEMPLATE,
    NVIDIA_INTEGRATE_TEMPLATE,
    OPENROUTER_FREE_TEMPLATE,
    ZEN_PUBLIC_TEMPLATE,
    _catalog_slug_for_model,
    _parse_models_dev_opencode_free_ids,
    _parse_models_dev_opencode_paid_ids,
    _rows_to_shim_models,
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
from codex_shim.settings import ShimModel, byok_model_has_credentials


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


def test_discovered_zen_public_is_keyless_with_hermes_attribution(monkeypatch):
    monkeypatch.setattr(
        "codex_shim.discover.fetch_models_dev_opencode_free_model_ids",
        lambda: ["laguna-s-2.1-free"],
    )
    models = discover_byok_models([])
    route = next(model for model in models if model.model == "laguna-s-2.1-free")
    assert route.slug == "oc-free-laguna-s-2-1-free"
    assert route.api_key == "public"
    assert byok_model_has_credentials(route)
    assert route.extra_headers["HTTP-Referer"] == "https://hermes-agent.nousresearch.com"
    assert route.extra_headers["X-Title"] == "Hermes Agent"
    assert "User-Agent" not in route.extra_headers
    assert "Authorization" not in route.extra_headers


def test_zen_public_template_has_hermes_attribution_without_user_agent():
    assert ZEN_PUBLIC_TEMPLATE.api_key == "public"
    assert ZEN_PUBLIC_TEMPLATE.extra_headers["HTTP-Referer"] == "https://hermes-agent.nousresearch.com"
    assert ZEN_PUBLIC_TEMPLATE.extra_headers["X-Title"] == "Hermes Agent"
    assert "User-Agent" not in ZEN_PUBLIC_TEMPLATE.extra_headers
    assert "Authorization" not in ZEN_PUBLIC_TEMPLATE.extra_headers


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


def test_context_limit_for_ox_alpha_aliases_is_1m_not_128k():
    from codex_shim.discover import OX_ALPHA_CONTEXT_TOKENS, context_limit_for_discovered_model

    assert OX_ALPHA_CONTEXT_TOKENS == 1_048_576
    assert context_limit_for_discovered_model("stealth/ox-alpha") == 1_048_576
    assert context_limit_for_discovered_model("x-preview-f-free") == 1_048_576
    assert context_limit_for_discovered_model("big-pickle") is None


def test_discovered_nous_ox_alpha_carries_1m_context():
    models = _rows_to_shim_models(["stealth/ox-alpha"], NOUS_PORTAL_TEMPLATE)
    route = models[0]
    assert route.slug == "nous-stealth-ox-alpha"
    assert route.max_context_limit == 1_048_576


def test_discovered_opencode_ox_preview_carries_1m_context():
    models = _rows_to_shim_models(["x-preview-f-free"], ZEN_PUBLIC_TEMPLATE)
    route = models[0]
    assert route.slug == "oc-free-x-preview-f-free"
    assert route.max_context_limit == 1_048_576


def test_parse_models_dev_opencode_free_ids_keeps_explicit_limit_context():
    payload = {
        "opencode": {
            "models": {
                "x-preview-f-free": {
                    "id": "x-preview-f-free",
                    "status": "active",
                    "cost": {"input": 0, "output": 0},
                    "limit": {"context": 1_000_000, "output": 131_072},
                }
            }
        }
    }
    assert _parse_models_dev_opencode_free_ids(payload) == ["x-preview-f-free"]
    from codex_shim.discover import context_limit_for_discovered_model

    assert context_limit_for_discovered_model("x-preview-f-free") == 1_048_576


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


@pytest.mark.enable_model_discovery
def test_fetch_models_dev_catalog_keeps_stale_payload_when_refresh_fails(monkeypatch):
    from urllib.error import URLError

    from codex_shim import discover

    discover.clear_models_dev_catalog_cache()
    monkeypatch.setattr(
        discover,
        "fetch_http_json",
        lambda *_args, **_kwargs: {"opencode": {"models": {"hy3-free": {"id": "hy3-free"}}}},
    )
    first = discover.fetch_models_dev_catalog()
    assert first["opencode"]["models"]["hy3-free"]["id"] == "hy3-free"

    def boom(*_args, **_kwargs):
        raise URLError("models.dev down")

    monkeypatch.setattr(discover, "fetch_http_json", boom)
    discover.expire_models_dev_catalog_cache()
    assert discover.fetch_models_dev_catalog() == first


def test_list_opencode_cli_models_keeps_stale_payload_when_refresh_fails(monkeypatch):
    from types import SimpleNamespace

    from codex_shim import discover

    discover.clear_opencode_cli_models_cache()
    monkeypatch.setattr("codex_shim.discover.shutil.which", lambda _name: "/usr/bin/opencode")
    monkeypatch.setattr(
        "codex_shim.discover.subprocess.run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0, stdout="nvidia/meta/muse-glimmer-30b\n", stderr=""
        ),
    )
    assert discover.list_opencode_cli_models() == ["nvidia/meta/muse-glimmer-30b"]

    def boom(*_args, **_kwargs):
        raise OSError("opencode models failed")

    monkeypatch.setattr("codex_shim.discover.subprocess.run", boom)
    discover.expire_opencode_cli_models_cache()
    assert discover.list_opencode_cli_models() == ["nvidia/meta/muse-glimmer-30b"]


def test_list_opencode_cli_models_caches_subprocess(monkeypatch):
    from types import SimpleNamespace

    from codex_shim import discover

    assert discover._OPENCODE_CLI_MODELS_CACHE_TTL_SEC == 3 * 60 * 60

    calls = {"n": 0}

    def fake_run(*_args, **_kwargs):
        calls["n"] += 1
        return SimpleNamespace(returncode=0, stdout="opencode/big-pickle\nopenrouter/free\n", stderr="")

    discover.clear_opencode_cli_models_cache()
    monkeypatch.setattr("codex_shim.discover.shutil.which", lambda _name: "/usr/bin/opencode")
    monkeypatch.setattr("codex_shim.discover.subprocess.run", fake_run)

    assert discover.list_opencode_cli_models() == ["opencode/big-pickle", "openrouter/free"]
    assert discover.discover_opencode_cli_ids("openrouter") == ["free"]
    assert discover.list_opencode_cli_models() == ["opencode/big-pickle", "openrouter/free"]
    assert calls["n"] == 1


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


def _models_dev_catalog() -> dict:
    return {
        "opencode": {
            "models": {
                "x-preview-f-free": {
                    "id": "x-preview-f-free",
                    "name": "Ox Alpha Free (Unlimited)",
                    "description": "Stealth reasoning model for coding, agent work, and tool use",
                    "attachment": True,
                    "reasoning": True,
                    "reasoning_options": [{"type": "effort", "values": ["low", "high", "max"]}],
                    "limit": {"context": 1_000_000, "output": 131_072},
                    "modalities": {"input": ["text", "image", "video"], "output": ["text"]},
                    "interleaved": {"field": "reasoning_content"},
                }
            }
        },
        "openrouter": {
            "models": {
                "stealth/ox-alpha": {
                    "id": "stealth/ox-alpha",
                    "name": "Ox Alpha",
                    "description": "Multimodal reasoning model for visual analysis, planning, and tool use",
                    "attachment": True,
                    "reasoning": True,
                    "reasoning_options": [{"type": "effort", "values": ["low", "high", "max"]}],
                    "limit": {"context": 1_048_576, "output": 131_072},
                    "modalities": {"input": ["text", "image", "video"], "output": ["text"]},
                },
                "nvidia/nemotron-3-super-120b-a12b:free": {
                    "id": "nvidia/nemotron-3-super-120b-a12b:free",
                    "name": "Nemotron 3 Super 120B A12B (free)",
                    "reasoning": True,
                    "reasoning_options": [
                        {"type": "toggle"},
                        {"type": "effort", "values": ["low", "medium"]},
                    ],
                    "limit": {"context": 262_144, "output": 262_144},
                    "modalities": {"input": ["text"], "output": ["text"]},
                    "attachment": False,
                },
            }
        },
        "nvidia": {
            "models": {
                "nvidia/llama-3.3-nemotron-super-49b-v1": {
                    "id": "nvidia/llama-3.3-nemotron-super-49b-v1",
                    "name": "Llama 3.3 Nemotron Super 49B v1",
                    "reasoning": True,
                    "reasoning_options": [{"type": "toggle"}],
                    "limit": {"context": 131_072, "output": 65_536},
                    "modalities": {"input": ["text"], "output": ["text"]},
                    "attachment": False,
                }
            }
        },
    }


def test_metadata_from_models_dev_row_maps_ox_alpha_efforts_and_limits():
    from codex_shim.discover import metadata_from_models_dev_row

    meta = metadata_from_models_dev_row(
        _models_dev_catalog()["openrouter"]["models"]["stealth/ox-alpha"]
    )
    assert meta["upstream_name"] == "Ox Alpha"
    assert "visual analysis" in meta["upstream_description"]
    assert meta["reasoning_efforts"] == ["low", "high", "max"]
    assert meta["reported_context_limit"] == 1_048_576
    assert meta["output_limit"] == 131_072
    assert meta["input_modalities"] == ["text", "image", "video"]
    assert meta["reasoning"] is True
    assert meta["no_image_support"] is False


def test_discovered_nous_ox_alpha_maps_models_dev_metadata(monkeypatch):
    monkeypatch.setattr(
        "codex_shim.discover.fetch_models_dev_catalog",
        lambda: _models_dev_catalog(),
        raising=False,
    )
    models = _rows_to_shim_models(["stealth/ox-alpha"], NOUS_PORTAL_TEMPLATE)
    route = models[0]
    assert route.max_context_limit == 1_048_576
    assert route.max_output_tokens == 131_072
    assert route.supports_reasoning_summaries is True
    assert route.raw["reasoning_efforts"] == ["low", "high", "max"]
    assert route.raw["upstream_name"] == "Ox Alpha"
    assert route.raw["input_modalities"] == ["text", "image", "video"]


def test_discovered_opencode_ox_preview_maps_models_dev_reasoning(monkeypatch):
    monkeypatch.setattr(
        "codex_shim.discover.fetch_models_dev_catalog",
        lambda: _models_dev_catalog(),
        raising=False,
    )
    models = _rows_to_shim_models(["x-preview-f-free"], ZEN_PUBLIC_TEMPLATE)
    route = models[0]
    assert route.max_context_limit == 1_048_576
    assert route.max_output_tokens == 131_072
    assert route.raw["reasoning_efforts"] == ["low", "high", "max"]
    assert route.raw["upstream_name"] == "Ox Alpha Free (Unlimited)"
    assert route.supports_reasoning_summaries is True


def test_discovered_openrouter_free_maps_effort_variants(monkeypatch):
    monkeypatch.setattr(
        "codex_shim.discover.fetch_models_dev_catalog",
        lambda: _models_dev_catalog(),
        raising=False,
    )
    models = _rows_to_shim_models(
        ["nvidia/nemotron-3-super-120b-a12b:free"],
        OPENROUTER_FREE_TEMPLATE,
    )
    route = models[0]
    assert route.max_context_limit == 262_144
    assert route.max_output_tokens == 262_144
    assert route.raw["reasoning_efforts"] == ["low", "medium"]
    assert route.no_image_support is True
    assert route.raw["input_modalities"] == ["text"]


def test_discovered_nvidia_text_only_maps_output_limit(monkeypatch):
    monkeypatch.setattr(
        "codex_shim.discover.fetch_models_dev_catalog",
        lambda: _models_dev_catalog(),
        raising=False,
    )
    models = _rows_to_shim_models(
        ["nvidia/llama-3.3-nemotron-super-49b-v1"],
        NVIDIA_INTEGRATE_TEMPLATE,
    )
    route = models[0]
    assert route.max_context_limit == 131_072
    assert route.max_output_tokens == 65_536
    assert route.no_image_support is True
    assert route.supports_reasoning_summaries is True
    assert not route.raw.get("reasoning_efforts")

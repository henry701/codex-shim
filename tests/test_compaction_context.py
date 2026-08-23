from __future__ import annotations

from codex_shim.compaction.config import (
    CompactionSettings,
    DEFAULT_COMPACTION_OUTPUT_TOKEN_RESERVE,
    effective_compaction_output_token_reserve,
    load_compaction_settings,
)
from codex_shim.compaction.context import (
    compaction_budget_slug,
    compute_compaction_input_token_budget,
    context_window_tokens_for_slug,
)
from codex_shim.settings import ShimModel


def test_effective_output_reserve_defaults_to_20k():
    settings = CompactionSettings(summary_max_output_tokens=4096)
    assert effective_compaction_output_token_reserve(settings) == DEFAULT_COMPACTION_OUTPUT_TOKEN_RESERVE


def test_effective_output_reserve_honors_explicit_config():
    settings = CompactionSettings(compaction_output_token_reserve=96000)
    assert effective_compaction_output_token_reserve(settings) == 96000


def test_compute_input_budget_subtracts_output_reserve():
    settings = CompactionSettings(compaction_output_token_reserve=12000)
    assert compute_compaction_input_token_budget(1_000_000, settings) == 988_000


def test_compute_input_budget_returns_none_without_model_context():
    settings = CompactionSettings()
    assert compute_compaction_input_token_budget(None, settings) is None


def test_compaction_budget_slug_uses_override_model():
    settings = CompactionSettings(model="big-1m", override_current_model=True)
    assert compaction_budget_slug(settings, "oc-free-big-pickle") == "big-1m"


def test_compaction_budget_slug_keeps_thread_model_when_not_overridden():
    settings = CompactionSettings(model="big-1m", override_current_model=False)
    assert compaction_budget_slug(settings, "oc-free-big-pickle") == "oc-free-big-pickle"


def test_context_window_tokens_for_slug_reads_byok_model(tmp_path):
    models = [
        ShimModel(
            slug="oc-free-big-pickle",
            model="big-pickle",
            display_name="Pickle",
            provider="generic-chat-completion-api",
            base_url="https://example.test/v1",
            max_context_limit=128_000,
        )
    ]
    assert (
        context_window_tokens_for_slug(
            "oc-free-big-pickle",
            byok_models=models,
            catalog_path=tmp_path / "missing-custom_model_catalog.json",
        )
        == 128_000
    )


def test_context_window_tokens_for_slug_reads_catalog_entry(tmp_path):
    catalog = tmp_path / "custom_model_catalog.json"
    catalog.write_text(
        """
        {
          "models": [
            {
              "slug": "big-1m",
              "context_window": 1000000,
              "max_context_window": 1000000
            }
          ]
        }
        """
    )
    assert (
        context_window_tokens_for_slug(
            "big-1m",
            byok_models=[],
            catalog_path=catalog,
        )
        == 1_000_000
    )


def test_load_compaction_settings_accepts_output_reserve_and_legacy_alias(tmp_path):
    settings_path = tmp_path / "models.json"
    settings_path.write_text(
        """
        {
          "compaction": {
            "compaction_output_token_reserve": 16000
          }
        }
        """
    )
    loaded = load_compaction_settings(settings_path)
    assert loaded.compaction_output_token_reserve == 16000

    settings_path.write_text(
        """
        {
          "compaction": {
            "context_window_token_budget": 24000
          }
        }
        """
    )
    legacy = load_compaction_settings(settings_path)
    assert legacy.compaction_output_token_reserve == 24000


def test_load_compaction_settings_accepts_max_recent_user_prompts(tmp_path):
    settings_path = tmp_path / "models.json"
    settings_path.write_text(
        """
        {
          "compaction": {
            "max_recent_user_prompts": 50
          }
        }
        """
    )
    loaded = load_compaction_settings(settings_path)
    assert loaded.max_recent_user_prompts == 50

    settings_path.write_text(
        """
        {
          "compaction": {
            "max_recent_user_prompts": 12
          }
        }
        """
    )
    loaded = load_compaction_settings(settings_path)
    assert loaded.max_recent_user_prompts == 12

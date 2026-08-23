from __future__ import annotations

from codex_shim.naming import catalog_display_name, display_name_from_slug, format_cursor_display_name
from codex_shim.settings import ShimModel


def test_display_name_from_slug_handles_versions_and_vendors():
    assert display_name_from_slug("oc-free-minimax-m2-5-free", label_prefix="oc-free") == (
        "OpenCode Zen (free) — MiniMax M2.5 (free)"
    )
    assert display_name_from_slug("oc-free-nemotron-3-ultra-free", label_prefix="oc-free") == (
        "OpenCode Zen (free) — Nemotron 3 Ultra (free)"
    )
    assert display_name_from_slug("zen-deepseek-v4-flash", label_prefix="zen") == (
        "OpenCode Zen — DeepSeek V4 Flash"
    )
    assert display_name_from_slug("cursor-composer-2-5") == "Composer 2.5"
    assert display_name_from_slug("nous-stealth-ox-alpha", label_prefix="nous") == (
        "Nous Portal — Stealth Ox Alpha"
    )


def test_catalog_display_name_normalizes_explicit_zen_free_models():
    model = ShimModel(
        slug="zen-big-pickle",
        model="big-pickle",
        display_name="OpenCode Zen — Big Pickle (free)",
        provider="generic-chat-completion-api",
        base_url="https://opencode.ai/zen/v1",
        api_key="public",
    )
    assert catalog_display_name(model) == "OpenCode Zen (free) — Big Pickle"


def test_format_cursor_display_name_adds_prefix():
    assert format_cursor_display_name("Opus 4.5") == "Cursor - Opus 4.5"
    assert format_cursor_display_name("Cursor - Opus 4.5") == "Cursor - Opus 4.5"


def test_catalog_display_name_prefers_upstream_name_for_zen_routes():
    model = ShimModel(
        slug="oc-free-x-preview-f-free",
        model="x-preview-f-free",
        display_name="OpenCode Zen (free) — Ox Alpha Free (Unlimited)",
        provider="generic-chat-completion-api",
        base_url="https://opencode.ai/zen/v1",
        api_key="public",
        raw={"upstream_name": "Ox Alpha Free (Unlimited)"},
    )
    assert catalog_display_name(model) == "OpenCode Zen (free) — Ox Alpha Free (Unlimited)"

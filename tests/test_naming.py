from __future__ import annotations

from codex_shim.naming import display_name_from_slug


def test_display_name_from_slug_handles_versions_and_vendors():
    assert display_name_from_slug("zen-minimax-m2-5-free", label_prefix="zen") == (
        "Zen — MiniMax M2.5 (free)"
    )
    assert display_name_from_slug("zen-nemotron-3-ultra-free", label_prefix="zen") == (
        "Zen — Nemotron 3 Ultra (free)"
    )
    assert display_name_from_slug("zen-kimi-k2-6", label_prefix="zen") == "Zen — Kimi K2.6"
    assert display_name_from_slug("composer-2-5") == "Composer 2.5"

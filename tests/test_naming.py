from __future__ import annotations

from codex_shim.naming import display_name_from_slug


def test_display_name_from_slug_handles_versions_and_vendors():
    assert display_name_from_slug("oc-free-minimax-m2-5-free", label_prefix="oc-free") == (
        "OpenCode Zen (free) — MiniMax M2.5 (free)"
    )
    assert display_name_from_slug("oc-free-nemotron-3-ultra-free", label_prefix="oc-free") == (
        "OpenCode Zen (free) — Nemotron 3 Ultra (free)"
    )
    assert display_name_from_slug("cursor-composer-2-5") == "Composer 2.5"

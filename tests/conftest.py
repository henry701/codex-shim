from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _disable_model_discovery_by_default(monkeypatch, request):
    if "enable_model_discovery" in request.keywords:
        return
    monkeypatch.setattr("codex_shim.discover.fetch_zen_model_ids", lambda: [])
    monkeypatch.setattr("codex_shim.discover.fetch_zen_public_model_ids", lambda: [])
    monkeypatch.setattr("codex_shim.discover.fetch_openrouter_free_model_ids", lambda: [])
    monkeypatch.setattr("codex_shim.discover.fetch_nvidia_integrate_model_ids", lambda: [])
    monkeypatch.setattr("codex_shim.discover.discover_opencode_cli_ids", lambda *_args, **_kwargs: [])
    monkeypatch.setattr("codex_shim.discover.fetch_local_openai_models", lambda *_args, **_kwargs: [])
    monkeypatch.setattr("codex_shim.discover.discover_chatgpt_models_from_cursor", lambda: [])
    monkeypatch.setattr("codex_shim.discover.discover_chatgpt_model_ids_from_openai_api", lambda: [])


@pytest.fixture(autouse=True)
def _disable_cursor_passthrough_by_default(monkeypatch, request):
    if "cursor_present" in request.fixturenames:
        return

    def _off(**_kwargs):
        return False

    for target in (
        "codex_shim.cursor_passthrough.cursor_passthrough_available",
        "codex_shim.server.cursor_passthrough_available",
        "codex_shim.catalog.cursor_passthrough_available",
        "codex_shim.cli.cursor_passthrough_available",
    ):
        monkeypatch.setattr(target, _off, raising=False)

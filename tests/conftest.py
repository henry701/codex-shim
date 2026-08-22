from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _disable_model_discovery_by_default(monkeypatch, request):
    if "enable_model_discovery" in request.keywords:
        return
    monkeypatch.setattr("codex_shim.discover.fetch_zen_model_ids", lambda **_kwargs: [])
    monkeypatch.setattr("codex_shim.discover.fetch_models_dev_opencode_free_model_ids", lambda: [])
    monkeypatch.setattr("codex_shim.discover.fetch_zen_paid_model_ids", lambda **_kwargs: [])
    monkeypatch.setattr("codex_shim.discover.fetch_zen_public_model_ids", lambda: [])
    monkeypatch.setattr("codex_shim.discover.fetch_openrouter_free_model_ids", lambda: [])
    monkeypatch.setattr("codex_shim.discover.fetch_nvidia_integrate_model_ids", lambda: [])
    monkeypatch.setattr("codex_shim.discover.discover_opencode_cli_ids", lambda *_args, **_kwargs: [])
    monkeypatch.setattr("codex_shim.discover.fetch_local_openai_models", lambda *_args, **_kwargs: [])
    monkeypatch.setattr("codex_shim.discover.fetch_nous_portal_model_ids", lambda **_kwargs: [])
    monkeypatch.setattr("codex_shim.discover.discover_chatgpt_models_from_cursor", lambda: [])
    monkeypatch.setattr("codex_shim.discover.discover_chatgpt_model_ids_from_openai_api", lambda: [])
    monkeypatch.setattr("codex_shim.discover.fetch_chatgpt_codex_backend_models", lambda **_kwargs: [])


@pytest.fixture(autouse=True)
def _disable_nous_portal_by_default(monkeypatch, request):
    if request.node.get_closest_marker("nous_portal"):
        return
    monkeypatch.setattr("codex_shim.discover.fetch_nous_portal_model_ids", lambda **_kwargs: [], raising=False)
    monkeypatch.setattr("codex_shim.nous_auth.refresh_nous_oauth_on_startup", lambda **_kwargs: False, raising=False)
    monkeypatch.setattr("codex_shim.nous_auth.refresh_nous_oauth", lambda **_kwargs: False, raising=False)


@pytest.fixture(autouse=True)
def _clear_opencode_cli_models_cache():
    from codex_shim.discover import clear_opencode_cli_models_cache

    clear_opencode_cli_models_cache()
    yield
    clear_opencode_cli_models_cache()


@pytest.fixture(autouse=True)
def _isolate_published_desktop_catalog(monkeypatch, tmp_path_factory):
    # Keep catalog sync tests from reading the developer's live Desktop catalog.
    monkeypatch.setattr(
        "codex_shim.settings.DEFAULT_DESKTOP_MODEL_CATALOG",
        tmp_path_factory.mktemp("published-catalog") / "missing-custom_model_catalog.json",
    )


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

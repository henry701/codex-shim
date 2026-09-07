from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _reset_origin_backoff_between_tests():
    from codex_shim.net.retry import reset_origin_backoff

    reset_origin_backoff()
    yield
    reset_origin_backoff()


@pytest.fixture(autouse=True)
def _disable_periodic_catalog_refresh(monkeypatch):
    # Serve would otherwise rewrite the live Desktop catalog from stubbed discovery.
    monkeypatch.setattr("codex_shim.server._CATALOG_REFRESH_INITIAL_DELAY_SEC", None)


@pytest.fixture(autouse=True)
def _isolate_hermes_auth_paths(monkeypatch, tmp_path_factory):
    # Refresh tokens are single-use. Tests must never read or rotate ~/.hermes.
    home = tmp_path_factory.mktemp("hermes-home")
    for key in (
        "HERMES_SHARED_AUTH_DIR",
        "NOUS_API_KEY",
        "HERMES_PORTAL_BASE_URL",
        "NOUS_PORTAL_BASE_URL",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HERMES_HOME", str(home))


@pytest.fixture(autouse=True)
def _stub_nous_api_key_unless_portal(monkeypatch, request):
    if request.node.get_closest_marker("nous_portal"):
        return

    def _empty(*, environ=None, hermes_dir=None):
        return ""

    monkeypatch.setattr("codex_shim.nous_auth.resolve_nous_api_key", _empty, raising=False)
    monkeypatch.setattr("codex_shim.discover.resolve_nous_api_key", _empty, raising=False)


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
    monkeypatch.setattr("codex_shim.discover.fetch_models_dev_catalog", lambda: {}, raising=False)
    monkeypatch.setattr("codex_shim.discover.discover_opencode_cli_ids", lambda *_args, **_kwargs: [])
    monkeypatch.setattr("codex_shim.discover.fetch_local_openai_models", lambda *_args, **_kwargs: [])
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
def _reset_nous_oauth_startup_state():
    from codex_shim.nous_auth import reset_nous_oauth_startup_state

    reset_nous_oauth_startup_state()
    yield
    reset_nous_oauth_startup_state()


@pytest.fixture(autouse=True)
def _clear_opencode_cli_models_cache():
    from codex_shim.discover import (
        clear_detected_codex_cli_version,
        clear_models_dev_catalog_cache,
        clear_opencode_cli_models_cache,
    )

    clear_opencode_cli_models_cache()
    clear_models_dev_catalog_cache()
    clear_detected_codex_cli_version()
    yield
    clear_opencode_cli_models_cache()
    clear_models_dev_catalog_cache()
    clear_detected_codex_cli_version()


@pytest.fixture(autouse=True)
def _isolate_published_desktop_catalog(monkeypatch, tmp_path_factory):
    # Keep catalog sync/generate tests from reading or writing the live Desktop catalog.
    isolated = tmp_path_factory.mktemp("published-catalog") / "custom_model_catalog.json"
    monkeypatch.setattr("codex_shim.settings.DEFAULT_DESKTOP_MODEL_CATALOG", isolated)
    monkeypatch.setattr("codex_shim.cli.DESKTOP_CATALOG_PATH", isolated)


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

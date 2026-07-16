"""ChatGPT Codex backend /models is the primary passthrough catalog source."""

from __future__ import annotations

import json
import logging
from urllib.error import HTTPError, URLError

import pytest

from codex_shim.catalog_slugs import upstream_from_codex_catalog_slug
from codex_shim.discover import (
    CHATGPT_CODEX_MODELS_URL,
    fetch_chatgpt_codex_backend_models,
    fetch_http_json,
    persist_chatgpt_models_cache,
    refresh_codex_auth_tokens,
)
from codex_shim.settings import load_chatgpt_passthrough_catalog_models


def test_upstream_from_codex_catalog_slug_recovers_dotted_versions():
    assert upstream_from_codex_catalog_slug("codex-gpt-5-6-terra") == "gpt-5.6-terra"
    assert upstream_from_codex_catalog_slug("codex-gpt-5-4-mini") == "gpt-5.4-mini"
    assert upstream_from_codex_catalog_slug("codex-auto-review") == "codex-auto-review"


def test_load_chatgpt_prefers_backend_models_over_cache(monkeypatch, tmp_path):
    cache = tmp_path / "models_cache.json"
    cache.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "slug": "gpt-5.5",
                        "display_name": "GPT-5.5 Cached Stale",
                        "visibility": "list",
                    }
                ]
            }
        )
    )
    monkeypatch.setattr(
        "codex_shim.discover.fetch_chatgpt_codex_backend_models",
        lambda: [
            {
                "slug": "gpt-5.6-sol",
                "display_name": "GPT-5.6 Sol",
                "visibility": "list",
                "context_window": 400000,
            },
            {
                "slug": "gpt-5.5",
                "display_name": "GPT-5.5 Live",
                "visibility": "list",
            },
        ],
    )
    # Avoid writing into the developer's live models_cache during this unit test.
    monkeypatch.setattr("codex_shim.discover.persist_chatgpt_models_cache", lambda *a, **k: None)
    entries = load_chatgpt_passthrough_catalog_models(cache)
    by_slug = {entry["slug"]: entry for entry in entries}
    assert set(by_slug) == {"codex-gpt-5-6-sol", "codex-gpt-5-5"}
    assert by_slug["codex-gpt-5-6-sol"]["_upstream_model"] == "gpt-5.6-sol"
    assert by_slug["codex-gpt-5-5"]["display_name"] == "GPT-5.5 Live"


def test_load_chatgpt_persists_backend_models_to_cache(monkeypatch, tmp_path):
    cache = tmp_path / "models_cache.json"
    monkeypatch.setattr(
        "codex_shim.discover.fetch_chatgpt_codex_backend_models",
        lambda: [
            {"slug": "gpt-5.6-sol", "display_name": "GPT-5.6 Sol", "visibility": "list"},
            {"slug": "gpt-5.5", "display_name": "GPT-5.5 Live", "visibility": "list"},
        ],
    )
    load_chatgpt_passthrough_catalog_models(cache)
    persisted = json.loads(cache.read_text())
    assert [m["slug"] for m in persisted["models"]] == ["gpt-5.6-sol", "gpt-5.5"]


def test_load_chatgpt_falls_back_to_cache_with_warning(monkeypatch, tmp_path, caplog):
    cache = tmp_path / "models_cache.json"
    cache.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "slug": "gpt-5.5",
                        "display_name": "GPT-5.5 Cached",
                        "visibility": "list",
                        "context_window": 272000,
                    }
                ]
            }
        )
    )
    monkeypatch.setattr(
        "codex_shim.discover.fetch_chatgpt_codex_backend_models",
        lambda: [],
    )
    with caplog.at_level(logging.WARNING, logger="codex_shim.settings"):
        entries = load_chatgpt_passthrough_catalog_models(cache)
    assert any("models_cache.json" in record.getMessage() for record in caplog.records)
    assert any("backend" in record.getMessage().lower() for record in caplog.records)
    by_slug = {entry["slug"]: entry for entry in entries}
    assert by_slug["codex-gpt-5-5"]["display_name"] == "GPT-5.5 Cached"
    assert by_slug["codex-gpt-5-5"]["_upstream_model"] == "gpt-5.5"


def test_load_chatgpt_merges_published_catalog_when_cache_stale(monkeypatch, tmp_path, caplog):
    cache = tmp_path / "models_cache.json"
    cache.write_text(
        json.dumps(
            {
                "models": [
                    {"slug": "gpt-5.5", "display_name": "GPT-5.5", "visibility": "list"},
                ]
            }
        )
    )
    published = tmp_path / "custom_model_catalog.json"
    published.write_text(
        json.dumps(
            {
                "models": [
                    {"slug": "codex-gpt-5-5", "display_name": "GPT-5.5", "visibility": "list"},
                    {"slug": "codex-gpt-5-6-terra", "display_name": "GPT-5.6-Terra", "visibility": "list"},
                    {"slug": "cursor-gpt-5-6-terra-high", "display_name": "Cursor Terra", "visibility": "list"},
                ]
            }
        )
    )
    monkeypatch.setattr("codex_shim.discover.fetch_chatgpt_codex_backend_models", lambda: [])
    with caplog.at_level(logging.WARNING, logger="codex_shim.settings"):
        entries = load_chatgpt_passthrough_catalog_models(cache, catalog_path=published)
    by_slug = {entry["slug"]: entry for entry in entries}
    assert set(by_slug) == {"codex-gpt-5-5", "codex-gpt-5-6-terra"}
    assert by_slug["codex-gpt-5-6-terra"]["_upstream_model"] == "gpt-5.6-terra"
    assert any("custom_model_catalog.json" in record.getMessage() for record in caplog.records)


def test_persist_chatgpt_models_cache_writes_client_version(tmp_path):
    path = tmp_path / "models_cache.json"
    written = persist_chatgpt_models_cache(
        [{"slug": "gpt-5.6-terra", "display_name": "GPT-5.6-Terra", "visibility": "list"}],
        path,
        client_version="0.144.1",
    )
    assert written == path
    payload = json.loads(path.read_text())
    assert payload["client_version"] == "0.144.1"
    assert payload["models"][0]["slug"] == "gpt-5.6-terra"


def test_refresh_codex_auth_tokens_updates_auth_file(monkeypatch, tmp_path):
    auth = tmp_path / "auth.json"
    auth.write_text(
        json.dumps(
            {
                "tokens": {
                    "access_token": "old-access",
                    "refresh_token": "old-refresh",
                    "id_token": "old-id",
                    "account_id": "acct-1",
                }
            }
        )
    )

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(
                {
                    "access_token": "new-access",
                    "refresh_token": "new-refresh",
                    "id_token": "new-id",
                    "expires_in": 864000,
                }
            ).encode()

    monkeypatch.setattr("codex_shim.discover.urlopen", lambda *a, **k: _Resp())
    assert refresh_codex_auth_tokens(auth) is True
    data = json.loads(auth.read_text())
    assert data["tokens"]["access_token"] == "new-access"
    assert data["tokens"]["refresh_token"] == "new-refresh"
    assert data["tokens"]["account_id"] == "acct-1"
    assert data["last_refresh"]


def test_fetch_chatgpt_codex_backend_models_refreshes_on_401(monkeypatch, tmp_path):
    auth = tmp_path / "auth.json"
    auth.write_text(
        json.dumps(
            {
                "tokens": {
                    "access_token": "expired",
                    "refresh_token": "refresh-me",
                    "account_id": "acct-1",
                }
            }
        )
    )
    monkeypatch.setattr("codex_shim.settings.DEFAULT_CODEX_AUTH", auth)
    calls: list[str] = []

    def fake_fetch(url, *, headers=None, timeout=20.0, retries=3, backoff_base=0.5, backoff_factor=2.0):
        token = (headers or {}).get("Authorization", "")
        calls.append(token)
        if "expired" in token:
            raise HTTPError(url, 401, "Unauthorized", hdrs=None, fp=None)
        return {"models": [{"slug": "gpt-5.6-terra", "display_name": "GPT-5.6-Terra", "visibility": "list"}]}

    def fake_refresh(path=None, *, timeout=20.0):
        data = json.loads(auth.read_text())
        data["tokens"]["access_token"] = "fresh"
        auth.write_text(json.dumps(data))
        return True

    monkeypatch.setattr("codex_shim.discover.fetch_http_json", fake_fetch)
    monkeypatch.setattr("codex_shim.discover.refresh_codex_auth_tokens", fake_refresh)
    models = fetch_chatgpt_codex_backend_models(client_version="0.144.1")
    assert [m["slug"] for m in models] == ["gpt-5.6-terra"]
    assert calls == ["Bearer expired", "Bearer fresh"]


def test_fetch_http_json_retries_with_backoff(monkeypatch):
    sleeps: list[float] = []
    attempts = {"n": 0}

    def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b'{"ok":true}'

    def fake_urlopen(request, timeout=20.0):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise URLError("temporary")
        return _Resp()

    monkeypatch.setattr("codex_shim.discover.time.sleep", fake_sleep)
    monkeypatch.setattr("codex_shim.discover.urlopen", fake_urlopen)
    payload = fetch_http_json(
        "https://example.test/models",
        retries=3,
        backoff_base=0.01,
        backoff_factor=2.0,
    )
    assert payload == {"ok": True}
    assert attempts["n"] == 3
    assert sleeps == pytest.approx([0.01, 0.02])


def test_fetch_chatgpt_codex_backend_models_builds_url_and_auth(monkeypatch, tmp_path):
    auth = tmp_path / "auth.json"
    auth.write_text(
        json.dumps(
            {
                "tokens": {
                    "access_token": "tok-abc",
                    "account_id": "acct-1",
                }
            }
        )
    )
    monkeypatch.setattr("codex_shim.settings.DEFAULT_CODEX_AUTH", auth)
    captured: dict[str, object] = {}

    def fake_fetch(url, *, headers=None, timeout=20.0, retries=3, backoff_base=0.5, backoff_factor=2.0):
        captured["url"] = url
        captured["headers"] = headers
        captured["retries"] = retries
        return {
            "models": [
                {"slug": "gpt-5.6-luna", "display_name": "GPT-5.6 Luna", "visibility": "list"},
                {"slug": "hidden-model", "visibility": "hidden"},
            ]
        }

    monkeypatch.setattr("codex_shim.discover.fetch_http_json", fake_fetch)
    models = fetch_chatgpt_codex_backend_models(client_version="0.144.1")
    assert captured["url"] == f"{CHATGPT_CODEX_MODELS_URL}?client_version=0.144.1"
    headers = captured["headers"]
    assert isinstance(headers, dict)
    assert headers["Authorization"] == "Bearer tok-abc"
    assert headers["ChatGPT-Account-Id"] == "acct-1"
    assert captured["retries"] == 3
    assert [m["slug"] for m in models] == ["gpt-5.6-luna"]


def test_fetch_chatgpt_codex_backend_models_returns_empty_without_auth(monkeypatch, tmp_path):
    missing = tmp_path / "missing-auth.json"
    monkeypatch.setattr("codex_shim.settings.DEFAULT_CODEX_AUTH", missing)
    called = {"n": 0}

    def boom(*_args, **_kwargs):
        called["n"] += 1
        raise AssertionError("must not fetch without auth")

    monkeypatch.setattr("codex_shim.discover.fetch_http_json", boom)
    assert fetch_chatgpt_codex_backend_models() == []
    assert called["n"] == 0

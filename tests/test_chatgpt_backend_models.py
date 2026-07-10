"""ChatGPT Codex backend /models is the primary passthrough catalog source."""

from __future__ import annotations

import json
import logging
from urllib.error import HTTPError, URLError

import pytest

from codex_shim.discover import (
    CHATGPT_CODEX_MODELS_URL,
    fetch_chatgpt_codex_backend_models,
    fetch_http_json,
)
from codex_shim.settings import load_chatgpt_passthrough_catalog_models


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
    entries = load_chatgpt_passthrough_catalog_models(cache)
    by_slug = {entry["slug"]: entry for entry in entries}
    assert set(by_slug) == {"codex-gpt-5-6-sol", "codex-gpt-5-5"}
    assert by_slug["codex-gpt-5-6-sol"]["_upstream_model"] == "gpt-5.6-sol"
    assert by_slug["codex-gpt-5-5"]["display_name"] == "GPT-5.5 Live"


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

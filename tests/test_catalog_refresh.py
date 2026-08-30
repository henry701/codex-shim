"""In-process catalog refresh must rewrite Desktop + ChatGPT catalog files."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from aiohttp.test_utils import TestClient, TestServer

from codex_shim import cli
from codex_shim.server import ShimServer


def _explicit_settings(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "model": "llama3.2",
                        "display_name": "Llama",
                        "provider": "generic-chat-completion-api",
                        "base_url": "http://127.0.0.1:11434/v1",
                        "api_key": "local",
                    }
                ]
            }
        )
    )


@pytest.mark.asyncio
async def test_periodic_catalog_refresh_writes_desktop_and_chatgpt_cache(
    tmp_path, monkeypatch
):
    settings = tmp_path / "models.json"
    _explicit_settings(settings)
    runtime_catalog = tmp_path / "runtime-catalog.json"
    desktop_catalog = tmp_path / "desktop-catalog.json"
    chatgpt_cache = tmp_path / "models_cache.json"

    monkeypatch.setattr(cli, "CATALOG_PATH", runtime_catalog)
    monkeypatch.setattr(cli, "DESKTOP_CATALOG_PATH", desktop_catalog)
    monkeypatch.setattr("codex_shim.settings.DEFAULT_CODEX_MODELS_CACHE", chatgpt_cache)
    monkeypatch.setattr("codex_shim.settings.chatgpt_passthrough_available", lambda **_k: True)
    monkeypatch.setattr("codex_shim.catalog.chatgpt_passthrough_available", lambda **_k: True)
    monkeypatch.setattr("codex_shim.server.chatgpt_passthrough_available", lambda **_k: True)
    monkeypatch.setattr(
        "codex_shim.discover.fetch_chatgpt_codex_backend_models",
        lambda **_k: [
            {"slug": "gpt-5.6-terra", "display_name": "GPT-5.6-Terra", "visibility": "list"}
        ],
    )
    monkeypatch.setattr("codex_shim.server._CATALOG_REFRESH_INITIAL_DELAY_SEC", 0)
    monkeypatch.setattr("codex_shim.server._MODELS_CACHE_TTL_SEC", 3600)

    shim = ShimServer(settings)
    async with TestClient(TestServer(shim.app())) as _client:
        for _ in range(50):
            if desktop_catalog.exists() and chatgpt_cache.exists():
                break
            await asyncio.sleep(0.05)

    assert desktop_catalog.exists(), "periodic refresh never wrote ~/.codex custom_model_catalog.json"
    desktop = json.loads(desktop_catalog.read_text())
    slugs = {entry.get("slug") for entry in desktop.get("models", [])}
    assert "llama3-2" in slugs
    assert any(str(slug).startswith("codex-gpt-5-6-terra") for slug in slugs)

    cached = json.loads(chatgpt_cache.read_text())
    assert [row["slug"] for row in cached["models"]] == ["gpt-5.6-terra"]

from __future__ import annotations

import asyncio

import pytest
from aiohttp.test_utils import TestClient, TestServer

from codex_shim.server import ShimServer


@pytest.mark.asyncio
async def test_health_returns_cached_snapshot_without_recomputing(tmp_path, monkeypatch):
    settings = tmp_path / "settings.json"
    settings.write_text('{"models": []}')
    shim = ShimServer(settings)
    calls = {"count": 0}

    def tracked_load():
        calls["count"] += 1
        return []

    monkeypatch.setattr(shim.settings, "load", tracked_load)

    async with TestClient(TestServer(shim.app())) as client:
        assert calls["count"] == 0
        for _ in range(3):
            resp = await client.get("/health")
            assert resp.status == 200
            payload = await resp.json()
            assert payload["ok"] is True
        assert calls["count"] == 0


@pytest.mark.asyncio
async def test_health_responds_quickly_after_startup(tmp_path, monkeypatch):
    settings = tmp_path / "settings.json"
    settings.write_text('{"models": []}')
    shim = ShimServer(settings)
    monkeypatch.setattr(shim.settings, "load", lambda: [])

    async with TestClient(TestServer(shim.app())) as client:
        await asyncio.sleep(0.01)
        resp = await client.get("/health")
        assert resp.status == 200


@pytest.mark.asyncio
async def test_startup_health_counts_explicit_models_without_discovery(tmp_path, monkeypatch):
    import json

    settings = tmp_path / "settings.json"
    settings.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "model": "a",
                        "display_name": "A",
                        "provider": "openai",
                        "base_url": "http://127.0.0.1:9/v1",
                        "api_key": "k",
                    },
                    {
                        "model": "b",
                        "display_name": "B",
                        "provider": "openai",
                        "base_url": "http://127.0.0.1:9/v1",
                        "api_key": "k",
                    },
                ]
            }
        )
    )
    monkeypatch.setattr("codex_shim.server.chatgpt_passthrough_available", lambda: False)
    monkeypatch.setattr("codex_shim.server.cursor_passthrough_available", lambda: False)

    def boom(*_args, **_kwargs):
        raise AssertionError("discovery must not run on serve bind")

    monkeypatch.setattr("codex_shim.discover.discover_byok_models", boom)
    shim = ShimServer(settings)
    async with TestClient(TestServer(shim.app())) as client:
        payload = await (await client.get("/health")).json()
    assert payload["ok"] is True
    assert payload["models"] == 2

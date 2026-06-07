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
        assert calls["count"] == 1
        for _ in range(3):
            resp = await client.get("/health")
            assert resp.status == 200
            payload = await resp.json()
            assert payload["ok"] is True
        assert calls["count"] == 1


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

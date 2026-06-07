from __future__ import annotations

import asyncio
import time

import pytest
from aiohttp.test_utils import TestClient, TestServer

from codex_shim.server import ShimServer


@pytest.mark.asyncio
async def test_health_responds_while_model_load_is_slow(monkeypatch, tmp_path):
    settings = tmp_path / "settings.json"
    settings.write_text('{"models": []}')
    shim = ShimServer(settings)

    def slow_load():
        time.sleep(2.0)
        return []

    monkeypatch.setattr(shim.settings, "load", slow_load)

    async with TestClient(TestServer(shim.app())) as client:
        await asyncio.sleep(0.05)
        started = time.monotonic()
        resp = await client.get("/health")
        elapsed = time.monotonic() - started
        assert resp.status == 200
        payload = await resp.json()
        assert payload["ok"] is True
        assert elapsed < 0.5

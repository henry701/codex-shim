"""Observe the live Codex Desktop ChatGPT-passthrough goal session.

These skip when 8765 is down so CI stays hermetic. Locally the health probe
asserts the shim is serving ChatGPT passthrough. The Luna goal-session use
case (Helmholtz / lcs-new-age parent-agent) is driven explicitly through
``scripts/smoke_chatgpt_passthrough.sh`` (default Luna slug) or
``CODEX_SHIM_LIVE_SMOKE=1 uv run pytest tests/test_smoke_chatgpt_passthrough.py``
instead of depending on incidental traffic since the last bind.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

import pytest

LIVE_HEALTH_URL = "http://127.0.0.1:8765/health"


def _live_health() -> dict | None:
    try:
        with urllib.request.urlopen(LIVE_HEALTH_URL, timeout=2) as response:
            payload = json.loads(response.read().decode())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return None
    return payload if isinstance(payload, dict) else None


@pytest.mark.integration
def test_live_goal_session_health_chatgpt_passthrough():
    health = _live_health()
    if health is None:
        pytest.skip("live shim is not reachable on 8765")
    assert health.get("ok") is True
    assert health.get("chatgpt_passthrough") is True
    models = health.get("models")
    count = models if isinstance(models, int) else len(models or [])
    assert count > 0

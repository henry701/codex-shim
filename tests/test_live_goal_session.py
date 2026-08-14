"""Observe the live Codex Desktop ChatGPT-passthrough goal session.

These skip when 8765 is down so CI stays hermetic. Locally they assert the
shim is serving ChatGPT passthrough and that Luna turns completed after the
last bind — the Helmholtz / lcs-new-age parent-agent use-case.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path

import pytest

LIVE_HEALTH_URL = "http://127.0.0.1:8765/health"
SHIM_LOG = Path.home() / ".codex-shim" / "shim.log"
BIND_MARKER = "======== Running on http://127.0.0.1:8765 ========"


def _live_health() -> dict | None:
    try:
        with urllib.request.urlopen(LIVE_HEALTH_URL, timeout=2) as response:
            payload = json.loads(response.read().decode())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return None
    return payload if isinstance(payload, dict) else None


def _log_since_last_bind() -> str:
    if not SHIM_LOG.is_file():
        return ""
    text = SHIM_LOG.read_text(errors="replace")
    idx = text.rfind(BIND_MARKER)
    return text[idx:] if idx >= 0 else text[-120000:]


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


@pytest.mark.integration
def test_live_goal_session_luna_responses_after_last_bind():
    if _live_health() is None:
        pytest.skip("live shim is not reachable on 8765")
    if not SHIM_LOG.is_file():
        pytest.skip("shim.log is missing")
    tail = _log_since_last_bind()
    assert "[req] /v1/responses" in tail
    assert "status=200" in tail
    assert "gpt-5-6-luna" in tail or "gpt-5.6-luna" in tail
    assert "chatgpt-passthrough" in tail

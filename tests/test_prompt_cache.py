from __future__ import annotations

import json

import pytest
from aiohttp.test_utils import TestClient, TestServer

from codex_shim.prompt_cache import (
    PROMPT_CACHE_BODY_KEYS,
    format_prompt_cache_req_suffix,
    prompt_cache_fields_from_body,
)
from codex_shim.server import ShimServer, _log_client_request, _sanitize_chatgpt_passthrough_body


def test_prompt_cache_fields_from_body_returns_known_keys_in_order():
    body = {
        "model": "codex-gpt-5-6-terra",
        "prompt_cache_retention": "24h",
        "prompt_cache_options": {"mode": "explicit", "ttl": "30m"},
        "prompt_cache_key": "thread-abc",
        "prompt_cache_breakpoint": {"type": "message", "index": 3},
        "input": [],
    }

    assert list(prompt_cache_fields_from_body(body).keys()) == list(PROMPT_CACHE_BODY_KEYS)


def test_format_prompt_cache_req_suffix_empty_when_absent():
    assert format_prompt_cache_req_suffix({"model": "codex-gpt-5-6-terra"}) == ""


def test_format_prompt_cache_req_suffix_renders_all_fields():
    body = {
        "prompt_cache_key": "thread-abc",
        "prompt_cache_options": {"mode": "implicit", "ttl": "30m"},
        "prompt_cache_breakpoint": {"type": "message", "index": 1},
    }
    suffix = format_prompt_cache_req_suffix(body)
    assert "prompt_cache_key='thread-abc'" in suffix
    assert "prompt_cache_options={'mode': 'implicit', 'ttl': '30m'}" in suffix
    assert "prompt_cache_breakpoint={'type': 'message', 'index': 1}" in suffix


def test_sanitize_chatgpt_passthrough_body_preserves_prompt_cache_fields():
    body = {
        "model": "codex-gpt-5-6-terra",
        "prompt_cache_key": "thread-abc",
        "prompt_cache_options": {"mode": "explicit", "ttl": "30m"},
        "prompt_cache_breakpoint": {"type": "message", "index": 2},
        "prompt_cache_retention": "24h",
        "input": [{"type": "message", "role": "user", "content": "hi"}],
    }

    sanitized = _sanitize_chatgpt_passthrough_body(body)

    for key in PROMPT_CACHE_BODY_KEYS:
        assert sanitized[key] == body[key]


def test_log_client_request_includes_prompt_cache_fields(capsys):
    _log_client_request(
        "/v1/responses",
        {
            "model": "codex-gpt-5-6-terra",
            "stream": True,
            "previous_response_id": "resp_prev",
            "prompt_cache_key": "thread-abc",
            "prompt_cache_options": {"mode": "implicit"},
            "input": [{"type": "message", "role": "user", "content": "hi"}],
        },
        transport="ws",
    )

    line = capsys.readouterr().out.strip()
    assert line.startswith("[req] /v1/responses transport=ws")
    assert "prompt_cache_key='thread-abc'" in line
    assert "prompt_cache_options={'mode': 'implicit'}" in line


@pytest.fixture
def auth_present(monkeypatch, tmp_path):
    auth = tmp_path / "auth.json"
    auth.write_text(json.dumps({"tokens": {"access_token": "stub", "account_id": "acct"}}))
    monkeypatch.setattr("codex_shim.settings.DEFAULT_CODEX_AUTH", auth)
    monkeypatch.setattr("codex_shim.server.DEFAULT_CODEX_AUTH", auth)
    return auth


@pytest.mark.asyncio
async def test_chatgpt_passthrough_forwards_prompt_cache_fields_to_upstream(
    monkeypatch, tmp_path, auth_present
):
    captured: dict[str, object] = {}

    class FakeUpstream:
        status = 200
        content_type = "application/json"
        headers = {}

        async def json(self, content_type=None):
            return {"id": "resp_1", "model": "gpt-5.6-terra", "output": []}

        def release(self):
            pass

    async def fake_post(self, url, json=None, headers=None):
        captured["url"] = url
        captured["body"] = json
        return FakeUpstream()

    monkeypatch.setattr("codex_shim.server.ClientSession.post", fake_post)
    settings = tmp_path / "settings.json"
    settings.write_text("{}")
    shim_client = TestClient(TestServer(ShimServer(settings).app()))
    await shim_client.start_server()

    cache_fields = {
        "prompt_cache_key": "thread-abc",
        "prompt_cache_options": {"mode": "explicit", "ttl": "30m"},
        "prompt_cache_breakpoint": {"type": "message", "index": 4},
    }
    resp = await shim_client.post(
        "/v1/responses",
        json={
            "model": "codex-gpt-5-5",
            "input": [{"type": "message", "role": "user", "content": "hi"}],
            **cache_fields,
        },
    )

    assert resp.status == 200
    upstream_body = captured["body"]
    assert isinstance(upstream_body, dict)
    for key, value in cache_fields.items():
        assert upstream_body[key] == value

    await shim_client.close()

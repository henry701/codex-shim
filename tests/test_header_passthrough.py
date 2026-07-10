from __future__ import annotations

from multidict import CIMultiDict

from codex_shim.header_passthrough import (
    apply_upstream_headers_to_response,
    chatgpt_passthrough_upstream_headers,
    client_headers_for_upstream,
    forwardable_upstream_response_headers,
    log_upstream_response_headers,
    openai_upstream_headers,
)


def test_client_headers_for_upstream_preserves_codex_headers_and_applies_overrides():
    request_headers = CIMultiDict(
        [
            ("Host", "127.0.0.1:8765"),
            ("session_id", "sess-123"),
            ("x-codex-turn-state", "active"),
            ("Authorization", "Bearer codex-sent"),
            ("Accept-Encoding", "zstd, gzip"),
            ("X-Codex-Shim-Picker-Token", "secret"),
        ]
    )
    merged = client_headers_for_upstream(
        request_headers,
        setdefaults={"Accept-Encoding": "zstd, gzip, deflate", "OpenAI-Beta": "responses=2026-02-06"},
        overrides={
            "Authorization": "Bearer auth-json",
            "Accept": "application/json",
        },
    )
    assert merged["session_id"] == "sess-123"
    assert merged["x-codex-turn-state"] == "active"
    assert merged["Authorization"] == "Bearer auth-json"
    assert merged["Accept"] == "application/json"
    assert merged["Accept-Encoding"] == "zstd, gzip"
    assert merged["OpenAI-Beta"] == "responses=2026-02-06"
    assert "Host" not in merged
    assert "X-Codex-Shim-Picker-Token" not in merged


def test_chatgpt_passthrough_upstream_headers_overrides_websocket_beta_for_http():
    merged = chatgpt_passthrough_upstream_headers(
        {"OpenAI-Beta": "responses_websockets=2026-02-06", "originator": "Codex Desktop"},
        access_token="token",
        account_id="acct",
        accept="application/json",
    )
    assert merged["OpenAI-Beta"] == "responses=2026-02-06"
    assert merged["Authorization"] == "Bearer token"


def test_forwardable_upstream_response_headers_skips_hop_by_hop():
    upstream = CIMultiDict(
        [
            ("x-request-id", "req_abc"),
            ("openai-processing-ms", "12"),
            ("content-length", "999"),
            ("transfer-encoding", "chunked"),
        ]
    )
    assert forwardable_upstream_response_headers(upstream) == {
        "x-request-id": "req_abc",
        "openai-processing-ms": "12",
    }


def test_apply_upstream_headers_to_response():
    from aiohttp import web

    response = web.Response(text="ok")
    apply_upstream_headers_to_response(
        response,
        {"x-request-id": "req_abc", "content-length": "2"},
    )
    assert response.headers["x-request-id"] == "req_abc"
    assert "content-length" not in response.headers


def test_client_headers_for_upstream_drops_websocket_upgrade_headers():
    request_headers = CIMultiDict(
        [
            ("session-id", "sess-1"),
            ("Sec-WebSocket-Key", "abc"),
            ("Sec-WebSocket-Version", "13"),
            ("Sec-WebSocket-Extensions", "permessage-deflate"),
        ]
    )
    merged = client_headers_for_upstream(request_headers)
    assert merged == {"session-id": "sess-1"}


def test_client_headers_for_upstream_drops_content_encoding():
    """Desktop may zstd-compress the body to the shim with Content-Encoding.

    The shim decompresses, mutates JSON, then re-POSTs plain JSON via
    ``ClientSession.post(..., json=...)``. Forwarding Content-Encoding makes
    ChatGPT try to decompress uncompressed bytes and return bare Bad Request.
    """
    merged = client_headers_for_upstream(
        {
            "Content-Type": "application/json",
            "Content-Encoding": "zstd",
            "session-id": "sess-1",
            "originator": "Codex Desktop",
        }
    )
    assert "Content-Encoding" not in merged
    assert merged["session-id"] == "sess-1"
    assert merged["originator"] == "Codex Desktop"


def test_chatgpt_passthrough_upstream_headers_drop_content_encoding():
    merged = chatgpt_passthrough_upstream_headers(
        {
            "Content-Encoding": "zstd",
            "originator": "Codex Desktop",
            "x-codex-beta-features": "memories,prevent_idle_sleep,remote_compaction_v2",
        },
        access_token="token",
        account_id="acct",
        accept="text/event-stream",
    )
    assert "Content-Encoding" not in merged
    assert merged["Authorization"] == "Bearer token"
    assert merged["originator"] == "Codex Desktop"
    assert merged["x-codex-beta-features"] == "memories,prevent_idle_sleep,remote_compaction_v2"


def test_openai_upstream_headers_preserves_client_accept_encoding():
    merged = openai_upstream_headers(
        {"Accept-Encoding": "zstd, gzip"},
        api_key="sk-test",
    )
    assert merged["Accept-Encoding"] == "zstd, gzip"
    assert merged["Authorization"] == "Bearer sk-test"


def test_log_upstream_response_headers_includes_usage(monkeypatch, capsys):
    monkeypatch.setenv("CODEX_SHIM_UPSTREAM_HEADER_LOG", "1")
    log_upstream_response_headers(
        "chatgpt-passthrough",
        {"x-request-id": "req_1", "openai-processing-ms": "9"},
        usage={
            "input_tokens": 100,
            "output_tokens": 20,
            "input_tokens_details": {"cached_tokens": 80},
        },
    )
    out = capsys.readouterr().out
    assert "[upstream-headers]" in out
    assert "x-request-id" in out
    assert "cached_tokens" in out or "_cached_tokens" in out

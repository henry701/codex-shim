from __future__ import annotations

import re

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


def test_chatgpt_passthrough_upgrades_stale_desktop_client_identity(monkeypatch):
    """gpt-6-luna is rejected for ChatGPT accounts when Desktop still sends CLI 0.153.4.

    The system CLI that the shim can see is newer. Upstream must see that version
    and the codex_cli_rs originator, which is the identity that accepts the model.
    """
    monkeypatch.setattr(
        "codex_shim.header_passthrough.chatgpt_client_version_for_upstream",
        lambda: "0.156.0",
    )
    merged = chatgpt_passthrough_upstream_headers(
        {
            "User-Agent": "codex_cli_rs/0.153.4 (Ubuntu; x86_64)",
            "originator": "Codex Desktop",
            "x-codex-beta-features": "memories",
        },
        access_token="token",
        account_id="acct",
        accept="text/event-stream",
    )
    assert merged["User-Agent"] == "codex_cli_rs/0.156.0"
    assert merged["originator"] == "codex_cli_rs"
    assert merged["x-codex-beta-features"] == "memories"


def test_chatgpt_passthrough_keeps_current_client_identity(monkeypatch):
    monkeypatch.setattr(
        "codex_shim.header_passthrough.chatgpt_client_version_for_upstream",
        lambda: "0.156.0",
    )
    merged = chatgpt_passthrough_upstream_headers(
        {
            "User-Agent": "codex_cli_rs/0.156.0",
            "originator": "Codex Desktop",
        },
        access_token="token",
        account_id="acct",
        accept="application/json",
    )
    assert merged["User-Agent"] == "codex_cli_rs/0.156.0"
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


_HERMES_FREE_HEADERS = {
    "HTTP-Referer": "https://hermes-agent.nousresearch.com",
    "X-Title": "Hermes Agent",
}


def test_openai_upstream_headers_strips_authorization_for_public_sentinel():
    merged = openai_upstream_headers(
        {
            "User-Agent": "ChatGPT/1.0",
            "Authorization": "Bearer chatgpt-token",
        },
        api_key="public",
        extra_headers=_HERMES_FREE_HEADERS,
    )
    assert not any(key.lower() == "authorization" for key in merged)
    assert "Bearer public" not in merged.values()
    assert "Bearer chatgpt-token" not in merged.values()
    assert merged["User-Agent"] == "ChatGPT/1.0"
    assert merged["HTTP-Referer"] == "https://hermes-agent.nousresearch.com"
    assert merged["X-Title"] == "Hermes Agent"


def test_openai_upstream_headers_public_keeps_desktop_referer():
    merged = openai_upstream_headers(
        {
            "User-Agent": "ChatGPT/Desktop",
            "HTTP-Referer": "https://chatgpt.com/",
        },
        api_key="public",
        extra_headers=_HERMES_FREE_HEADERS,
    )
    assert not any(key.lower() == "authorization" for key in merged)
    assert merged["User-Agent"] == "ChatGPT/Desktop"
    assert merged["HTTP-Referer"] == "https://chatgpt.com/"
    assert merged["X-Title"] == "Hermes Agent"


def test_opencode_free_headers_replace_desktop_identity():
    from types import SimpleNamespace

    from codex_shim.header_passthrough import (
        OPENCODE_FREE_USER_AGENT,
        apply_opencode_free_client_headers,
        ensure_opencode_free_chat_tools,
        is_opencode_free_route,
        opencode_free_inference_base,
        opencode_free_session_id,
    )
    from codex_shim.server import _openai_headers, _route_base

    route = SimpleNamespace(
        slug="oc-free-beam-free",
        model="beam-free",
        base_url="https://opencode.ai/zen/v1",
        api_key="public",
        extra_headers={"HTTP-Referer": "https://hermes-agent.nousresearch.com", "X-Title": "Hermes Agent"},
    )
    assert is_opencode_free_route(route)
    assert not is_opencode_free_route(
        SimpleNamespace(slug="zen-glm", model="glm", base_url="https://opencode.ai/zen/go/v1", api_key="sk", extra_headers={})
    )
    assert opencode_free_inference_base(route.base_url) == "https://opencode.ai/inference/v1"
    assert opencode_free_inference_base("https://opencode.ai/zen/go/v1") == "https://opencode.ai/zen/go/v1"
    assert _route_base(route) == "https://opencode.ai/inference/v1"

    merged = _openai_headers(
        {
            "User-Agent": "codex_cli_rs/0.160.0",
            "Authorization": "Bearer chatgpt-token",
            "x-opencode-request": "msg_old",
            "x-session-affinity": "desktop-session",
            "session-id": "01a12c76-thread",
        },
        route,
        accept="text/event-stream",
    )
    assert merged["User-Agent"] == OPENCODE_FREE_USER_AGENT
    assert merged["Authorization"] == "Bearer public"
    assert merged["x-opencode-client"] == "cli"
    assert merged["x-opencode-project"] == "global"
    session = merged["x-opencode-session"]
    assert session == merged["x-opencode-session-id"] == merged["x-session-affinity"] == merged["x-session-id"]
    assert session == opencode_free_session_id("01a12c76-thread")
    assert re.fullmatch(r"ses_[0-9a-f]{9}f[c-f][0-9a-f][0-9A-Za-z]{14}", session)
    assert not any(key.lower() == "x-opencode-request" for key in merged)
    again = apply_opencode_free_client_headers({}, session_signal="01a12c76-thread")
    assert again["x-opencode-session"] == session

    body = ensure_opencode_free_chat_tools(
        {"model": "beam-free", "tools": [{"type": "function", "function": {"name": "exec_command"}}], "tool_choice": "none"}
    )
    names = [tool["function"]["name"] for tool in body["tools"]]
    assert names == ["exec_command", "read", "shell"]
    assert "tool_choice" not in body
    assert body["tools"][1]["function"]["parameters"]["required"] == ["path"]
    assert body["tools"][2]["function"]["parameters"]["required"] == ["command"]


def test_opencode_free_user_agent_tracks_installed_cli_version(monkeypatch):
    from codex_shim.header_passthrough import opencode_free_user_agent

    monkeypatch.delenv("CODEX_SHIM_OPENCODE_FREE_CLIENT_VERSION", raising=False)
    monkeypatch.setattr(
        "codex_shim.discover.detect_opencode_cli_version", lambda **_kwargs: "2.3.0"
    )
    assert opencode_free_user_agent() == "opencode/latest/2.3.0/cli"


def test_opencode_free_user_agent_env_override_wins(monkeypatch):
    from codex_shim.header_passthrough import opencode_free_user_agent

    monkeypatch.setenv("CODEX_SHIM_OPENCODE_FREE_CLIENT_VERSION", "9.9.9")
    monkeypatch.setattr(
        "codex_shim.discover.detect_opencode_cli_version", lambda **_kwargs: "2.3.0"
    )
    assert opencode_free_user_agent() == "opencode/latest/9.9.9/cli"


def test_opencode_free_user_agent_falls_back_without_cli(monkeypatch):
    from codex_shim.header_passthrough import (
        OPENCODE_FREE_USER_AGENT,
        opencode_free_user_agent,
    )

    monkeypatch.delenv("CODEX_SHIM_OPENCODE_FREE_CLIENT_VERSION", raising=False)
    monkeypatch.setattr(
        "codex_shim.discover.detect_opencode_cli_version", lambda **_kwargs: None
    )
    assert opencode_free_user_agent() == OPENCODE_FREE_USER_AGENT


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

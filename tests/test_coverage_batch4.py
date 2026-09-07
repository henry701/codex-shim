from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError, URLError

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer, make_mocked_request

from codex_shim import cli
from codex_shim.compaction.errors import parse_upstream_error_detail
from codex_shim.compaction.protocol import CompactionTriggerError, strip_terminal_compaction_trigger
from codex_shim.cursor_bridge import (
    CursorBridgeSession,
    _item_text,
    cursor_bridge_registry,
)
from codex_shim.cursor_passthrough import (
    CursorStreamParser,
    _format_grep_result,
    _format_shell_started,
    _load_cursor_catalog_models,
    build_cursor_prompt,
    iter_cursor_agent_events,
)
from codex_shim.cursor_stream_visualizer import format_event_line
from codex_shim.discover import (
    _load_codex_auth_tokens,
    discover_chatgpt_model_ids_from_openai_api,
    fetch_chatgpt_codex_backend_models,
    list_opencode_cli_models,
    persist_chatgpt_models_cache,
    refresh_codex_auth_tokens,
)
from codex_shim.mcp_search import augment_response_with_tool_search
from codex_shim.net.errors import parse_resets_in_seconds
from codex_shim.net.retry import HttpPostResult
from codex_shim.nous_auth import _atomic_write_json_pair
from codex_shim.quota_dashboard import _credit_rates
from codex_shim.server import ShimServer, _write_ws_error
from codex_shim.settings import (
    ShimModel,
    _load_chatgpt_cache_catalog_models,
    chatgpt_passthrough_available,
)
from codex_shim.translate import (
    apply_session_title_candidate,
    omit_hosted_codex_tools,
    original_responses_tool_type,
    prepare_codex_byok_responses_body,
    responses_tool_resolve_map,
    SessionTitleCandidate,
)
from codex_shim.ws_passthrough import _upstream_lane_reusable


class _FakeWs:
    def __init__(self) -> None:
        self.sent: list[str] = []

    async def send_str(self, payload: str) -> None:
        self.sent.append(payload)


def _settings(tmp_path: Path) -> Path:
    path = tmp_path / "models.json"
    path.write_text("{}")
    return path


def test_discover_auth_cache_and_cli_models(monkeypatch, tmp_path):
    missing = tmp_path / "missing.json"
    assert _load_codex_auth_tokens(missing) is None
    bad = tmp_path / "bad.json"
    bad.write_text("{")
    assert _load_codex_auth_tokens(bad) is None
    not_obj = tmp_path / "list.json"
    not_obj.write_text("[]")
    assert _load_codex_auth_tokens(not_obj) is None
    empty_tokens = tmp_path / "empty.json"
    empty_tokens.write_text(json.dumps({"tokens": {}}))
    assert _load_codex_auth_tokens(empty_tokens) is None
    auth = tmp_path / "auth.json"
    auth.write_text(json.dumps({"tokens": {"access_token": "tok", "account_id": "acct", "refresh_token": "rt"}}))
    assert _load_codex_auth_tokens(auth) == ("tok", "acct")

    assert refresh_codex_auth_tokens(missing) is False
    assert refresh_codex_auth_tokens(bad) is False
    assert refresh_codex_auth_tokens(not_obj) is False
    no_refresh = tmp_path / "no-rt.json"
    no_refresh.write_text(json.dumps({"tokens": {"access_token": "tok"}}))
    assert refresh_codex_auth_tokens(no_refresh) is False

    class Result:
        body = b'{"access_token":"new","refresh_token":"rt2","id_token":"id"}'

    monkeypatch.setattr("codex_shim.discover.request_urllib", lambda *a, **k: Result())
    assert refresh_codex_auth_tokens(auth) is True
    stored = json.loads(auth.read_text())
    assert stored["tokens"]["access_token"] == "new"
    assert stored["tokens"]["refresh_token"] == "rt2"

    monkeypatch.setattr("codex_shim.discover.request_urllib", lambda *a, **k: (_ for _ in ()).throw(URLError("down")))
    assert refresh_codex_auth_tokens(auth) is False

    class EmptyToken:
        body = b'{"access_token":""}'

    monkeypatch.setattr("codex_shim.discover.request_urllib", lambda *a, **k: EmptyToken())
    assert refresh_codex_auth_tokens(auth) is False

    class NotDict:
        body = b"[1]"

    monkeypatch.setattr("codex_shim.discover.request_urllib", lambda *a, **k: NotDict())
    assert refresh_codex_auth_tokens(auth) is False

    monkeypatch.setattr("codex_shim.discover.request_urllib", lambda *a, **k: Result())
    auth.chmod(0o444)
    try:
        assert refresh_codex_auth_tokens(auth) is False
    finally:
        auth.chmod(0o644)

    assert persist_chatgpt_models_cache([]) is None
    cache = tmp_path / "models_cache.json"
    cache.write_text("{")
    path = persist_chatgpt_models_cache([{"slug": "gpt-5.6-luna"}], cache)
    assert path == cache
    blocked_parent = tmp_path / "blocked"
    blocked_parent.write_text("not-a-dir")
    assert persist_chatgpt_models_cache([{"slug": "gpt-x"}], blocked_parent / "models_cache.json") is None

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert discover_chatgpt_model_ids_from_openai_api() == []
    monkeypatch.setenv("OPENAI_API_KEY", "sk")
    monkeypatch.setattr(
        "codex_shim.discover.fetch_http_json",
        lambda *a, **k: {"data": [{"id": "gpt-5.6-luna"}, {"id": "skip-me"}]},
    )
    assert "gpt-5.6-luna" in discover_chatgpt_model_ids_from_openai_api()
    monkeypatch.setattr(
        "codex_shim.discover.fetch_http_json",
        lambda *a, **k: (_ for _ in ()).throw(URLError("down")),
    )
    assert discover_chatgpt_model_ids_from_openai_api() == []

    from codex_shim.discover import clear_opencode_cli_models_cache

    monkeypatch.setattr("codex_shim.discover.shutil.which", lambda name: None)
    assert list_opencode_cli_models() == []
    clear_opencode_cli_models_cache()
    monkeypatch.setattr("codex_shim.discover.shutil.which", lambda name: "/usr/bin/opencode")
    monkeypatch.setattr(
        "codex_shim.discover.subprocess.run",
        lambda *a, **k: SimpleNamespace(returncode=0, stdout="opencode/foo\n", stderr=""),
    )
    assert "opencode/foo" in list_opencode_cli_models()


def test_fetch_chatgpt_codex_backend_models_auth_refresh(monkeypatch, tmp_path):
    auth = tmp_path / "auth.json"
    auth.write_text(json.dumps({"tokens": {"access_token": "tok", "account_id": "acct"}}))
    monkeypatch.setattr("codex_shim.discover._load_codex_auth_tokens", lambda path=None: None)
    assert fetch_chatgpt_codex_backend_models(auth_path=auth) == []

    calls = {"n": 0}

    def load(path=None):
        return ("tok", "acct")

    def fetch(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise HTTPError("https://x", 401, "nope", hdrs=None, fp=BytesIO(b""))
        return {"models": [{"slug": "gpt-5.6-luna"}]}

    monkeypatch.setattr("codex_shim.discover._load_codex_auth_tokens", load)
    monkeypatch.setattr("codex_shim.discover.fetch_http_json", fetch)
    monkeypatch.setattr("codex_shim.discover.refresh_codex_auth_tokens", lambda *a, **k: True)
    rows = fetch_chatgpt_codex_backend_models(auth_path=auth)
    assert rows[0]["slug"] == "gpt-5.6-luna"

    calls["n"] = 0

    def fetch_fail(*a, **k):
        calls["n"] += 1
        raise HTTPError("https://x", 401, "nope", hdrs=None, fp=BytesIO(b""))

    monkeypatch.setattr("codex_shim.discover.fetch_http_json", fetch_fail)
    assert fetch_chatgpt_codex_backend_models(auth_path=auth) == []

    monkeypatch.setattr(
        "codex_shim.discover.fetch_http_json",
        lambda *a, **k: (_ for _ in ()).throw(URLError("down")),
    )
    assert fetch_chatgpt_codex_backend_models(auth_path=auth) == []


def test_cursor_passthrough_prompt_events_catalog(monkeypatch):
    prompt = build_cursor_prompt(
        {
            "model": "cursor-auto",
            "instructions": "sys",
            "input": [
                {"role": "user", "content": "hello"},
                {
                    "type": "function_call",
                    "call_id": "c1",
                    "name": "shell",
                    "arguments": "{}",
                },
                {"type": "function_call_output", "call_id": "c1", "output": "ok"},
            ],
        }
    )
    assert "[SYSTEM]" in prompt or "[USER]" in prompt
    assert "shell" in prompt or "[TOOL" in prompt

    parser = CursorStreamParser()
    assert parser.feed_events("") == []
    assert parser.feed_events("{") == []
    assert parser.feed_events("[]") == []
    assert parser.feed_events(
        json.dumps(
            {
                "type": "assistant",
                "timestamp_ms": 1,
                "message": {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
            }
        )
    )
    assert parser.feed_events(
        json.dumps(
            {
                "type": "assistant",
                "model_call_id": "seg",
                "message": {"role": "assistant", "content": [{"type": "text", "text": ""}]},
            }
        )
    )
    assert parser.feed_events(json.dumps({"type": "tool_call", "subtype": "started", "call_id": "t1", "tool_call": {"shellToolCall": {"args": {"command": "ls"}}}}))
    assert parser.feed_events(json.dumps({"type": "tool_call", "subtype": "completed", "call_id": "t1", "tool_call": {}}))
    assert parser.feed_events(json.dumps({"type": "tool_call", "subtype": "other"})) == []
    assert parser.feed_events(json.dumps({"type": "thinking", "subtype": "delta", "text": "hmm"}))
    assert parser.feed_events(json.dumps({"type": "thinking", "subtype": "completed"}))
    assert parser.feed_events(json.dumps({"type": "thinking", "text": "more"}))
    assert parser.feed_events(json.dumps({"type": "connection", "subtype": "reconnecting"}))
    parser.feed_events(json.dumps({"type": "result", "subtype": "error", "result": "boom"}))
    assert parser.error
    parser2 = CursorStreamParser()
    assert parser2.feed_events(json.dumps({"type": "result", "result": "done", "usage": {"inputTokens": 1, "outputTokens": 2}})) == []
    assert parser2.final_text == "done"
    assert parser2.feed_events(json.dumps({"type": "error", "message": "nope"})) == []

    assert _format_grep_result({}) == ""
    assert _format_grep_result({"result": {"success": {"workspaceResults": {"/": "skip"}}}}) == ""
    formatted = _format_grep_result(
        {
            "result": {
                "success": {
                    "workspaceResults": {
                        "/": {
                            "content": {
                                "matches": [
                                    {"file": "a.py", "matches": [{"lineNumber": 3, "content": "needle"}]},
                                    {"file": "b.py", "matches": ["raw"]},
                                    {"file": "c.py"},
                                    "skip",
                                ]
                            }
                        },
                        "/other": "skip",
                    }
                }
            }
        }
    )
    assert "a.py:3" in formatted
    assert _format_grep_result({"result": {"success": {"pattern": "zzz"}}}) == "pattern `zzz` — no matches"
    assert "ls" in _format_shell_started({"args": {"command": "ls -la"}}) or _format_shell_started({"command": "ls"}) != ""

    import codex_shim.cursor_passthrough as cp

    cp._models_cache = None
    monkeypatch.setattr(cp.shutil, "which", lambda name: None)
    monkeypatch.delenv("CURSOR_AGENT_BIN", raising=False)
    models = _load_cursor_catalog_models(force_refresh=True)
    assert models
    models2 = _load_cursor_catalog_models()
    assert models2 is models or models2
    monkeypatch.setattr(cp.shutil, "which", lambda name: "/bin/cursor-agent")
    monkeypatch.setattr(
        cp.subprocess,
        "run",
        lambda *a, **k: (_ for _ in ()).throw(cp.subprocess.TimeoutExpired("cursor-agent", 30)),
    )
    fallback = _load_cursor_catalog_models(force_refresh=True)
    assert fallback
    cp._models_cache = None


async def test_iter_cursor_agent_events_fake_proc(monkeypatch):
    class FakeStdin:
        def write(self, data):
            del data

        async def drain(self):
            return None

        def close(self):
            return None

    class FakeStream:
        def __init__(self, chunks):
            self._chunks = list(chunks)

        async def read(self, n):
            del n
            if not self._chunks:
                return b""
            return self._chunks.pop(0)

    class FakeProc:
        def __init__(self):
            self.stdin = FakeStdin()
            self.stdout = FakeStream([b'{"type":"result","result":"hi","usage":{"inputTokens":1}}\n'])
            self.stderr = FakeStream([b""])
            self.returncode = None

        def kill(self):
            self.returncode = -9

        async def wait(self):
            self.returncode = 0

    async def spawn(*a, **k):
        del a, k
        return FakeProc()

    monkeypatch.setattr("codex_shim.cursor_passthrough.asyncio.create_subprocess_exec", spawn)
    monkeypatch.setattr("codex_shim.cursor_passthrough._cursor_agent_bin", lambda: "cursor-agent")
    monkeypatch.setattr("codex_shim.cursor_passthrough.cursor_workspace", lambda: "/tmp")
    events = [event async for event in iter_cursor_agent_events("hi", "auto")]
    types = [event.get("type") for event in events]
    assert "completed" in types
    assert "usage" in types


async def test_cursor_bridge_poll_ingest_and_delivery():
    session = CursorBridgeSession.create(
        allowed_tools=frozenset({"shell"}),
        tool_types={},
        tool_resolve={},
    )
    idle = await session.poll_jobs(timeout_s=0)
    assert idle["idle"] is True
    session.attach_collector(
        SimpleNamespace(append_function_call=lambda **kwargs: None)
    )
    await cursor_bridge_registry.register(session)
    try:
        accepted = await session.invoke(tool="shell", arguments={"command": "echo"})
        call_id = accepted["codex_call_id"]
        ingested = cursor_bridge_registry.ingest_function_call_outputs(
            [{"type": "function_call_output", "call_id": call_id, "output": "ok"}, "skip", {"type": "message"}]
        )
        assert ingested == 1
        polled = await session.poll_jobs(timeout_s=0.1)
        assert polled["jobs"]
        session.mark_turn_closed()
        session.append_post_terminal_text(" leftover ")
        session.mark_passthrough_finished()
        leftover = await cursor_bridge_registry.wait_for_delivery_passthroughs(timeout_s=1)
        assert "leftover" in leftover
        assert _item_text("x") == "x"
        assert _item_text({"text": "y"}) == "y"
        assert _item_text([{"text": "a"}, {"output_text": "b"}]) == "ab"
        assert _item_text(12) == ""
    finally:
        cursor_bridge_registry.close(session.bridge_id)


async def test_mcp_search_augment_and_ws_compaction(monkeypatch, tmp_path):
    empty = await augment_response_with_tool_search({"output": []})
    assert empty["output"] == []
    mixed = await augment_response_with_tool_search(
        {
            "output": [
                "skip",
                {"type": "message", "content": "x"},
                {"type": "function_call", "name": "mcp__exa__search", "call_id": "c1", "arguments": {"q": "a"}},
                {"type": "function_call", "name": "tool_search", "call_id": "c2", "arguments": "{}"},
                {"type": "function_call", "name": "shell", "call_id": "c3", "arguments": "{}"},
            ]
        }
    )
    types = [item.get("type") if isinstance(item, dict) else item for item in mixed["output"]]
    assert "tool_search_call" in types or "mcp_tool_call" in types or "function_call" in types

    shim = ShimServer(_settings(tmp_path))
    ws = _FakeWs()
    monkeypatch.setattr("codex_shim.server.is_chatgpt_passthrough_slug", lambda model: True)
    assert await shim._maybe_handle_ws_compaction_v2(make_mocked_request("GET", "/ws"), ws, {"model": "codex-gpt-5-6-luna"}) is False
    monkeypatch.setattr("codex_shim.server.is_chatgpt_passthrough_slug", lambda model: False)
    assert await shim._maybe_handle_ws_compaction_v2(make_mocked_request("GET", "/ws"), ws, {"model": "local", "input": []}) is False
    with pytest.raises(CompactionTriggerError):
        strip_terminal_compaction_trigger(
            [{"type": "compaction_trigger"}, {"type": "message"}, {"type": "compaction_trigger"}]
        )
    assert strip_terminal_compaction_trigger("nope") is None
    stripped = strip_terminal_compaction_trigger([{"role": "user"}, {"type": "compaction_trigger"}])
    assert stripped == [{"role": "user"}]

    async def boom_strip(items):
        del items
        raise CompactionTriggerError("bad trigger")

    monkeypatch.setattr("codex_shim.server.strip_terminal_compaction_trigger", lambda items: (_ for _ in ()).throw(CompactionTriggerError("bad trigger")))
    handled = await shim._maybe_handle_ws_compaction_v2(
        make_mocked_request("GET", "/ws"),
        ws,
        {"model": "local", "input": [{"type": "compaction_trigger"}]},
    )
    assert handled is True
    assert any("error" in payload for payload in ws.sent)

    async def resolve_ok(*a, **k):
        return ({"id": "cmp_1", "encrypted_content": "blob"}, "local", {"input_tokens": 1}, None)

    monkeypatch.setattr("codex_shim.server.strip_terminal_compaction_trigger", lambda items: list(items or []))
    monkeypatch.setattr(ShimServer, "_resolve_compaction_v2_result", resolve_ok)
    ws2 = _FakeWs()
    handled_ok = await shim._maybe_handle_ws_compaction_v2(
        make_mocked_request("GET", "/ws"),
        ws2,
        {"model": "local", "input": [{"role": "user"}]},
    )
    assert handled_ok is True

    async def resolve_err(*a, **k):
        return ({}, "local", None, web.Response(text='{"error":{"message":"nope"}}', status=400))

    monkeypatch.setattr(ShimServer, "_resolve_compaction_v2_result", resolve_err)
    ws3 = _FakeWs()
    handled_err = await shim._maybe_handle_ws_compaction_v2(
        make_mocked_request("GET", "/ws"),
        ws3,
        {"model": "local", "input": [{"role": "user"}]},
    )
    assert handled_err is True


async def test_handle_response_create_websocket_missing_auth(monkeypatch, tmp_path):
    auth = tmp_path / "auth.json"
    monkeypatch.setattr("codex_shim.server.DEFAULT_CODEX_AUTH", auth)
    monkeypatch.setattr("codex_shim.settings.DEFAULT_CODEX_AUTH", auth)
    shim = ShimServer(_settings(tmp_path))
    ws = _FakeWs()

    async def no_target(self, payload):
        del payload
        return None

    monkeypatch.setattr(ShimServer, "_resolve_ws_passthrough_target", no_target)
    monkeypatch.setattr("codex_shim.server.is_chatgpt_passthrough_slug", lambda model: True)
    monkeypatch.setattr("codex_shim.server.ws_passthrough_enabled", lambda: False)
    await shim._handle_response_create_websocket(
        make_mocked_request("GET", "/ws"),
        ws,
        {"model": "codex-gpt-5-6-luna"},
    )
    assert any("auth.json not found" in payload for payload in ws.sent)

    auth.write_text("{")
    ws2 = _FakeWs()
    await shim._handle_response_create_websocket(make_mocked_request("GET", "/ws"), ws2, {"model": "codex-gpt-5-6-luna"})
    assert any("not valid JSON" in payload for payload in ws2.sent)

    auth.write_text(json.dumps({"tokens": {}}))
    ws3 = _FakeWs()
    await shim._handle_response_create_websocket(make_mocked_request("GET", "/ws"), ws3, {"model": "codex-gpt-5-6-luna"})
    assert any("no access_token" in payload for payload in ws3.sent)

    await _write_ws_error(ws3, 502, "upstream_error", "boom")


async def test_post_openai_responses_json_error_and_ok(monkeypatch, tmp_path):
    shim = ShimServer(_settings(tmp_path))
    route = ShimModel(
        slug="console-model",
        model="gpt-x",
        display_name="Console",
        provider="openai-responses",
        base_url="http://example.invalid/v1",
        api_key="secret",
    )
    request = make_mocked_request("POST", "/v1/responses")

    async def fail_post(*a, **k):
        return HttpPostResult(
            response=SimpleNamespace(headers={}, release=lambda: None),
            status=400,
            content_type="application/json",
            error_text='{"error":{"message":"bad"}}',
        )

    monkeypatch.setattr("codex_shim.server.retry_aiohttp_post", fail_post)
    error = await shim._post_openai_responses(request, route, {"model": "console-model", "input": []})
    assert error.status == 400

    class OkUpstream:
        headers = {}

        async def json(self, content_type=None):
            del content_type
            return {"id": "resp_1", "output": [], "usage": {"input_tokens": 1}}

        def release(self):
            return None

    async def ok_post(*a, **k):
        return HttpPostResult(response=OkUpstream(), status=200, content_type="application/json")

    monkeypatch.setattr("codex_shim.server.retry_aiohttp_post", ok_post)
    ok = await shim._post_openai_responses(request, route, {"model": "console-model", "input": []})
    assert ok.status == 200


async def test_cursor_bridge_http_validation(tmp_path):
    shim = ShimServer(_settings(tmp_path))
    client = TestClient(TestServer(shim.app()))
    await client.start_server()
    try:
        bad_json = await client.post(
            "/_cursor_bridge/v1/invoke",
            data="{",
            headers={"Host": "127.0.0.1", "Content-Type": "application/json"},
        )
        assert bad_json.status == 400
        not_obj = await client.post("/_cursor_bridge/v1/invoke", json=[], headers={"Host": "127.0.0.1"})
        assert not_obj.status == 400
        missing_bridge = await client.post(
            "/_cursor_bridge/v1/invoke",
            json={"tool": "shell", "arguments": {}},
            headers={"Host": "127.0.0.1"},
        )
        assert missing_bridge.status == 400
        missing_tool = await client.post(
            "/_cursor_bridge/v1/invoke",
            json={"bridge": "x", "arguments": {}},
            headers={"Host": "127.0.0.1"},
        )
        assert missing_tool.status == 400
        missing_args = await client.post(
            "/_cursor_bridge/v1/invoke",
            json={"bridge": "x", "tool": "shell", "arguments": "nope"},
            headers={"Host": "127.0.0.1"},
        )
        assert missing_args.status == 400
        wait_missing = await client.post(
            "/_cursor_bridge/v1/wait",
            json={"bridge": "x"},
            headers={"Host": "127.0.0.1"},
        )
        assert wait_missing.status == 400
        poll_bad_timeout = await client.post(
            "/_cursor_bridge/v1/poll",
            json={"bridge": "x", "timeout_ms": "nope"},
            headers={"Host": "127.0.0.1"},
        )
        assert poll_bad_timeout.status == 400
        poll_missing = await client.post(
            "/_cursor_bridge/v1/poll",
            json={},
            headers={"Host": "127.0.0.1"},
        )
        assert poll_missing.status == 400
    finally:
        await client.close()


def test_server_needs_image_gen(tmp_path):
    shim = ShimServer(_settings(tmp_path))
    assert shim._needs_image_gen({"tools": ["skip", {"type": "image_generation", "name": "image_gen"}]}) is False
    assert shim._needs_image_gen({"tools": [{"type": "image_generation", "name": "image_gen"}]}) is True
    assert shim._needs_image_gen(
        {
            "tools": [
                {"type": "function", "function": {"name": "image_gen"}},
                {"type": "function", "name": "shell"},
            ],
            "tool_choice": "image_gen",
            "input": [{"role": "user", "content": "please imagegen a cat"}],
        }
    ) is True
    assert shim._needs_image_gen({"tools": [{"type": "function", "name": "shell"}]}) is False
    assert shim._needs_image_gen(
        {
            "tools": [{"type": "image_generation"}],
            "tool_choice": {"type": "image_generation"},
        }
    ) is True


def test_helpers_settings_cli_errors_visualizer(monkeypatch, tmp_path):
    assert parse_upstream_error_detail(SimpleNamespace(status=502, text="")).message.startswith("Upstream")
    assert "not-json" in parse_upstream_error_detail(SimpleNamespace(status=400, text="not-json")).message
    assert parse_upstream_error_detail(SimpleNamespace(status=400, text="[]")).raw_body == "[]"
    nested = parse_upstream_error_detail(
        SimpleNamespace(
            status=400,
            text=json.dumps(
                {
                    "error": {"message": "inner", "code": "bad", "type": "invalid", "param": "model", "hint": "x"},
                    "message": "top",
                    "code": "top-code",
                    "detail": ["a", "b"],
                    "request_id": "r1",
                }
            ),
        )
    )
    assert nested.message == "a; b"
    assert nested.code == "top-code"
    string_err = parse_upstream_error_detail(SimpleNamespace(status=400, text=json.dumps({"error": "plain"})))
    assert string_err.message == "plain"

    assert parse_resets_in_seconds('{"resets_in_seconds": 12}') == 12
    assert parse_resets_in_seconds('{"resets_at": 1}') is not None
    assert parse_resets_in_seconds('{"error":{"resets_at":"not-a-date"}}') is None
    assert parse_resets_in_seconds('{"resets_at":"2099-01-01T00:00:00Z"}') > 0
    assert parse_resets_in_seconds("{}") is None

    assert _upstream_lane_reusable(SimpleNamespace(closed=True)) is False
    assert _upstream_lane_reusable(SimpleNamespace(closed=False)) is True
    assert _upstream_lane_reusable(SimpleNamespace(closed=False, exception=lambda: (_ for _ in ()).throw(RuntimeError("x")))) is False
    assert _upstream_lane_reusable(SimpleNamespace(closed=False, exception=lambda: None)) is True

    class AwaitableExc:
        def close(self):
            return None

        def __await__(self):
            async def _inner():
                return None

            return _inner().__await__()

    assert _upstream_lane_reusable(SimpleNamespace(closed=False, exception=lambda: AwaitableExc())) is True

    assert format_event_line({"type": "text_delta", "delta": "hi"})
    assert format_event_line({"type": "thinking_delta", "delta": "x"})
    assert format_event_line({"type": "thinking_completed"})
    assert format_event_line({"type": "tool_started", "tool_call": {"unknownTool": {}}, "markdown": "**x**\n\n> detail\n{}"})
    assert format_event_line({"type": "tool_completed"})
    assert format_event_line({"type": "segment_boundary"})
    assert format_event_line({"type": "connection_interrupted"})
    assert format_event_line({"type": "other", "x": 1})

    assert _credit_rates("codex-gpt-5-6-luna")
    assert _credit_rates("gpt-5-6-sol-preview")
    assert _credit_rates("gpt-5-6-terra")
    assert _credit_rates("something-5-6-luna")
    assert _credit_rates("gpt-5-5-mini")
    assert _credit_rates("unknown-model")

    monkeypatch.setenv("CODEX_SHIM_DISABLE_CHATGPT", "1")
    assert chatgpt_passthrough_available() is False
    monkeypatch.delenv("CODEX_SHIM_DISABLE_CHATGPT", raising=False)
    missing = tmp_path / "no-auth.json"
    assert chatgpt_passthrough_available(missing) is False
    bad = tmp_path / "bad-auth.json"
    bad.write_text("{")
    assert chatgpt_passthrough_available(bad) is False
    listed = tmp_path / "auth.json"
    listed.write_text("[]")
    assert chatgpt_passthrough_available(listed) is False
    cache = tmp_path / "cache.json"
    cache.write_text("{")
    assert _load_chatgpt_cache_catalog_models(cache) == []
    cache.write_text("[]")
    assert _load_chatgpt_cache_catalog_models(cache) == []
    cache.write_text(json.dumps({"models": [{"slug": "gpt-5.6-luna"}, "skip"]}))
    assert _load_chatgpt_cache_catalog_models(cache)

    candidate = apply_session_title_candidate({"model": "x"}, SessionTitleCandidate("gpt-5.6-luna", reasoning_effort="low"))
    assert candidate["model"] == "gpt-5.6-luna"
    assert candidate["reasoning"]["effort"] == "low"
    assert original_responses_tool_type("apply_patch") == "apply_patch"
    assert original_responses_tool_type("x", {"x": "custom"}) == "custom"
    assert original_responses_tool_type("web_search") == "web_search"
    assert omit_hosted_codex_tools("nope") == []
    assert omit_hosted_codex_tools([{"type": "web_search"}, {"name": "shell"}]) == [{"name": "shell"}]
    prepared = prepare_codex_byok_responses_body(
        {"tools": [{"type": "web_search"}], "tool_choice": {"type": "web_search"}},
        {"user-agent": "codex_cli_rs", "x-codex-turn-state": "1"},
    )
    assert prepared.get("tools") == []
    resolved = responses_tool_resolve_map(
        [
            "skip",
            {
                "type": "namespace",
                "name": "goals",
                "tools": [{"type": "function", "name": "update_goal"}, {"type": "other"}],
            },
            {"name": "shell"},
        ]
    )
    assert resolved

    monkeypatch.setattr(cli.shutil, "which", lambda name: None)
    assert cli._doctor_codex_cli()[0].status == "WARN"
    monkeypatch.setattr(cli.shutil, "which", lambda name: "/usr/bin/codex")
    monkeypatch.setattr(
        cli.subprocess,
        "run",
        lambda *a, **k: (_ for _ in ()).throw(__import__("subprocess").TimeoutExpired("codex", 5)),
    )
    assert any(check.status == "WARN" for check in cli._doctor_codex_cli())
    monkeypatch.setattr(
        cli.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=0, stdout="codex-cli 0.1.0\n", stderr=""),
    )
    assert any("version" in check.message for check in cli._doctor_codex_cli())

    monkeypatch.setattr(cli, "_env_flag", lambda name: True)
    assert cli._doctor_chatgpt()[0].status == "INFO"
    assert cli._doctor_cursor()[0].status == "INFO"
    monkeypatch.setattr(cli, "_env_flag", lambda name: False)
    monkeypatch.setattr(cli, "chatgpt_passthrough_available", lambda: False)
    assert cli._doctor_chatgpt()[0].status == "WARN"
    monkeypatch.setattr(cli, "cursor_passthrough_available", lambda: False)
    assert any(check.status == "WARN" for check in cli._doctor_cursor())
    monkeypatch.setattr(cli, "cursor_passthrough_available", lambda: True)
    monkeypatch.setattr(cli, "cursor_passthrough_display_names", lambda: {f"m{i}": "M" for i in range(10)})
    assert any("exposed models" in check.message for check in cli._doctor_cursor())

    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
    assert cli._doctor_proxy_env()[0].status == "WARN"
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost,::1")
    assert cli._doctor_proxy_env()[0].status == "OK"

    monkeypatch.setattr(cli, "_systemd_active_state", lambda: "inactive")
    monkeypatch.setattr(cli, "_read_pid", lambda: None)
    monkeypatch.setattr(cli, "_health", lambda port: None)
    checks = cli._doctor_daemon(8767)
    assert any("unavailable" in check.message for check in checks)
    monkeypatch.setattr(cli, "_read_pid", lambda: 99)
    monkeypatch.setattr(cli, "_pid_running", lambda pid: False)
    monkeypatch.setattr(cli, "_systemd_active_state", lambda: "active")
    monkeypatch.setattr(cli, "_health", lambda port: {"ok": True, "models": 3, "chatgpt_passthrough": True})
    monkeypatch.setattr(cli, "_systemd_main_pid", lambda: 12)
    monkeypatch.setattr(cli, "_listener_pid", lambda port: 12)
    checks = cli._doctor_daemon(8767)
    assert any("health ok" in check.message for check in checks)

    monkeypatch.setattr(cli, "SYSTEMD_USER_UNIT", tmp_path / "codex-shim.service")
    monkeypatch.setattr(cli, "_stop_systemd_unit_if_active", lambda: True)
    monkeypatch.setattr(cli, "_wait_for_port_free", lambda *a, **k: True)
    monkeypatch.setattr(cli, "_health", lambda port: None)
    monkeypatch.setattr(cli, "PID_PATH", tmp_path / "shim.pid")
    assert cli.stop() == 0

    monkeypatch.setattr(cli, "_stop_systemd_unit_if_active", lambda: False)
    monkeypatch.setattr(cli, "stop", lambda: 1)
    monkeypatch.setattr(cli, "DEFAULT_PORT", 8765)
    assert cli.restart(_settings(tmp_path), 8765) == 1


def test_nous_atomic_write_pair(tmp_path, monkeypatch):
    auth = tmp_path / "auth.json"
    shared = tmp_path / "shared" / "nous_auth.json"
    shared.parent.mkdir()
    _atomic_write_json_pair(auth, {"a": 1}, shared, {"b": 2})
    assert json.loads(auth.read_text()) == {"a": 1}
    assert json.loads(shared.read_text()) == {"b": 2}

    from codex_shim import nous_auth

    real_stage = nous_auth._stage_json

    def boom_stage(path, payload):
        if path == shared:
            raise OSError("nope")
        return real_stage(path, payload)

    monkeypatch.setattr("codex_shim.nous_auth._stage_json", boom_stage)
    with pytest.raises(OSError):
        _atomic_write_json_pair(auth, {"a": 2}, shared, {"b": 3})

from __future__ import annotations

import asyncio
import fcntl
import json
import os
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from aiohttp import ClientError
from aiohttp.test_utils import TestClient, TestServer, make_mocked_request

from codex_shim import mcp_search, nous_auth, quota_dashboard
from codex_shim.catalog import _reasoning_effort
from codex_shim.catalog_context import _parse_tiers, _positive_float, _positive_int
from codex_shim.chatgpt_conversation_cache import ChatgptConversationCache, parse_cache_byte_limit
from codex_shim.compaction.config import CompactionSettings, _coerce_optional_positive_int, _load_output_token_reserve
from codex_shim.compaction.context import _load_desktop_catalog_models, context_window_tokens_for_slug
from codex_shim.compaction.errors import describe_upstream_error, format_compaction_failure_detail
from codex_shim.compaction.local import (
    _collect_tool_names,
    deterministic_fallback_summary,
    estimate_item_tokens,
    is_real_user_turn,
    select_summarization_span,
    serialize_conversation,
)
from codex_shim.compaction.logging import log_compaction_cache_expansion, log_compaction_path
from codex_shim.compaction.orchestrator import (
    CompactionOrchestrator,
    CompactionOrchestratorError,
    native_item_from_payload,
)
from codex_shim.compaction.protocol import compaction_output_item
from codex_shim.compaction.pipeline import (
    _approx_token_count,
    _estimate_item_chars,
    _message_content_key,
    collapse_consecutive_duplicate_user_messages,
    extract_previous_summary,
    truncate_tool_output_chars,
)
from codex_shim.compaction.types import (
    CompactionAdapters,
    CompactionRequest,
    NativeAttemptResult,
    SummarizationAttemptResult,
)
from codex_shim.cursor_bridge import (
    _compact_parameters,
    _iter_bridge_tool_candidates,
    _truncate,
    is_bridge_denied_tool,
)
from codex_shim.cursor_passthrough import (
    _format_edit_started,
    _format_glob_result,
    _format_grep_result,
    _format_write_started,
    _is_cursor_auth_failure,
    _preview_text,
    _strip_tool_noise,
    build_cursor_prompt,
    cursor_passthrough_entries,
    format_cursor_thinking_markdown,
    iter_cursor_agent_events,
)
from codex_shim.cursor_stream_visualizer import format_event_line, main as visualize_main
from codex_shim.header_passthrough import _format_usage_body, log_upstream_response_headers
from codex_shim.net.errors import _json_object, classify_ws_event_throttle, is_rate_limit, parse_upstream_error
from codex_shim.net.prefill import ReplaySkipper, assistant_prefill_message, _iter_indexed_tools
from codex_shim.net.retry import (
    HttpPostResult,
    RetryPolicy,
    _apply_jitter,
    _clamp,
    _header_get,
    _http_date_retry_seconds,
    _on_running_event_loop,
    _parse_retry_after_value,
    retry_policy_from_env,
    throttle_sleep,
    throttle_sleep_sync,
)
from codex_shim.net.sse import ClientDisconnected, DownstreamPinger, DownstreamWriter, keepalive_interval, ping_websocket
from codex_shim.net.stream_guard import StreamGuard
from codex_shim.opencode_go import _settings_rows, display_name_from_model_id
from codex_shim.responses_input_pipeline import align_responses_tool_call_ids, synthesize_orphan_tool_calls
from codex_shim.router import RouterCandidate, RouterConfig, build_user_content, parse_scores, resolve_auto
from codex_shim.server import PICKER_TOKEN_HEADER, ShimServer, _set_active_model
from codex_shim.settings import (
    ModelSettings,
    ShimModel,
    _is_listed_gpt_model,
    _normalize_model_row,
    available_model_slugs,
    chatgpt_catalog_slug,
    chatgpt_upstream_model,
)
from codex_shim.translate import (
    _agent_message_looks_like_plaintext,
    _anthropic_content_to_chat_content,
    _anthropic_image_block_to_chat_part,
    _anthropic_tool_choice_to_chat,
    _chat_finish_to_anthropic_stop,
    _jsonish,
    _merge_chat_content,
    _native_tool_description,
    _native_tool_parameters,
    _normalize_chat_roles,
    _responses_tools_to_chat_tools,
    chat_to_responses_request,
    original_responses_tool_type,
    rewrite_openai_responses_custom_payload,
)
from codex_shim.upstream_compat import (
    is_console_invalid_parameter_error,
    is_parallel_tool_calls_unsupported_error,
    learn_console_chat_compat_if_needed,
)
from codex_shim.ws_passthrough import WsPassthroughSession, _is_upstream_transport_dead


def _settings(tmp_path: Path) -> Path:
    path = tmp_path / "models.json"
    path.write_text("{}")
    return path


def _model(**kwargs) -> ShimModel:
    defaults = dict(
        slug="console-model",
        model="foo",
        display_name="Foo",
        provider="openai-responses",
        base_url="http://127.0.0.1:9/v1",
        api_key="k",
    )
    defaults.update(kwargs)
    return ShimModel(**defaults)


def _user(text: str) -> dict:
    return {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}


def test_small_helper_misses(tmp_path, monkeypatch):
    assert _reasoning_effort(_model(display_name="Mystery")) in {"medium", "high", "low", "minimal", "xhigh"}
    assert _parse_tiers(123)
    assert _positive_int("nope") is None
    assert _positive_float("") is None
    assert _coerce_optional_positive_int(None) is None
    assert _coerce_optional_positive_int(True) is None
    assert _coerce_optional_positive_int("abc") is None
    assert _load_output_token_reserve({}) is None
    assert parse_cache_byte_limit("   ") is None
    assert _json_object("") == {}
    assert parse_upstream_error("", 502)[0].startswith("upstream_http_")
    assert is_rate_limit(429, json.dumps({"error": {"message": "usage limit reached"}})) is False
    classify_ws_event_throttle({"error": {"status": 429, "message": "rate limit"}})
    assert _is_cursor_auth_failure("  ") is False
    assert _preview_text("") == ""
    assert _strip_tool_noise(["a"], verbose=True) == ["a"]
    assert _strip_tool_noise({"hookAdditionalContexts": 1, "keep": 2}, verbose=False)["keep"] == 2
    assert _format_write_started({"args": {"path": "a.py"}}) == "`a.py`"
    assert _format_edit_started({"args": {"path": "a.py"}}) == "`a.py`"
    assert _format_glob_result({"result": {"success": {}}}) == ""
    assert "a.py" in _format_grep_result(
        {
            "result": {
                "success": {
                    "workspaceResults": {
                        "/": {"content": {"matches": [{"file": "a.py", "matches": [{"lineNumber": 1, "content": "hi"}]}]}},
                        "skip": "nope",
                        "empty": {"content": "nope"},
                        "nomatch": {"content": {"matches": "nope"}},
                    }
                }
            }
        }
    )
    assert format_cursor_thinking_markdown("  ") == ""
    assert is_bridge_denied_tool(chat_name="") is True
    assert is_bridge_denied_tool(chat_name="shell", tool_type="web_search") is True
    assert "…" in _truncate("x" * 80, 8)
    specs = _iter_bridge_tool_candidates(
        [
            "skip",
            {"type": "namespace", "name": "ns", "tools": ["no", {"type": "other"}, {"type": "function", "name": ""}]},
            {"function": {"name": "fn", "description": "d", "parameters": {"type": "object"}}},
            {"type": "function"},
        ]
    )
    assert any(spec.emit_name == "fn" for spec in specs)
    assert _compact_parameters("nope")["type"] == "object"
    assert "Qwen" in display_name_from_model_id("qwen2-v2") 
    assert _settings_rows([{"a": 1}])
    assert _settings_rows("nope") == []
    assert _approx_token_count("abcd") >= 1
    circ = {}
    circ["x"] = circ
    assert _estimate_item_chars(circ) >= 1
    assert estimate_item_tokens(circ) >= 1
    assert extract_previous_summary(["skip", {"type": "message"}]) is None
    assert truncate_tool_output_chars([{"type": "message"}], 0)[1] == 0
    assert collapse_consecutive_duplicate_user_messages([])[1] == 0
    class Boom:
        def __str__(self):
            raise RuntimeError("no")
    _message_content_key({"type": "message", "role": "user", "content": [Boom()]})
    assert is_real_user_turn({"type": "message", "role": "user", "content": "compacted conversation"}) is False
    assert select_summarization_span([], tail_turns=2).head == []
    serialize_conversation(
        [
            {"type": "compaction", "encrypted_content": "c"},
            {"type": "function_call", "name": "t", "arguments": circ},
        ]
    )
    _collect_tool_names([{"type": "function_call", "name": "a"}] * 20)
    summary = deterministic_fallback_summary(
        [_user("Ship a local compaction module that never drops user work.")],
        previous_summary="p" * 2500,
        reason="test",
    )
    assert "## Goal" in summary
    log_compaction_cache_expansion(
        context="t",
        session_key="s",
        previous_response_id="p",
        cached_items=None,
        delta_items=1,
        total_items=None,
    )
    log_compaction_path("native", provider="byok", slug="s", extra="yes")
    assert _format_usage_body(None) == ""
    assert "_cache_read_input_tokens" in _format_usage_body({"cache_read_input_tokens": 3})
    log_upstream_response_headers("src", {"X-Other": "1"})
    skipper = ReplaySkipper(text_prefix="hello", tool_prefixes={0: "ab"})
    skipper.filter_chunk({"choices": [{"delta": {"content": "hello", "tool_calls": [{"index": "x", "function": {"arguments": "ab"}}]}}]})
    list(_iter_indexed_tools(SimpleNamespace(tool_calls={"bad": {"name": "t"}, 0: {"type": "function", "name": "t", "arguments": "{}"}})))
    assistant_prefill_message(SimpleNamespace(message_text="", tool_calls={0: {"name": "t", "arguments": "{}"}}))
    assert original_responses_tool_type("web_search") == "web_search"
    rewrite_openai_responses_custom_payload(
        {
            "type": "response.function_call_arguments.done",
            "item_id": "id1",
            "call_id": "id1",
            "arguments": "{}",
            "response": {"output": []},
        },
        {"apply_patch": "custom"},
        custom_call_ids={"id1"},
    )
    chat_to_responses_request({"messages": [], "stream": False}, "m", max_tokens=10)
    assert _agent_message_looks_like_plaintext("   ") is False
    assert _anthropic_content_to_chat_content([]) == ""
    assert _anthropic_content_to_chat_content({"type": "text", "text": "hi"})
    assert _anthropic_image_block_to_chat_part({"source": {}}) is None
    assert _anthropic_image_block_to_chat_part({"source": {"type": "url", "url": ""}}) is None
    assert _jsonish("plain") == "plain"
    assert _merge_chat_content("a", "") == "a"
    assert _native_tool_description({"type": "local_shell"})
    assert "command" in _native_tool_parameters({"type": "shell"})["properties"]
    assert _normalize_chat_roles([{"role": "developer", "content": "x"}])[0]["role"] == "system"
    assert _chat_finish_to_anthropic_stop("content_filter") == "refusal"
    _anthropic_tool_choice_to_chat("any")
    _responses_tools_to_chat_tools(
        [{"type": "namespace", "name": "ns", "tools": ["no", {"type": "other"}, {"type": "function"}]}]
    )
    assert is_parallel_tool_calls_unsupported_error(400, "") is False
    assert is_console_invalid_parameter_error(400, "") is False
    learn_console_chat_compat_if_needed(_model(), 200, "nope")
    mcp_search._CONFIG_CACHE = None
    fake = MagicMock()
    fake.exists.return_value = True
    fake.read_text.side_effect = OSError("no")
    monkeypatch.setattr(mcp_search, "_config_path", lambda: fake)
    assert mcp_search._read_codex_mcp_servers() == {}
    mcp_search._CONFIG_CACHE = None
    assert mcp_search.parse_mcp_function_name("mcp__exa__") is None
    assert mcp_search.full_mcp_tool_name("mcp__exa", "mcp__exa__web_search") == "mcp__exa__web_search"
    bad = tmp_path / "catalog.json"
    bad.write_text("not json")
    assert _load_desktop_catalog_models(bad) == []
    listed = tmp_path / "list.json"
    listed.write_text("[]")
    assert _load_desktop_catalog_models(listed) == []
    monkeypatch.setattr("codex_shim.compaction.context.is_chatgpt_passthrough_slug", lambda slug, path=None: True)
    assert context_window_tokens_for_slug("codex-x", byok_models=[], catalog_path=tmp_path / "missing.json") == 400_000
    monkeypatch.setattr("codex_shim.settings.load_chatgpt_passthrough_catalog_models", lambda *a, **k: [])
    assert chatgpt_catalog_slug("not-a-codex-model")
    assert chatgpt_upstream_model("totally-unknown-model-xyz")
    assert _is_listed_gpt_model({}) is False
    assert _is_listed_gpt_model({"slug": "gpt-x", "visibility": "hidden"}) is False
    row = _normalize_model_row({"name": "N", "baseURL": "http://x", "bearerToken": "t", "provider": "openai"})
    assert row["api_key"] == "t"
    settings = tmp_path / "dup.json"
    settings.write_text(
        json.dumps(
            {
                "models": [
                    {"model": "same", "display_name": "A", "provider": "openai", "base_url": "http://x", "api_key": "1"},
                    {"model": "same", "display_name": "A", "provider": "openai", "base_url": "http://x", "api_key": "2"},
                ]
            }
        )
    )
    loaded = ModelSettings(settings).load()
    assert len({m.slug for m in loaded}) == 2
    match = ModelSettings(settings).by_slug_or_model("same")
    assert match is None or match.model == "same"
    monkeypatch.setattr("codex_shim.settings.chatgpt_passthrough_available", lambda: True)
    monkeypatch.setattr("codex_shim.cursor_passthrough.cursor_passthrough_available", lambda: True)
    monkeypatch.setattr("codex_shim.cursor_passthrough.cursor_passthrough_display_names", lambda: ["cursor-auto"])
    slugs = available_model_slugs(loaded)
    assert slugs
    synthesize_orphan_tool_calls(["skip", {"type": "function_call_output", "call_id": "c1", "output": "o"}])
    align_responses_tool_call_ids(["skip", {"type": "function_call", "call_id": "uuid"}])
    assert build_user_content({"task": "t" * 7000, "has_images": False, "tool_count": 0, "input_items": 1}, [RouterCandidate("a")])
    parse_scores("{not json} extra {", ["a"])
    describe_upstream_error(SimpleNamespace(status=400, text=json.dumps({"error": {"message": "x"}, "extra_field": 1})))
    format_compaction_failure_detail(
        slug="s",
        provider="byok",
        native_message="n",
        tertiary_skip_reason="not_configured",
    )
    assert native_item_from_payload({"output": []}) is None
    native_item_from_payload(
        {
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "compacted"}],
                }
            ]
        }
    )
    cache = ChatgptConversationCache(tmp_path / "cache")
    cache.put("", "r1", [{"t": 1}])
    cache._response_id_from_filename("no-ext")
    (tmp_path / "cache" / "sess").mkdir(parents=True)
    (tmp_path / "cache" / "sess" / "file.json").write_text("{}")
    dangling = tmp_path / "cache" / "sess" / "gone.json"
    dangling.write_text("{}")
    cache._index_loaded = False
    cache._ensure_index_locked()
    cache._drop_disk_locked(("missing", "id"))
    monkeypatch.setattr(quota_dashboard, "discover_log_files", lambda *a, **k: [])
    assert quota_dashboard.main(["--log-dir", str(tmp_path)]) == 1
    log = tmp_path / "shim.log"
    log.write_text("not a usage line\nusage={not python\nusage={1:}\n[req] /v1 transport=http model='m' stream=false previous_response_id=none tools=0\n")
    old = tmp_path / "shim.log.old"
    old.write_text("x")
    old.touch()
    os.utime(old, (0, 0))
    files = quota_dashboard.discover_log_files(tmp_path, since=quota_dashboard.date.today(), until=None)
    assert old not in files
    report = quota_dashboard.parse_log_files([log])
    text = quota_dashboard.format_report(quota_dashboard.summarize(report))
    assert "no upstream usage" in text
    ndjson = tmp_path / "stream.ndjson"
    ndjson.write_text(
        "\n"
        + json.dumps({"type": "assistant", "message": {"content": [{"type": "text", "text": "hi"}]}, "timestamp_ms": 1, "model_call_id": "m"})
        + "\n"
        + json.dumps({"type": "tool_call", "subtype": "started", "call_id": "c1", "tool_call": {"unknownToolCall": {"args": {"x": "y" * 80}}}})
        + "\n"
    )
    visualize_main([str(ndjson), "--delay", "0", "--workdir", str(tmp_path)])
    format_event_line({"type": "thinking_completed"})
    session = WsPassthroughSession(client_session=SimpleNamespace(), client_ws=SimpleNamespace(closed=False))
    session.upstream_by_url["a"] = SimpleNamespace()
    assert session.upstream_ws is not None
    session.upstream_by_url["b"] = SimpleNamespace()
    assert session.upstream_ws is None
    session.note_chained_response("a", None)
    session.invalidate_native_chain("a")
    session.invalidate_native_chain()
    assert session.matches_thread(None) is False
    assert _is_upstream_transport_dead(ClientError("closing transport"))
    assert _is_upstream_transport_dead(OSError(104, "reset"))


def test_parse_tiers_and_retry_sync(monkeypatch):
    from codex_shim.catalog_context import DEFAULT_MODIFIER_TIERS

    assert _parse_tiers("") == DEFAULT_MODIFIER_TIERS
    assert _parse_tiers(0) == DEFAULT_MODIFIER_TIERS
    assert _clamp(1, 5, 3) == 5
    assert _apply_jitter(0, 0.2) == 0
    jittered = _apply_jitter(10.0, 0.2)
    assert 8.0 <= jittered <= 12.0
    assert RetryPolicy().is_retryable(exc=json.JSONDecodeError("m", "d", 0)) is True
    slept = []
    RetryPolicy(backoff_base=0.01).sleep_sync(0, sleep_fn=slept.append)
    assert slept
    throttle_sleep_sync(0)
    throttle_sleep_sync(0.01, sleep_fn=lambda d: None)
    assert _on_running_event_loop() is False
    throttle_sleep_sync(0.001)
    monkeypatch.setenv("CODEX_SHIM_RETRY_RATE_LIMIT_MIN", "9")
    monkeypatch.setenv("CODEX_SHIM_RETRY_RATE_LIMIT_MAX", "1")
    policy = retry_policy_from_env(attempts=2)
    assert policy.rate_limit_max == policy.rate_limit_min
    assert policy.attempts == 2
    assert _header_get([1, 2], "Retry-After") is None
    assert _parse_retry_after_value(None) is None
    assert _parse_retry_after_value("  ") is None
    assert _parse_retry_after_value(-1) is None
    assert _http_date_retry_seconds("not-a-date") is None
    naive = _http_date_retry_seconds("Wed, 21 Oct 2015 07:28:00", now=0)
    assert naive is None or naive >= 0
    monkeypatch.setenv("CODEX_SHIM_KEEPALIVE_MIN", "nope")
    monkeypatch.delenv("CODEX_SHIM_SSE_KEEPALIVE_INTERVAL", raising=False)
    monkeypatch.setenv("CODEX_SHIM_KEEPALIVE_MAX", "1")
    monkeypatch.setenv("CODEX_SHIM_KEEPALIVE_MIN", "10")
    keepalive_interval()
    monkeypatch.setenv("CODEX_SHIM_KEEPALIVE_MIN", "2")
    monkeypatch.setenv("CODEX_SHIM_KEEPALIVE_MAX", "2")
    assert keepalive_interval() == 2
    writer = DownstreamWriter(None)
    assert writer._ready() is False
    writer.response = SimpleNamespace(prepared=False, _payload_writer=object())
    assert writer._ready() is True


async def test_retry_sleep_and_stream_guard(monkeypatch):
    orig_sleep = asyncio.sleep
    slept = []

    async def fake_sleep(delay):
        slept.append(delay)

    monkeypatch.setattr("codex_shim.net.retry.asyncio.sleep", fake_sleep)
    await RetryPolicy(backoff_base=0.2).sleep(0)
    assert slept
    n = {"i": 0}

    def disconnect():
        n["i"] += 1
        return n["i"] >= 2

    with pytest.raises(ClientDisconnected):
        await throttle_sleep(2.5, origin="https://example.test", disconnect_fn=disconnect)

    async def boom_sleep(delay):
        del delay
        raise asyncio.CancelledError()

    monkeypatch.setattr("codex_shim.net.retry.asyncio.sleep", boom_sleep)
    with pytest.raises(asyncio.CancelledError):
        await throttle_sleep(2, origin="https://example.test", disconnect_fn=lambda: False)
    monkeypatch.setattr("codex_shim.net.retry.asyncio.sleep", orig_sleep)

    class DiscEmitter:
        already_emitted = False

        async def complete(self, response, *, upstream_saw_done):
            del response, upstream_saw_done
            raise ClientDisconnected()

        async def fail(self, response, message, *, code):
            del response, message, code
            raise asyncio.CancelledError()

    class Resp:
        prepared = True

        async def write(self, data):
            del data

        async def write_eof(self):
            return None

    async with StreamGuard(Resp(), DiscEmitter(), label="disc", keepalive=False):
        pass

    with pytest.raises(asyncio.CancelledError):
        guard = StreamGuard(Resp(), DiscEmitter(), label="fail", keepalive=False)
        await guard._fail_terminal("m", "c")

    async def ping_disc():
        raise ClientDisconnected()

    pinger = DownstreamPinger(ping_disc, interval=0.05)
    pinger.start()
    await asyncio.sleep(0.08)
    await pinger.stop()

    async def ping_boom():
        raise RuntimeError("ping")

    async def hold():
        await asyncio.sleep(5)

    owner = asyncio.create_task(hold())
    boom = DownstreamPinger(ping_boom, interval=0.05, owner_task=owner)
    boom.start()
    await asyncio.sleep(0.08)
    await boom.stop()
    owner.cancel()
    with pytest.raises(asyncio.CancelledError):
        await owner
    with pytest.raises(ClientDisconnected):
        await ping_websocket(SimpleNamespace(closed=True))


async def test_orchestrator_paths():
    usable = (
        "## Goal\n- Ship a local compaction module that never drops user work.\n\n"
        + ("progress notes " * 40)
    )
    items = [_user("Ship a local compaction module that never drops user work.")]

    async def unused(*a, **k):
        raise AssertionError("unexpected adapter")

    async def native_empty(request, prepared):
        del request, prepared
        return NativeAttemptResult(item=None, native_message="")

    async def native_item(request, prepared):
        del request, prepared
        return NativeAttemptResult(item=compaction_output_item(usable))

    async def summary_ok(request, prepared, native_message):
        del request, prepared, native_message
        return SummarizationAttemptResult(summary=usable)

    async def summary_bad(request, prepared, native_message):
        del request, prepared, native_message
        return SummarizationAttemptResult(summary="too short")

    async def tertiary_ok(request, prepared, native_message, slug):
        del request, prepared, native_message, slug
        return SummarizationAttemptResult(summary=usable)

    orch = CompactionOrchestrator(
        CompactionAdapters(
            native_chatgpt=native_item,
            native_cursor=native_empty,
            native_byok=native_empty,
            summarization_chatgpt=summary_ok,
            summarization_cursor=summary_bad,
            summarization_byok=summary_ok,
            tertiary_byok=tertiary_ok,
            acquire_chatgpt_lock=unused,
        )
    )
    empty = await orch.run(
        CompactionRequest(
            http_request=object(),
            body={"model": "m"},
            stripped_input=[],
            requested_slug="m",
            provider="byok",
        )
    )
    assert empty.phase == "sanitization_only"
    native = await orch.run(
        CompactionRequest(
            http_request=object(),
            body={"model": "m"},
            stripped_input=items,
            requested_slug="m",
            provider="chatgpt",
        )
    )
    assert native.phase == "native"
    with pytest.raises(CompactionOrchestratorError):
        await orch.run(
            CompactionRequest(
                http_request=object(),
                body={"model": "m"},
                stripped_input=items,
                requested_slug="m",
                provider="cursor",
                settings=CompactionSettings(fallback_enabled=False),
            )
        )
    summarized = await orch.run(
        CompactionRequest(
            http_request=object(),
            body={"model": "m"},
            stripped_input=items,
            requested_slug="m",
            provider="byok",
            skip_native=True,
            preset_native_message="native failed",
        )
    )
    assert summarized.phase in {"summarization", "local_fallback", "tertiary"}


async def test_cursor_agent_error_paths(monkeypatch):
    monkeypatch.setattr("codex_shim.cursor_passthrough.cursor_catalog_models", lambda: [])
    entries = cursor_passthrough_entries()
    assert entries
    build_cursor_prompt(
        {
            "model": "cursor-auto",
            "input": [_user("hi")],
            "messages": [{"role": "assistant", "content": "ok", "tool_calls": [{"function": {"name": "t", "arguments": "{}"}}]}],
        }
    )

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

    class ErrProc:
        def __init__(self):
            self.stdin = FakeStdin()
            self.stdout = FakeStream([b'{"type":"error","message":"not authenticated"}\nleftover'])
            self.stderr = FakeStream([b"not authenticated"])
            self.returncode = None

        def kill(self):
            self.returncode = -9

        async def wait(self):
            self.returncode = 1

    async def spawn(*a, **k):
        del a, k
        return ErrProc()

    monkeypatch.setattr("codex_shim.cursor_passthrough.asyncio.create_subprocess_exec", spawn)
    monkeypatch.setattr("codex_shim.cursor_passthrough._cursor_agent_bin", lambda: "cursor-agent")
    monkeypatch.setattr("codex_shim.cursor_passthrough.cursor_workspace", lambda: "/tmp")
    events = [event async for event in iter_cursor_agent_events("hi", "auto")]
    assert any(event.get("type") == "error" for event in events)

    class ExitProc(ErrProc):
        def __init__(self):
            super().__init__()
            self.stdout = FakeStream([b""])
            self.stderr = FakeStream([b"not authenticated"])

    async def spawn_exit(*a, **k):
        del a, k
        return ExitProc()

    monkeypatch.setattr("codex_shim.cursor_passthrough.asyncio.create_subprocess_exec", spawn_exit)
    events2 = [event async for event in iter_cursor_agent_events("hi", "auto")]
    assert any("authenticated" in str(event.get("message") or "").lower() or event.get("type") == "error" for event in events2)


async def test_websocket_invalid_frames_and_stream_400(monkeypatch, tmp_path):
    settings = _settings(tmp_path)
    shim = ShimServer(settings)
    client = TestClient(TestServer(shim.app()))
    await client.start_server()
    try:
        ws = await client.ws_connect("/v1/responses")
        await ws.send_str("not-json")
        bad = json.loads((await ws.receive(timeout=2)).data)
        assert bad["error"]["message"] == "invalid JSON websocket frame"
        await ws.send_str("[]")
        obj = json.loads((await ws.receive(timeout=2)).data)
        assert "JSON object" in obj["error"]["message"]
        await ws.send_json({"type": "ping"})
        typ = json.loads((await ws.receive(timeout=2)).data)
        assert "response.create" in typ["error"]["message"]
        await ws.send_bytes(b"bin")
        binary = json.loads((await ws.receive(timeout=2)).data)
        assert "binary" in binary["error"]["message"]
        await ws.close()
        resp = await client.post(
            "/api/switch",
            data="{",
            headers={PICKER_TOKEN_HEADER: shim.picker_token, "Content-Type": "application/json"},
        )
        assert resp.status == 400
    finally:
        await client.close()

    auth = tmp_path / "auth.json"
    auth.write_text(json.dumps({"tokens": {"access_token": "tok", "account_id": "acct"}}))
    monkeypatch.setattr("codex_shim.server.DEFAULT_CODEX_AUTH", auth)
    monkeypatch.setattr("codex_shim.settings.DEFAULT_CODEX_AUTH", auth)

    class QuietGuard(StreamGuard):
        def __init__(self, *args, **kwargs):
            kwargs["keepalive"] = False
            super().__init__(*args, **kwargs)

    monkeypatch.setattr("codex_shim.server.StreamGuard", QuietGuard)

    async def chatgpt_400(*a, **k):
        del a, k
        return SimpleNamespace(
            response=SimpleNamespace(headers={}, release=lambda: None, status=400),
            status=400,
            error_text='{"error":{"message":"nope"}}',
        )

    monkeypatch.setattr("codex_shim.server.post_chatgpt_with_retry", chatgpt_400)
    request = make_mocked_request("POST", "/v1/responses")
    out = await ShimServer(settings)._chatgpt_passthrough(
        request,
        {"model": "codex-gpt-5-6-luna", "input": [_user("hi")], "stream": True},
        collect_stream=False,
    )
    assert out.status == 200

    async def openai_400(*a, **k):
        del a, k
        return HttpPostResult(
            response=SimpleNamespace(headers={}, release=lambda: None),
            status=400,
            content_type="application/json",
            error_text='{"error":{"message":"bad"}}',
        )

    monkeypatch.setattr("codex_shim.server.retry_aiohttp_post", openai_400)
    err = await ShimServer(settings)._post_openai_responses(
        request,
        _model(),
        {"model": "console-model", "input": [], "stream": True},
    )
    assert err.status == 200

    cfg = tmp_path / "codex.toml"
    monkeypatch.setattr("codex_shim.server.CODEX_CONFIG_PATH", cfg)
    _set_active_model("x")
    class Boom:
        def exists(self):
            return True

        def read_text(self):
            return 'model = "old"\n'

        def write_text(self, text):
            raise OSError("no")

    monkeypatch.setattr("codex_shim.server.CODEX_CONFIG_PATH", Boom())
    _set_active_model("new", "New")
    class BoomRead:
        def exists(self):
            return True

        def read_text(self):
            raise OSError("no")

    monkeypatch.setattr("codex_shim.server.CODEX_CONFIG_PATH", BoomRead())
    _set_active_model("new")


def test_nous_auth_isolated_error_paths(tmp_path, monkeypatch):
    home = Path(os.environ["HERMES_HOME"])
    auth = home / "auth.json"
    shared = home / "shared" / "nous_auth.json"
    shared.parent.mkdir(parents=True, exist_ok=True)
    auth.write_text("[]")
    assert nous_auth._token_from_auth_file(auth) == ""
    with pytest.raises(nous_auth.AuthStoreUnreadable):
        nous_auth._load_json_object(auth)
    auth.write_text("{}")
    shared.write_text("[]")
    assert (
        nous_auth._refresh_nous_oauth_locked(home=home, shared_path=shared, env={}, timeout=1) is False
    )
    auth.write_text(json.dumps({"refresh_token": "rt"}))
    monkeypatch.setattr(nous_auth, "_exchange_refresh_token", lambda **k: {})
    assert nous_auth._refresh_nous_oauth_locked(home=home, shared_path=shared, env={}, timeout=1) is False
    monkeypatch.setattr(
        nous_auth,
        "_exchange_refresh_token",
        lambda **k: {"access_token": "at", "refresh_token": "rt2", "expires_in": 30},
    )
    assert nous_auth._refresh_nous_oauth_locked(home=home, shared_path=shared, env={}, timeout=1) is True
    assert nous_auth._nous_state_from_store({"refresh_token": "r"})["refresh_token"] == "r"
    assert nous_auth._nous_state_from_store({}) == {}
    assert nous_auth.hermes_home(environ={}) == Path.home() / ".hermes"
    assert nous_auth._portal_base_url({"portal_base_url": "https://portal.nousresearch.com"}, {}) 

    orig_resolve = Path.resolve

    def boom_resolve(self, *a, **k):
        if str(self).startswith(str(tmp_path)):
            raise OSError("no")
        return orig_resolve(self, *a, **k)

    monkeypatch.setattr(Path, "resolve", boom_resolve)
    nous_auth._refuse_live_hermes_paths_during_pytest(tmp_path, tmp_path / "s.json")
    monkeypatch.setattr(Path, "resolve", orig_resolve)

    lock_path = tmp_path / "held.lock"
    fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        with pytest.raises(TimeoutError):
            with nous_auth._exclusive_file_lock(lock_path, timeout=0.05):
                pass
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)

    def boom_commit(tmp, dest):
        del tmp, dest
        raise OSError("commit")

    monkeypatch.setattr(nous_auth, "_commit_staged", boom_commit)
    with pytest.raises(OSError):
        nous_auth._atomic_write_json_pair(tmp_path / "a.json", {}, tmp_path / "b.json", {})

    calls = {"n": 0}
    real_stage = nous_auth._stage_json

    def stage(path, payload):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("shared")
        return real_stage(path, payload)

    monkeypatch.setattr(nous_auth, "_stage_json", stage)
    with pytest.raises(OSError):
        nous_auth._atomic_write_json_pair(tmp_path / "c.json", {}, tmp_path / "d.json", {})

    nous_auth._pending_persist = nous_auth._PendingPersist(
        auth_path=tmp_path / "p.json",
        store={},
        shared_path=tmp_path / "q.json",
        shared={},
    )

    @contextmanager
    def boom_locks(*a, **k):
        raise TimeoutError("lock")
        yield

    monkeypatch.setattr(nous_auth, "_auth_locks", boom_locks)
    assert nous_auth._flush_pending_persist() is False
    nous_auth._pending_persist = None


def test_nous_exchange_rejects_non_object(monkeypatch):
    def fake_urllib(*a, **k):
        del a, k
        return SimpleNamespace(body=b"[1]")

    monkeypatch.setattr(nous_auth, "request_urllib", fake_urllib)
    with pytest.raises(ValueError, match="not an object"):
        nous_auth._exchange_refresh_token(
            portal_base_url="https://example.test",
            client_id="id",
            refresh_token="rt",
            timeout=1,
        )


async def test_resolve_auto_error_paths():
    config = RouterConfig(
        enabled=True,
        slug="auto",
        display_name="Auto",
        classifier=None,
        threshold=0.7,
        default=None,
        cache=True,
        candidates=(RouterCandidate("a"), RouterCandidate("b")),
        timeout=1,
        max_tokens=16,
    )
    slug, meta = await resolve_auto(config, [], {"input": [_user("hi")]}, None)
    assert slug is None and meta["reason"] == "no candidates"

    async def classify(system, user):
        del system, user
        raise RuntimeError("classifier")

    picked, info = await resolve_auto(config, list(config.candidates), {"input": [_user("hi")]}, classify, log=print)
    assert picked in {"a", "b"}
    assert info["reason"] == "classifier error"

    async def ok(system, user):
        del system, user
        return '{"scores": {"a": 1, "b": 0}}'

    class BoomCache(dict):
        def clear(self):
            raise RuntimeError("cache")

        def get(self, key, default=None):
            return None

        def __setitem__(self, key, value):
            raise RuntimeError("set")

    import codex_shim.router as router_mod

    router_mod._cache.clear()
    monkeypatch_cache = BoomCache()
    orig = router_mod._cache
    router_mod._cache = monkeypatch_cache
    try:
        none, err = await resolve_auto(config, list(config.candidates), {"input": [_user("task")]}, ok, log=print)
        assert err["reason"] == "error"
    finally:
        router_mod._cache = orig

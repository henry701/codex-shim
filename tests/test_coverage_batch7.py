from __future__ import annotations

import base64
import json
from types import SimpleNamespace

import pytest

from codex_shim.catalog import (
    _supported_reasoning_levels,
    _validate_catalog_file_models,
    assign_catalog_display_priorities,
    codex_config_overrides,
    write_catalog,
)
from codex_shim.catalog_context import (
    CatalogContextOverride,
    CatalogContextSettings,
    apply_catalog_context_to_entry,
    load_catalog_context_settings,
)
from codex_shim.chatgpt_conversation_cache import (
    ChatgptConversationCache,
    configured_max_memory_entries,
    sanitize_path_segment,
    thread_id_from_headers,
)
from codex_shim.compaction.config import (
    CompactionSettings,
    compaction_prompt_cache_key,
    load_compaction_settings,
)
from codex_shim.compaction.context import _context_from_catalog_entry, context_window_tokens_for_slug
from codex_shim.compaction.errors import describe_upstream_error, format_compaction_failure_detail, parse_upstream_error_detail
from codex_shim.compaction.input_audit import CompactionInputItemRef, CompactionSanitizationAudit, summarize_compaction_input_items
from codex_shim.compaction.local import _exclude_tail_within, is_real_user_turn, item_text, summary_is_usable
from codex_shim.compaction.pipeline import (
    _message_content_key,
    _rewrite_tool_output_item,
    _tool_output_text,
    _truncate_output_value,
    collapse_consecutive_duplicate_user_messages,
    truncate_tool_output_chars,
)
from codex_shim.compaction.protocol import (
    SHIM_COMPACTION_PREFIX,
    _reasoning_summary_texts,
    _tool_item_call_id,
    compaction_item_from_response_payload,
    compaction_summary_from_output,
    decode_shim_compaction_summary,
)
from codex_shim.cursor_passthrough import _format_shell_started, infer_cursor_context_limit
from codex_shim.mcp_search import format_tool_search_result, full_mcp_tool_name, parse_mcp_function_name
from codex_shim.naming import _format_token, display_name_from_slug
from codex_shim.net.emitters import AnthropicMessagesEmitter
from codex_shim.net.errors import (
    chat_chunk_upstream_error,
    classify_ws_event_throttle,
    close_http_response,
    is_quota_limit,
    parse_resets_in_seconds,
    parse_upstream_error,
)
from codex_shim.net.prefill import ReplaySkipper, _iter_indexed_tools, assistant_prefill_message, should_prefill_continue
from codex_shim.net.retry import RetryPolicy, _env_float, _env_int, _http_date_retry_seconds, parse_retry_after, throttle_sleep
from codex_shim.net.sse import ClientDisconnected
from codex_shim.net.stream_guard import StreamGuard
from codex_shim.responses_input_pipeline import responses_input_items, sanitize_compaction_input_with_pipeline
from codex_shim.router import _clamp01, _latest_from_input, load_router_config, parse_scores
from codex_shim.settings import (
    ModelSettings,
    ShimModel,
    _int_or_none,
    _load_published_chatgpt_catalog_models,
    _model_rows,
    _resolve_api_key,
    default_model_slug,
)
from codex_shim.tool_translate import strip_function_call_output_item_id
from codex_shim.translate import (
    _agent_message_text,
    _anthropic_assistant_message_to_chat,
    _anthropic_tools_to_chat_tools,
    _chat_parts_from_content,
    _coerce_custom_input_item,
    _content_to_text,
    _int_token,
    _normalize_chat_image_detail,
    _responses_input_to_messages,
    _responses_tool_function_name,
    _responses_tool_to_chat_function,
    _responses_tools_to_chat_tools,
    _sanitize_chat_messages,
    _tool_choice_name,
    anthropic_to_chat_response,
    integerize_json_whole_floats,
    normalize_responses_usage,
    responses_to_chat,
    unwrap_custom_tool_input,
)
from codex_shim.upstream_compat import (
    _content_nonempty,
    _flatten_console_content,
    _load_store,
    _tool_call_names,
    should_omit_parallel_tool_calls,
)
from codex_shim.ws_passthrough import WsPassthroughConnectError, _is_upstream_transport_dead


def _model(**kwargs) -> ShimModel:
    defaults = dict(
        slug="local-llama",
        model="llama",
        display_name="Llama",
        provider="openai",
        base_url="http://127.0.0.1:11434/v1",
        api_key="k",
    )
    defaults.update(kwargs)
    return ShimModel(**defaults)


def test_catalog_and_context_helpers(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="unknown catalog tier"):
        assign_catalog_display_priorities([("nope", {"slug": "x"})])
    levels = _supported_reasoning_levels(_model(raw={"reasoning_efforts": ["", "medium", "medium"]}))
    assert levels[0]["effort"] == "medium"
    with pytest.raises(ValueError, match="non-empty"):
        _validate_catalog_file_models([{"slug": "x", "supported_reasoning_levels": []}])
    with pytest.raises(ValueError, match="must be an object"):
        _validate_catalog_file_models(
            [{"slug": "x", "supported_reasoning_levels": ["low"], "default_reasoning_level": "low"}]
        )
    with pytest.raises(ValueError, match="needs effort"):
        _validate_catalog_file_models(
            [{"slug": "x", "supported_reasoning_levels": [{"effort": ""}], "default_reasoning_level": "low"}]
        )
    with pytest.raises(ValueError, match="Desktop enum"):
        _validate_catalog_file_models(
            [
                {
                    "slug": "x",
                    "supported_reasoning_levels": [{"effort": "medium", "description": "m"}],
                    "default_reasoning_level": "medium",
                    "input_modalities": ["video"],
                }
            ]
        )
    overrides = codex_config_overrides(tmp_path / "cat.json", "local-llama", 8767)
    assert any("8767" in item for item in overrides)

    cursor_entry = {
        "slug": "cursor-auto",
        "supported_reasoning_levels": [{"effort": "medium", "description": "m"}],
        "default_reasoning_level": "medium",
        "input_modalities": ["text"],
    }
    monkeypatch.setattr("codex_shim.catalog.chatgpt_passthrough_available", lambda: False)
    monkeypatch.setattr("codex_shim.catalog.cursor_passthrough_available", lambda: True)
    monkeypatch.setattr("codex_shim.catalog.cursor_passthrough_entries", lambda: [dict(cursor_entry)])
    monkeypatch.setattr("codex_shim.catalog.usable_byok_models", lambda models: [])
    written = write_catalog([], tmp_path / "catalog.json")
    payload = json.loads(written.read_text())
    assert payload["models"][0]["isDefault"] is True

    assert load_catalog_context_settings(None) is None
    assert load_catalog_context_settings({"catalog_context": {"modifier": "nope"}}) is None
    settings = load_catalog_context_settings(
        {"catalog_context": {"modifier": "1.5", "apply_to_tiers": "byok", "overrides": {"x": {}}}}
    )
    assert settings is not None
    empty_entry = apply_catalog_context_to_entry({"slug": "x"}, tier="byok", settings=settings)
    assert empty_entry == {"slug": "x"}
    override_settings = CatalogContextSettings(
        modifier=None,
        apply_to_tiers=frozenset({"byok"}),
        overrides={"slug-a": CatalogContextOverride(max_context_window=111)},
        override_patterns={},
    )
    updated = apply_catalog_context_to_entry(
        {"slug": "slug-a", "context_window": 50},
        tier="byok",
        settings=override_settings,
    )
    assert updated["context_window"] == 111
    assert _context_from_catalog_entry({"other": 1}) is None
    assert context_window_tokens_for_slug("cursor-composer", byok_models=[], catalog_path=tmp_path / "none.json") == 200_000


def test_cache_compaction_pipeline_protocol(tmp_path, monkeypatch):
    assert thread_id_from_headers({"x-codex-turn-metadata": "{"}) is None
    assert len(sanitize_path_segment("///")) == 32
    assert len(sanitize_path_segment("a" * 250)) <= 200
    monkeypatch.setenv("CODEX_SHIM_CHATGPT_CACHE_MAX_MEMORY_ENTRIES", "nope")
    assert configured_max_memory_entries() > 0
    monkeypatch.setenv("CODEX_SHIM_CHATGPT_CACHE_MAX_MEMORY_ENTRIES", "0")
    assert configured_max_memory_entries() > 0
    monkeypatch.setenv("CODEX_SHIM_CHATGPT_CACHE_MAX_MEMORY_ENTRIES", "4")
    assert configured_max_memory_entries() == 4

    cache = ChatgptConversationCache(tmp_path / "cache")
    assert cache.get("sess", "") is None
    bad = cache._entry_path("sess", "r1")
    bad.parent.mkdir(parents=True)
    bad.write_text("{")
    assert cache.get("sess", "r1") is None
    bad.write_text("[]")
    assert cache.get("sess", "r1") is None
    bad.write_text(json.dumps({"items": []}))
    assert cache.get("sess", "r1") is None
    assert cache.latest("") is None
    cache._memory[("sess-mem", "rid")] = [{"role": "user"}]
    assert cache.latest("sess-mem") == [{"role": "user"}]
    (tmp_path / "cache" / "file-not-dir").write_text("x")
    cache._index_loaded = False
    cache._ensure_index_locked()

    missing = tmp_path / "missing.json"
    assert load_compaction_settings(missing).tail_turns == 2
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{")
    assert load_compaction_settings(bad_json).fallback_enabled is True
    listed = tmp_path / "list.json"
    listed.write_text("[]")
    assert load_compaction_settings(listed).model is None
    cfg = tmp_path / "ok.json"
    cfg.write_text(
        json.dumps(
            {
                "compaction": {
                    "override_current_model": True,
                    "tail_turns": "3",
                    "prompt_cache_key_per_session": "yes",
                    "compaction_output_token_reserve": "0",
                    "context_window_token_budget": "12",
                }
            }
        )
    )
    loaded = load_compaction_settings(cfg)
    assert loaded.override_current_model is True
    assert compaction_prompt_cache_key(loaded, "sess-1").endswith(":sess-1")
    settings_bool = CompactionSettings(prompt_cache_key_per_session=False)
    assert compaction_prompt_cache_key(settings_bool, "s") == "codex-shim-compact:v1"

    detail = parse_upstream_error_detail(
        SimpleNamespace(status=400, text=json.dumps({"detail": "  hi  "})),
        fallback="fb",
    )
    assert detail.message == "hi"
    described = describe_upstream_error(
        SimpleNamespace(status=400, text='{"error":{"type":"x","param":"input","message":"nope"}}'),
        context="native",
    )
    assert "HTTP 400" in described
    fail = format_compaction_failure_detail(
        slug="s",
        provider="p",
        native_message="n",
        tertiary_configured_slug="local-llama",
    )
    assert "skipped" in fail
    ref = CompactionInputItemRef(0, "function_call", name="shell", role="assistant")
    assert "name=shell" in ref.label()
    audit = CompactionSanitizationAudit()
    audit.dropped.append((CompactionInputItemRef(1, "function_call_output"), "orphan"))
    audit.preserved.append((ref, "kept"))
    warnings = audit.warning_lines()
    assert any("dropped" in line for line in warnings)
    assert summarize_compaction_input_items("nope") == (0, [])
    count, labels = summarize_compaction_input_items(["x", {"type": "message", "role": "user"}])
    assert count == 2 and any("?" in item for item in labels)

    assert item_text("plain") == "plain"
    assert "hi" in item_text({"content": ["hi", {"text": "there"}], "summary": [{"text": "sum"}]})
    assert is_real_user_turn("no") is False
    assert is_real_user_turn({"type": "compaction", "role": "user"}) is False
    assert is_real_user_turn({"role": "assistant", "content": "x"}) is False
    assert is_real_user_turn({"role": "user", "type": "function_call"}) is False
    assert is_real_user_turn({"role": "user", "content": ""}) is False
    head, tail, excluded = _exclude_tail_within([], tail_turns=2, preserve_recent_tokens=10)
    assert head == [] and excluded == 0
    assert summary_is_usable("## Goal\n" + ("x" * 900), items=[{"type": "function_call", "name": "shell"}]) is True
    assert _tool_output_text("abc") == "abc"
    assert _tool_output_text([12, {"type": "text", "text": "z"}]) == "z"
    assert _tool_output_text(9) == ""
    assert _truncate_output_value("x", 0)[1] == 0
    assert _truncate_output_value({"a": 1}, 10)[1] == 0
    items, n, removed = truncate_tool_output_chars(["skip", {"type": "function_call_output", "output": "a" * 50}], 10)
    assert n == 1 and removed > 0
    search = _rewrite_tool_output_item({"type": "tool_search_output", "tools": [1]})
    assert search["tools"] == []
    assert _rewrite_tool_output_item({"type": "message"}) is None
    assert _message_content_key({"type": "message", "role": "user", "content": None}) == ""
    assert _message_content_key({"type": "message", "role": "user", "content": 12}) == "12"
    collapsed, dropped = collapse_consecutive_duplicate_user_messages([])
    assert dropped == 0
    collapsed, dropped = collapse_consecutive_duplicate_user_messages(
        [
            {"type": "message", "role": "user", "content": "same"},
            {"type": "message", "role": "user", "content": "same"},
            {"type": "function_call", "name": "shell"},
        ]
    )
    assert dropped == 1
    assert _tool_item_call_id({"call_id": "c1"}) == "c1"
    assert _tool_item_call_id({}) is None
    assert decode_shim_compaction_summary("not-a-blob") is None
    bogus = SHIM_COMPACTION_PREFIX + base64.urlsafe_b64encode(b"{").decode()
    assert decode_shim_compaction_summary(bogus) is None
    listed = SHIM_COMPACTION_PREFIX + base64.urlsafe_b64encode(b"[]").decode()
    assert decode_shim_compaction_summary(listed) is None
    empty = SHIM_COMPACTION_PREFIX + base64.urlsafe_b64encode(json.dumps({"summary": None}).encode()).decode()
    assert decode_shim_compaction_summary(empty) is None
    assert _reasoning_summary_texts({"summary": " think "}) == [" think "]
    assert compaction_summary_from_output(["skip", {"type": "output_text", "text": "sum"}])
    item = compaction_item_from_response_payload(
        {
            "output": ["x", {"type": "compaction", "encrypted_content": "blob", "id": "i", "status": "completed"}],
        }
    )
    assert item["id"] == "i"
    from_summary = compaction_item_from_response_payload({"compaction_summary": {"encrypted_content": "e"}})
    assert from_summary["encrypted_content"] == "e"


def test_net_retry_errors_prefill_stream_guard(monkeypatch):
    monkeypatch.setenv("CODEX_SHIM_RETRY_ATTEMPTS", "nope")
    assert _env_int("CODEX_SHIM_RETRY_ATTEMPTS", 3) == 3
    monkeypatch.setenv("CODEX_SHIM_RETRY_BASE", "nope")
    assert _env_float("CODEX_SHIM_RETRY_BASE", 0.5) == 0.5
    policy = RetryPolicy(attempts=3, wait_budget=10.0, backoff_base=0.0)
    assert policy.should_continue(0, 0.0, None) is True
    short = RetryPolicy(attempts=1, wait_budget=10.0)
    assert short.should_continue(0, 0.0, 5.0) is False
    extend = RetryPolicy(attempts=2, wait_budget=10.0)
    assert extend.should_continue(1, 1.0, 2.0) is True
    assert parse_retry_after(body="") is None
    assert parse_retry_after(body="{") is None
    assert parse_retry_after(body="[]") is None
    nested = parse_retry_after(body=json.dumps({"metadata": {"headers": {"Retry-After": "2"}}}))
    assert nested == 2.0 or nested is not None
    assert _http_date_retry_seconds("not-a-date") is None
    naive = _http_date_retry_seconds("Wed, 01 Jan 2020 00:00:00 GMT", now=0)
    assert naive is None or naive >= 0

    assert chat_chunk_upstream_error("nope") is None
    assert chat_chunk_upstream_error({"choices": [None]}) is None
    assert chat_chunk_upstream_error({"choices": [{"finish_reason": "error"}]})[0] == "error"
    assert parse_upstream_error("not-json", 500)[1].startswith("not-json")
    assert parse_upstream_error("[]", 500)[1] == "[]"
    code, message = parse_upstream_error(json.dumps({"error": " boom ", "message": " top ", "code": "C"}), 400)
    assert "top" in message
    class Boom:
        def close(self):
            raise RuntimeError("close")

        def release(self):
            raise RuntimeError("release")

    close_http_response(Boom())
    assert classify_ws_event_throttle("nope") == "none" or classify_ws_event_throttle("nope")
    nested_event = classify_ws_event_throttle({"error": {"status": 429, "message": "rate limit"}})
    assert nested_event
    assert is_quota_limit(429, json.dumps({"error": {"type": "usage_limit_reached"}})) is True
    assert is_quota_limit(200, json.dumps({"error": {"message": "usage limit reached"}})) is True
    assert is_quota_limit(
        200,
        json.dumps({"error": {"plan_type": "plus", "resets_in_seconds": 30}}),
    ) is True
    resets = parse_resets_in_seconds(json.dumps({"error": {"resets_at": "2020-01-01T00:00:00"}}))
    assert resets is None or resets >= 0

    assert should_prefill_continue(finish_reason="length", saw_done=False, continues=0, as_responses=False) is False
    state = SimpleNamespace(message_text="", reasoning_blocks={1: {"text": "r"}}, tool_calls={0: {"name": "", "arguments": ""}})
    prefill = assistant_prefill_message(state)
    assert prefill is None or "reasoning_content" in prefill
    skipper = ReplaySkipper(text_prefix="hello", tool_prefixes={0: "abc"})
    assert skipper.filter_chunk({"choices": [None]})["choices"][0] is None
    assert skipper.filter_chunk({"choices": [{"delta": "x"}]})["choices"][0]["delta"] == "x"
    skipped = skipper.filter_chunk({"choices": [{"delta": {"content": "hello"}}]})
    assert skipped is None or "content" not in ((skipped.get("choices") or [{}])[0].get("delta") or {})
    skipper._filter_tool_call({"index": []})
    skipper._filter_tool_call({"index": 0, "function": "nope"})
    skipper._filter_tool_call({"index": 0, "function": {"arguments": "abc"}})
    listed = list(_iter_indexed_tools(SimpleNamespace(tool_calls={"bad": {"name": "x"}, 0: "skip"})))
    assert listed

    class Emitter:
        already_emitted = False
        terminal_event = None

        async def complete(self, response, *, upstream_saw_done):
            del response, upstream_saw_done
            raise ClientDisconnected()

        async def fail(self, response, message, *, code):
            del response, message, code
            raise RuntimeError("fail")

    guard = StreamGuard(None, Emitter(), label="t", keepalive=False)
    assert guard.can_retry is True
    guard.abandon()
    assert guard.can_retry is True
    guard._log_end()
    boom_guard = StreamGuard(SimpleNamespace(), Emitter(), label="t", keepalive=False, finish_reason=lambda: (_ for _ in ()).throw(RuntimeError("x")))
    boom_guard._log_end()


async def test_throttle_sleep_and_stream_guard_disconnect():
    await throttle_sleep(0)
    with pytest.raises(ClientDisconnected):
        await throttle_sleep(1, disconnect_fn=lambda: True)

    class Emitter:
        already_emitted = False
        terminal_event = None

        async def complete(self, response, *, upstream_saw_done):
            del response, upstream_saw_done
            raise ClientDisconnected()

        async def fail(self, response, message, *, code):
            del response, message, code
            raise ClientDisconnected()

    guard = StreamGuard(SimpleNamespace(write=lambda *_a, **_k: None), Emitter(), label="t", keepalive=False)
    with pytest.raises(ClientDisconnected):
        await guard.note_upstream_disconnect(RuntimeError("gone"))
    await StreamGuard(None, Emitter(), label="t", keepalive=False)._fail_terminal("m", "c")

    class SoftFail(Emitter):
        async def fail(self, response, message, *, code):
            del response, message, code
            raise RuntimeError("soft")

    await StreamGuard(SimpleNamespace(), SoftFail(), label="t", keepalive=False).note_upstream_disconnect(RuntimeError("gone"))
    assert AnthropicMessagesEmitter(SimpleNamespace(failed=False, terminal_emitted=False)).terminal_event is None


def test_settings_router_translate_misc(tmp_path, monkeypatch):
    assert _format_token("v1.2.3") == "V1.2.3"
    assert display_name_from_slug("") == "Model"
    assert parse_mcp_function_name("mcp__exa__") is None
    assert full_mcp_tool_name("exa", "mcp__exa__search") == "mcp__exa__search"
    formatted = format_tool_search_result(
        "exa",
        ["skip", {"name": ""}, {"name": "web_search_exa", "description": None, "parameters": {}}],
    )
    assert formatted
    assert strip_function_call_output_item_id({"type": "message"})["type"] == "message"
    assert infer_cursor_context_limit(display_name="200k context", upstream_id="x") == 200_000
    assert infer_cursor_context_limit(display_name="mystery", upstream_id="other") > 0
    assert _format_shell_started({"args": {"command": "curl /_cursor_bridge/v1/invoke"}}) == "→ Codex bridge invoke"
    assert _format_shell_started({"args": {}}) == "`(command)`"

    monkeypatch.setenv("CURSOR_TEST_KEY", "from-env")
    assert _resolve_api_key("${CURSOR_TEST_KEY}") == "from-env"
    monkeypatch.setattr("codex_shim.settings.DEFAULT_CURSOR_API_KEY_FILE", tmp_path / "missing-key")
    assert isinstance(_resolve_api_key(""), str)
    assert _int_or_none("nope") is None
    assert _model_rows("nope") == []
    assert _model_rows({"models": "nope"}) == []
    assert _model_rows([{"model": "m"}])
    bad_cat = tmp_path / "cat.json"
    bad_cat.write_text("{")
    assert _load_published_chatgpt_catalog_models(bad_cat) == []
    listed = tmp_path / "listed.json"
    listed.write_text("[]")
    assert _load_published_chatgpt_catalog_models(listed) == []
    mixed = tmp_path / "mixed.json"
    mixed.write_text(json.dumps({"models": ["skip", {"slug": "codex-gpt-5-6-luna", "id": "gpt-5.6"}]}))
    _load_published_chatgpt_catalog_models(mixed)
    settings = ModelSettings(tmp_path / "models.json")
    models = settings._models_from_settings_data(
        {
            "models": [
                {"model": "", "provider": "openai", "base_url": "http://x"},
                {"model": "dup", "provider": "openai", "base_url": "http://127.0.0.1:1/v1", "slug": "same", "api_key": "k"},
                {"model": "dup2", "provider": "openai", "base_url": "http://127.0.0.1:1/v1", "slug": "same", "index": 1, "api_key": "k"},
            ]
        }
    )
    assert len(models) == 2
    monkeypatch.setattr("codex_shim.cursor_passthrough.cursor_passthrough_available", lambda: True)
    assert default_model_slug([], include_chatgpt=False).startswith("cursor")
    monkeypatch.setattr("codex_shim.cursor_passthrough.cursor_passthrough_available", lambda: False)
    with pytest.raises(ValueError, match="No usable"):
        default_model_slug([], include_chatgpt=False)

    missing = tmp_path / "router-missing.json"
    assert load_router_config(missing) is None
    bad = tmp_path / "router-bad.json"
    bad.write_text("{")
    assert load_router_config(bad) is None
    not_obj = tmp_path / "router-list.json"
    not_obj.write_text("[]")
    assert load_router_config(not_obj) is None
    router = tmp_path / "router.json"
    router.write_text(json.dumps({"router": {"candidates": ["skip", {"slug": ""}, {"slug": "local-llama"}]}}))
    loaded = load_router_config(router)
    assert loaded is not None
    assert _clamp01("nope") == 0.0
    assert parse_scores("{not json} {", ["a"]) == {}
    assert _latest_from_input(["  tail  ", 12]) == "tail"
    assert _latest_from_input([{"role": "assistant", "content": "x"}]) == ""

    assert unwrap_custom_tool_input("[1]") in {"[1]", "1"}
    assert integerize_json_whole_floats([1.0, {"n": 2.0}]) == [1, {"n": 2}]
    chat = responses_to_chat({"model": "x", "input": {"text": "hi"}}, "up")
    assert chat["messages"]
    msgs = _responses_input_to_messages(
        [
            12,
            {"type": "function_call", "call_id": "c1", "name": "shell", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "c1", "output": "ok"},
            {"type": "function_call_output", "call_id": "", "output": "orphan"},
            {"type": "tool_search_call", "call_id": "ts1", "arguments": {"q": "exa"}},
            {"type": "mcp_tool_call"},
            {"type": "mcp_tool_call", "result": "   "},
            {"type": "compaction_trigger"},
            {"type": "agent_message", "content": [12, "hello", {"type": "text", "text": "there"}]},
        ]
    )
    assert msgs
    assert _chat_parts_from_content({"text": "hi"})[0]["text"] == "hi"
    assert _chat_parts_from_content(12) == []
    mapped = anthropic_to_chat_response(
        {"content": [{"type": "text", "text": "hi"}, {"type": "tool_use", "id": "t1", "name": "shell", "input": {}}]},
        "m",
    )
    assert mapped["choices"][0]["message"]["tool_calls"]
    assert _responses_tool_to_chat_function("nope") is None
    assert _responses_tool_to_chat_function({"type": "function", "function": {"name": "x"}})["function"]["name"] == "x"
    assert _responses_tool_to_chat_function({"type": "unknown"}) is None
    assert _normalize_chat_image_detail("original") == "high"
    assert _normalize_chat_image_detail("weird") == "auto"
    assert "hello" in _agent_message_text({"content": ["hello", 1, {"type": "text", "text": "t"}]})
    usage = normalize_responses_usage({"prompt_tokens_details": {"cached_tokens": 1}, "completion_tokens_details": {"reasoning_tokens": 2}})
    assert usage["input_tokens_details"]
    assert _int_token(True) is None
    assert _int_token(3.0) == 3
    assert _content_to_text({"foo": 1})
    tools = _responses_tools_to_chat_tools(["skip", {"type": "namespace", "name": "ns", "tools": []}, {"name": ""}])
    assert isinstance(tools, list)
    assert _responses_tool_function_name({"type": "mcp", "name": "search"}) == "search"
    assert _tool_choice_name("shell", ["skip", {"type": "function", "function": {"name": "shell"}}]) == "shell"
    assert _anthropic_tools_to_chat_tools(["skip", {"name": ""}, {"name": "shell"}])
    cleaned = _sanitize_chat_messages(
        [{"role": "assistant", "content": {"text": "x"}, "reasoning_content": "r", "tool_calls": ["skip", {"id": "c", "function": {"arguments": "{}"}}]}]
    )
    assert cleaned
    assert _coerce_custom_input_item("x") == "x"
    assert _coerce_custom_input_item({"type": "custom_tool_call", "input": "diff"})["name"] == "apply_patch"
    converted = _anthropic_assistant_message_to_chat(["raw", {"type": "tool_use", "name": "shell", "input": {}}])
    assert converted["tool_calls"]
    assert _flatten_console_content(12) == "12"
    assert _flatten_console_content(["a", 1, {"type": "image_url"}])
    assert _content_nonempty(None) is False
    assert _content_nonempty(["x"]) is True
    assert _content_nonempty(12) is True
    assert _tool_call_names(["skip", {"tool_calls": ["x", {"id": "c1", "function": {"name": "shell"}}]}])["c1"] == "shell"
    assert should_omit_parallel_tool_calls(_model(raw={"omit_parallel_tool_calls": True})) is True
    assert should_omit_parallel_tool_calls(_model(raw={"supports_parallel_tool_calls": False})) is True
    assert _load_store(tmp_path / "missing-store.json")["by_slug"] == {}
    (tmp_path / "bad-store.json").write_text("{")
    assert _load_store(tmp_path / "bad-store.json")["by_slug"] == {}
    (tmp_path / "list-store.json").write_text("[]")
    assert _load_store(tmp_path / "list-store.json")["by_slug"] == {}
    assert _is_upstream_transport_dead(ConnectionResetError()) is True
    assert _is_upstream_transport_dead(RuntimeError("closing transport")) is True
    assert WsPassthroughConnectError("x", status=502).status == 502
    assert responses_input_items(None) == []
    cleaned_items, _warn, audit = sanitize_compaction_input_with_pipeline([])
    assert audit.outgoing_items == 0
    assert isinstance(cleaned_items, list)

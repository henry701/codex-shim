from __future__ import annotations

import asyncio

import pytest

from codex_shim.compaction.config import CompactionSettings
from codex_shim.compaction.orchestrator import CompactionOrchestrator, native_item_from_payload
from codex_shim.compaction.types import (
    CompactionAdapters,
    CompactionRequest,
    NativeAttemptResult,
    SummarizationAttemptResult,
)
from codex_shim.net.sse import ClientDisconnected
from codex_shim.net.stream_guard import StreamGuard
from codex_shim.translate import (
    _agent_message_to_chat_message,
    _anthropic_tool_choice_to_chat,
    _chat_image_part,
    _content_to_text,
    _responses_content_to_chat_content,
    _responses_tool_function_name,
    _tool_choice_name,
    chat_to_anthropic,
    normalize_responses_usage,
    original_responses_tool_type,
    responses_to_anthropic,
    responses_to_chat,
    responses_tool_type_map,
)
from codex_shim import upstream_io_trace as io_trace


def _user(text: str) -> dict:
    return {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}


def _usable(text: str = "Ship a local compaction module that never drops user work.") -> str:
    return f"## Goal\n- {text}\n\n" + ("progress notes " * 40)


class _Resp:
    prepared = True

    async def write(self, data):
        del data

    async def write_eof(self):
        return None


def test_upstream_io_trace_branches(tmp_path, monkeypatch):
    monkeypatch.setattr(io_trace, "UPSTREAM_IO_DIR", tmp_path / "upstream-io")
    monkeypatch.setattr(io_trace, "REQUEST_BODY_FILE_LIMIT", 20)
    assert io_trace._summarize_input_item("x") == "?"
    assert "query" in io_trace._summarize_input_item({"type": "web_search_call", "action": {"query": "hello"}})
    assert "server" in io_trace._summarize_input_item({"type": "mcp_tool_call", "server": "exa", "tool": "search"})
    circ = {}
    circ["x"] = circ
    assert io_trace._body_json_size(circ) == 0
    assert io_trace._body_for_error_file(None, verbose=True) is None
    truncated = io_trace._body_for_error_file({"model": "m", "input": [_user("hi" * 40)]}, verbose=True)
    assert truncated["_truncated"] is True
    monkeypatch.setenv("CODEX_SHIM_REQUEST_LOG", "1")
    io_trace.log_upstream_request("chatgpt-passthrough", "https://example.test", {"model": "m", "input": [_user("hi")]})
    io_trace.record_upstream_error("chatgpt", "https://example.test", 429, "slow", request_body={"model": "m"})
    assert (tmp_path / "upstream-io" / "last-request.json").exists()
    blocked = tmp_path / "blocked"
    blocked.write_text("not-a-dir")
    monkeypatch.setattr(io_trace, "UPSTREAM_IO_DIR", blocked)
    io_trace.record_upstream_error("chatgpt", "https://example.test", 500, "x")
    io_trace._write_request_artifact("chatgpt", "https://example.test", {"model": "m"})


async def test_orchestrator_tertiary_and_provider_adapters():
    usable = _usable()
    items = [_user("Ship a local compaction module that never drops user work.")]

    async def unused(*a, **k):
        raise AssertionError("unexpected")

    async def native_fail(request, prepared):
        del request, prepared
        return NativeAttemptResult(item=None, native_message="")

    async def summary_short(request, prepared, native_message):
        del request, prepared, native_message
        return SummarizationAttemptResult(summary="short")

    async def summary_ok(request, prepared, native_message):
        del request, prepared, native_message
        return SummarizationAttemptResult(summary=usable)

    async def tertiary_ok(request, prepared, native_message, slug):
        del request, prepared, native_message
        assert slug == "cheap"
        return SummarizationAttemptResult(summary=usable)

    async def tertiary_bad(request, prepared, native_message, slug):
        del request, prepared, native_message, slug
        return SummarizationAttemptResult(summary="not a goal")

    orch = CompactionOrchestrator(
        CompactionAdapters(
            native_chatgpt=native_fail,
            native_cursor=native_fail,
            native_byok=native_fail,
            summarization_chatgpt=summary_ok,
            summarization_cursor=summary_ok,
            summarization_byok=summary_short,
            tertiary_byok=tertiary_ok,
            acquire_chatgpt_lock=unused,
        )
    )
    chatgpt = await orch.run(
        CompactionRequest(
            http_request=object(),
            body={"model": "codex-x"},
            stripped_input=items,
            requested_slug="codex-x",
            provider="chatgpt",
            skip_native=True,
            preset_native_message="native failed",
        )
    )
    assert chatgpt.phase == "summarization"
    cursor = await orch.run(
        CompactionRequest(
            http_request=object(),
            body={"model": "cursor-x"},
            stripped_input=items,
            requested_slug="cursor-x",
            provider="cursor",
            skip_native=True,
            preset_native_message="native failed",
        )
    )
    assert cursor.phase == "summarization"
    tertiary = await orch.run(
        CompactionRequest(
            http_request=object(),
            body={"model": "m"},
            stripped_input=items,
            requested_slug="m",
            provider="byok",
            skip_native=True,
            preset_native_message="native failed",
            settings=CompactionSettings(fallback_enabled=True, tertiary_fallback_slug="cheap"),
        )
    )
    assert tertiary.phase == "tertiary"
    orch._adapters.tertiary_byok = tertiary_bad
    local = await orch.run(
        CompactionRequest(
            http_request=object(),
            body={"model": "m"},
            stripped_input=items,
            requested_slug="m",
            provider="byok",
            skip_native=True,
            preset_native_message="native failed",
            settings=CompactionSettings(fallback_enabled=True, tertiary_fallback_slug="cheap"),
        )
    )
    assert local.phase == "local_fallback"
    skipped = await orch.run(
        CompactionRequest(
            http_request=object(),
            body={"model": "m"},
            stripped_input=items,
            requested_slug="m",
            provider="byok",
            skip_native=True,
            preset_native_message="native failed",
            settings=CompactionSettings(fallback_enabled=True, tertiary_fallback_slug="cheap"),
            route_fn=lambda body: (_ for _ in ()).throw(RuntimeError("route")),
            has_credentials_fn=lambda route: True,
        )
    )
    assert skipped.phase == "local_fallback"
    item = native_item_from_payload(
        {"output": [{"type": "compaction", "encrypted_content": "abc", "id": "i1", "status": "completed"}]}
    )
    assert item is not None and item["type"] == "compaction"


def test_translate_remaining_branches():
    responses_tool_type_map(
        [{"type": "namespace", "name": "", "tools": [{"type": "function", "name": "x"}]}]
    )
    original_responses_tool_type("search", {"search": "web_search_preview"})
    chat_to_anthropic(
        {
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "hi"},
            ]
        },
        "claude",
        16,
    )
    responses_to_anthropic(
        {
            "input": [
                {"type": "message", "role": "system", "content": "s"},
                {"type": "message", "role": "user", "content": "u"},
            ]
        },
        "claude",
        16,
    )
    normalize_responses_usage(
        {
            "prompt_tokens": 1,
            "completion_tokens": 2,
            "input_tokens_details": {"cached_tokens": 1},
            "output_tokens_details": {"reasoning_tokens": 1},
        }
    )
    responses_to_chat(
        {
            "model": "m",
            "input": [
                {"type": "mcp_tool_call", "result": "", "server": "exa", "tool": "s"},
                {"type": "mcp_tool_call", "result": "ok", "server": "exa", "tool": "s"},
            ],
        },
        "m",
    )
    assert _agent_message_to_chat_message({}) is None
    assert _responses_content_to_chat_content([]) == ""
    assert _chat_image_part({"type": "input_image"}) is None
    assert _content_to_text(123) == "123"
    assert _responses_tool_function_name({"type": "mcp"})
    assert _tool_choice_name("mystery", []) == "mystery"
    assert _anthropic_tool_choice_to_chat({"type": "auto"}) == "auto"


async def test_stream_guard_cancel_and_log_end():
    class CancelComplete:
        already_emitted = False

        async def complete(self, response, *, upstream_saw_done):
            del response, upstream_saw_done
            raise asyncio.CancelledError()

        async def fail(self, response, message, *, code):
            del response, message, code
            raise ClientDisconnected()

    with pytest.raises(asyncio.CancelledError):
        async with StreamGuard(_Resp(), CancelComplete(), label="cancel-complete", keepalive=False):
            pass

    class FailDisc(CancelComplete):
        async def complete(self, response, *, upstream_saw_done):
            del response, upstream_saw_done
            return "done"

    guard = StreamGuard(_Resp(), FailDisc(), label="fail-disc", keepalive=False)
    with pytest.raises(ClientDisconnected):
        await guard._fail_terminal("m", "c")

    class FailBoom(CancelComplete):
        async def fail(self, response, message, *, code):
            del response, message, code
            raise RuntimeError("fail boom")

        async def complete(self, response, *, upstream_saw_done):
            del response, upstream_saw_done
            return "done"

    await StreamGuard(_Resp(), FailBoom(), label="fail-boom", keepalive=False)._fail_terminal("m", "c")

    class Soft:
        already_emitted = False

        async def complete(self, response, *, upstream_saw_done):
            del response, upstream_saw_done
            return "done"

        async def fail(self, response, message, *, code):
            del response, message, code
            return "fail"

    guard = StreamGuard(_Resp(), Soft(), label="log", keepalive=False)
    guard._log_end = lambda: (_ for _ in ()).throw(RuntimeError("log"))
    async with guard:
        pass

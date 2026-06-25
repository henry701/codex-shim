from __future__ import annotations

import pytest

from codex_shim.translate import (
    HOSTED_WEB_SEARCH_UNAVAILABLE_MESSAGE,
    _format_web_search_result_entries,
    _is_absolutely_empty_web_search_output,
    _is_hosted_web_search_name,
    _web_search_call_result_text,
    prepare_codex_byok_responses_body,
    responses_to_chat,
)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("web_search", True),
        ("web_search_preview", True),
        ("WEB_SEARCH", True),
        ("web_search_exa", False),
        ("mcp__exa__web_search_exa", False),
        ("", False),
    ],
)
def test_is_hosted_web_search_name(name: str, expected: bool) -> None:
    assert _is_hosted_web_search_name(name) is expected


@pytest.mark.parametrize(
    "output",
    [
        None,
        "",
        [],
        {},
        [{}],
        [{"type": "output_text", "text": ""}],
        [{"type": "output_text", "text": "   "}],
        {"type": "output_text", "text": ""},
        {"type": "web_search_call", "status": "completed", "call_id": "ws_1"},
    ],
)
def test_is_absolutely_empty_web_search_output_true(output: object) -> None:
    assert _is_absolutely_empty_web_search_output(output) is True


@pytest.mark.parametrize(
    "output",
    [
        "result text",
        [{"type": "output_text", "text": "found something"}],
        {"type": "output_text", "text": "found something"},
        [{"title": "Example", "snippet": "Details"}],
    ],
)
def test_is_absolutely_empty_web_search_output_false(output: object) -> None:
    assert _is_absolutely_empty_web_search_output(output) is False


def test_format_web_search_result_entries_covers_result_shapes() -> None:
    entries = [
        "plain string hit",
        42,
        {"title": "Title A", "snippet": "Snippet A"},
        {"snippet": "Snippet only"},
        {"title": "Title B", "url": "https://example.com/b"},
        {"title": "Title C"},
        {"url": "https://example.com/d"},
        {"type": "url", "url": "https://example.com/e"},
    ]
    parts = _format_web_search_result_entries(entries)
    assert parts == [
        "plain string hit",
        "Title A\nSnippet A",
        "Snippet only",
        "Title B\nhttps://example.com/b",
        "Title C",
        "https://example.com/d",
        "https://example.com/e",
    ]


def test_web_search_call_result_text_merges_sources_results_and_output() -> None:
    item = {
        "type": "web_search_call",
        "action": {
            "sources": [{"url": "https://example.com/source"}],
            "results": [{"title": "Result title", "snippet": "Result snippet"}],
        },
        "results": [{"title": "Top-level", "snippet": "Top-level snippet"}],
        "output": [{"type": "output_text", "text": "Inline output text"}],
    }
    text = _web_search_call_result_text(item)
    assert "https://example.com/source" in text
    assert "Result title\nResult snippet" in text
    assert "Top-level\nTop-level snippet" in text
    assert "Inline output text" in text


def test_web_search_call_with_action_results_preserves_snippets() -> None:
    body = {
        "model": "slug",
        "input": [
            {
                "type": "web_search_call",
                "call_id": "ws_snippets",
                "status": "completed",
                "action": {
                    "type": "search",
                    "query": "linux caffeinate",
                    "results": [
                        {
                            "title": "systemd-inhibit",
                            "snippet": "Built-in Linux alternative to macOS caffeinate",
                        }
                    ],
                },
            }
        ],
    }
    out = responses_to_chat(body, "upstream")
    tool_message = out["messages"][-1]
    assert tool_message["role"] == "tool"
    assert "systemd-inhibit" in tool_message["content"]
    assert "unavailable" not in tool_message["content"].lower()


def test_web_search_function_call_output_with_content_is_preserved() -> None:
    body = {
        "model": "slug",
        "input": [
            {
                "type": "function_call",
                "name": "web_search",
                "call_id": "call_ws",
                "arguments": '{"query":"linux caffeinate"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_ws",
                "output": [{"type": "output_text", "text": "Title: Caffeine\nSnippet: keep awake"}],
            },
        ],
    }
    out = responses_to_chat(body, "upstream")
    tool_message = out["messages"][-1]
    assert tool_message["content"] == "Title: Caffeine\nSnippet: keep awake"


@pytest.mark.parametrize("tool_name", ["web_search", "web_search_preview"])
def test_empty_function_call_output_for_hosted_web_search_names(tool_name: str) -> None:
    body = {
        "model": "slug",
        "input": [
            {
                "type": "function_call",
                "name": tool_name,
                "call_id": "call_ws",
                "arguments": '{"query":"linux caffeinate"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_ws",
                "output": [],
            },
        ],
    }
    out = responses_to_chat(body, "upstream")
    assert out["messages"][-1]["content"] == HOSTED_WEB_SEARCH_UNAVAILABLE_MESSAGE


def test_function_call_output_before_function_call_still_matches_call_id() -> None:
    body = {
        "model": "slug",
        "input": [
            {
                "type": "function_call_output",
                "call_id": "call_ws",
                "output": "",
            },
            {
                "type": "function_call",
                "name": "web_search",
                "call_id": "call_ws",
                "arguments": '{"query":"linux caffeinate"}',
            },
        ],
    }
    out = responses_to_chat(body, "upstream")
    roles = [m["role"] for m in out["messages"]]
    assert roles == ["assistant", "tool"]
    tool_messages = [m for m in out["messages"] if m["role"] == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0]["content"] == HOSTED_WEB_SEARCH_UNAVAILABLE_MESSAGE


def test_byok_prepare_and_translate_empty_web_search_round_trip() -> None:
    body = {
        "model": "local-llama",
        "input": [
            {"role": "user", "content": "Search the web for linux caffeinate"},
            {"type": "reasoning", "summary": [{"type": "summary_text", "text": "Need web search"}]},
            {
                "type": "function_call",
                "name": "web_search",
                "call_id": "call_ws",
                "arguments": '{"query":"linux caffeinate equivalent"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_ws",
                "output": "",
            },
        ],
        "tools": [
            {"type": "web_search_preview"},
            {"type": "function", "name": "tool_search", "parameters": {"type": "object"}},
        ],
    }
    prepared = prepare_codex_byok_responses_body(body, {"User-Agent": "codex-cli/0.135.0"})
    out = responses_to_chat(prepared, "local-llama")

    assert [tool["function"]["name"] for tool in out["tools"]] == ["tool_search"]
    tool_messages = [m for m in out["messages"] if m["role"] == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0]["content"] == HOSTED_WEB_SEARCH_UNAVAILABLE_MESSAGE
    assistant_with_tools = [m for m in out["messages"] if m.get("tool_calls")]
    assert assistant_with_tools[-1]["tool_calls"][0]["function"]["name"] == "web_search"


def test_multiple_empty_web_search_calls_each_get_unavailable_message() -> None:
    body = {
        "model": "slug",
        "input": [
            {
                "type": "web_search_call",
                "call_id": "ws_one",
                "status": "completed",
                "action": {"type": "search", "query": "first query"},
            },
            {
                "type": "web_search_call",
                "call_id": "ws_two",
                "status": "completed",
                "action": {"type": "search", "query": "second query"},
            },
        ],
    }
    out = responses_to_chat(body, "upstream")
    tool_messages = [m for m in out["messages"] if m["role"] == "tool"]
    assert len(tool_messages) == 2
    assert all(m["content"] == HOSTED_WEB_SEARCH_UNAVAILABLE_MESSAGE for m in tool_messages)


def test_function_call_output_after_intervening_message_keeps_tool_order() -> None:
    body = {
        "model": "slug",
        "input": [
            {
                "type": "function_call",
                "name": "web_search",
                "call_id": "call_ws",
                "arguments": '{"query":"linux caffeinate"}',
            },
            {"role": "user", "content": "continue"},
            {
                "type": "function_call_output",
                "call_id": "call_ws",
                "output": "",
            },
        ],
    }
    out = responses_to_chat(body, "upstream")
    roles = [m["role"] for m in out["messages"]]
    assert roles.index("assistant") < roles.index("tool")
    assert out["messages"][roles.index("tool")]["content"] == HOSTED_WEB_SEARCH_UNAVAILABLE_MESSAGE

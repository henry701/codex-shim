from __future__ import annotations

import json

from codex_shim import mcp_search
from codex_shim.translate import responses_to_chat


def test_mcp_tool_search_definition_is_nested_chat_format():
    fn = mcp_search.MCP_TOOL_SEARCH_DEFINITION["function"]
    assert fn["name"] == "tool_search_call"
    assert "query" in fn["parameters"]["properties"]
    assert "query" in fn["parameters"]["required"]


def test_responses_to_chat_injects_tool_search_when_mcp_present():
    body = {
        "model": "slug",
        "instructions": "Be brief.",
        "input": [{"role": "user", "content": "hi"}],
        "tools": [
            {"type": "function", "name": "shell_command", "description": "Run shell",
             "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
            {"type": "function", "name": "mcp__jina", "description": "MCP server"},
        ],
    }
    out = responses_to_chat(body, "gemma-4-E4B")
    names = [t["function"]["name"] for t in out["tools"]]
    assert names[0] == "tool_search_call"
    assert "mcp__jina" not in names
    assert "shell_command" in names
    assert "MCP tool-calling convention" in out["messages"][0]["content"]


def test_responses_to_chat_keeps_full_mcp_tool_names():
    body = {
        "model": "slug",
        "input": [{"role": "user", "content": "hi"}],
        "tools": [
            {"type": "function", "name": "mcp__jina", "description": "MCP server"},
            {"type": "function", "name": "mcp__jina__read_url", "description": "Read URL",
             "parameters": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}},
        ],
    }
    out = responses_to_chat(body, "gemma-4-E4B")
    names = [t["function"]["name"] for t in out["tools"]]
    assert "tool_search_call" in names
    assert "mcp__jina" not in names
    assert "mcp__jina__read_url" in names


def test_responses_to_chat_omits_tool_search_when_no_mcp():
    body = {
        "model": "slug",
        "input": [{"role": "user", "content": "hi"}],
        "tools": [
            {"type": "function", "name": "shell_command", "description": "Run shell",
             "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
        ],
    }
    out = responses_to_chat(body, "gemma-4-E4B")
    names = [t["function"]["name"] for t in out["tools"]]
    assert "tool_search_call" not in names
    assert names == ["shell_command"]


def test_format_tool_search_result_prefixes_full_names():
    out = mcp_search.format_tool_search_result(
        "mcp__jina",
        [{"name": "read_url", "description": "Read a URL."}],
    )
    parsed = json.loads(out)
    assert parsed["server"] == "mcp__jina"
    assert parsed["tool_count"] == 1
    assert parsed["tools"][0]["name"] == "mcp__jina__read_url"


def test_format_tool_search_result_keeps_already_prefixed():
    out = mcp_search.format_tool_search_result(
        "mcp__jina",
        [{"name": "mcp__jina__read_url", "description": "Read a URL."}],
    )
    parsed = json.loads(out)
    assert parsed["tools"][0]["name"] == "mcp__jina__read_url"


def test_resolve_mcp_url_uses_fallback_for_known_servers():
    assert mcp_search.resolve_mcp_url("mcp__exa") is not None
    assert mcp_search.resolve_mcp_url("mcp__jina") is None
    assert mcp_search.resolve_mcp_url("mcp__unknown_server_xyz") is None

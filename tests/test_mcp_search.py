from __future__ import annotations

import json

from codex_shim import mcp_search
from codex_shim.translate import responses_to_chat


def test_responses_to_chat_translates_native_tool_search():
    body = {
        "model": "slug",
        "input": [{"role": "user", "content": "hi"}],
        "tools": [
            {
                "type": "tool_search",
                "execution": "client",
                "description": "Search deferred tools",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
            {"type": "function", "name": "shell_command", "description": "Run shell",
             "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
        ],
    }
    out = responses_to_chat(body, "gemma-4-E4B")
    names = [t["function"]["name"] for t in out["tools"]]
    assert "tool_search" in names
    assert "tool_search_call" not in names
    assert "shell_command" in names
    tool_search = next(t for t in out["tools"] if t["function"]["name"] == "tool_search")
    assert tool_search["function"]["description"] == "Search deferred tools"
    assert "query" in tool_search["function"]["parameters"]["properties"]


def test_responses_to_chat_omits_deferred_mcp_server_stubs():
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
    assert "tool_search_call" not in names
    assert "tool_search" not in names
    assert "mcp__jina" not in names
    assert names == ["shell_command"]
    assert out["messages"][0]["content"].startswith("Be brief.")


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
    assert "tool_search_call" not in names
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
    assert "tool_search" not in names
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


def test_resolve_mcp_url_reads_config_only():
    assert mcp_search.resolve_mcp_url("mcp__exa") is None or isinstance(
        mcp_search.resolve_mcp_url("mcp__exa"), str
    )
    assert mcp_search.resolve_mcp_url("mcp__unknown_server_xyz") is None


def test_responses_tools_need_tool_search():
    assert mcp_search.responses_tools_need_tool_search(
        [{"type": "tool_search", "execution": "client"}]
    )
    assert mcp_search.responses_tools_need_tool_search(
        [{"type": "function", "name": "mcp__exa"}]
    )
    assert not mcp_search.responses_tools_need_tool_search(
        [{"type": "function", "name": "shell_command"}]
    )


def test_is_deferred_mcp_server_stub():
    assert mcp_search.is_deferred_mcp_server_stub("mcp__exa")
    assert not mcp_search.is_deferred_mcp_server_stub("mcp__exa__web_search_exa")
    assert not mcp_search.is_deferred_mcp_server_stub("shell_command")


def test_is_mcp_tool_call_parses_full_name():
    assert mcp_search.is_mcp_tool_call("mcp__exa__web_search_exa") == "mcp__exa"
    assert mcp_search.is_mcp_tool_call("mcp__context7__get_docs") == "mcp__context7"
    assert mcp_search.is_mcp_tool_call("mcp__exa") is None
    assert mcp_search.is_mcp_tool_call("shell_command") is None
    assert mcp_search.is_mcp_tool_call("mcp__exa__") is None
    assert mcp_search.is_mcp_tool_call("") is None


def test_is_tool_search_call():
    assert mcp_search.is_tool_search_call("tool_search")
    assert mcp_search.is_tool_search_call("tool_search_call")
    assert not mcp_search.is_tool_search_call("mcp__exa__web_search_exa")
    assert not mcp_search.is_tool_search_call("exec_command")
    assert not mcp_search.is_tool_search_call("mcp__exa")


def test_normalize_upstream_tool_name_is_passthrough():
    assert mcp_search.normalize_upstream_tool_name("web_search_exa") == "web_search_exa"
    assert mcp_search.normalize_upstream_tool_name("web_search") == "web_search"
    assert mcp_search.normalize_upstream_tool_name("exec_command") == "exec_command"


def test_flatten_tool_search_namespace_tools():
    flattened = mcp_search.flatten_tool_search_tools(
        [
            {
                "type": "namespace",
                "name": "mcp__exa",
                "tools": [
                    {
                        "type": "function",
                        "name": "web_search_exa",
                        "description": "Search the web via Exa.",
                        "parameters": {
                            "type": "object",
                            "properties": {"query": {"type": "string"}},
                            "required": ["query"],
                        },
                    }
                ],
            }
        ]
    )
    assert len(flattened) == 1
    assert flattened[0]["name"] == "mcp__exa__web_search_exa"
    assert flattened[0]["parameters"]["required"] == ["query"]


def test_flatten_tool_search_full_name_tools():
    flattened = mcp_search.flatten_tool_search_tools(
        [
            {
                "type": "function",
                "name": "mcp__exa__web_search_exa",
                "description": "Search the web via Exa.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            }
        ]
    )
    assert flattened[0]["name"] == "mcp__exa__web_search_exa"


def test_responses_to_chat_does_not_inject_discovered_tools_into_tools_array():
    body = {
        "model": "slug",
        "input": [
            {"role": "user", "content": "news"},
            {
                "type": "tool_search_output",
                "call_id": "search-1",
                "tools": [
                    {
                        "type": "namespace",
                        "name": "mcp__exa",
                        "tools": [
                            {
                                "type": "function",
                                "name": "web_search_exa",
                                "description": "Search the web via Exa.",
                                "parameters": {
                                    "type": "object",
                                    "properties": {"query": {"type": "string"}},
                                    "required": ["query"],
                                },
                            }
                        ],
                    }
                ],
            },
        ],
        "tools": [
            {"type": "tool_search", "execution": "client", "description": "Search deferred tools"},
            {"type": "function", "name": "exec_command", "description": "Run shell",
             "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}, "required": ["cmd"]}},
            {"type": "function", "name": "mcp__exa", "description": "Exa MCP server"},
        ],
    }
    out = responses_to_chat(body, "gemma-4-E4B")
    names = [t["function"]["name"] for t in out["tools"]]
    assert "mcp__exa__web_search_exa" not in names
    assert "mcp__exa" not in names
    assert "tool_search" in names
    assert "exec_command" in names
    tool_messages = [m for m in out["messages"] if m.get("role") == "tool"]
    assert "mcp__exa__web_search_exa" in tool_messages[0]["content"]

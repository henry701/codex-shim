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


def test_responses_to_chat_injects_pre_discovered_mcp_tools():
    body = {
        "model": "slug",
        "input": [{"role": "user", "content": "hi"}],
        "tools": [
            {"type": "function", "name": "mcp__exa", "description": "MCP server"},
            {"type": "function", "name": "shell_command", "description": "Run shell",
             "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
        ],
    }
    discovered = [
        {
            "type": "function",
            "function": {
                "name": "mcp__exa__web_search_exa",
                "description": "Search the web via exa",
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
            },
        }
    ]
    out = responses_to_chat(body, "gemma-4-E4B", discovered_mcp_tools=discovered)
    names = [t["function"]["name"] for t in out["tools"]]
    assert "mcp__exa__web_search_exa" in names
    assert "tool_search_call" in names
    assert "mcp__exa" not in names
    assert names.index("mcp__exa__web_search_exa") < names.index("tool_search_call")


def test_responses_to_chat_dedupes_pre_discovered_against_existing():
    body = {
        "model": "slug",
        "input": [{"role": "user", "content": "hi"}],
        "tools": [
            {"type": "function", "name": "mcp__exa", "description": "MCP server"},
            {"type": "function", "name": "mcp__exa__web_search_exa", "description": "Search the web via exa",
             "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
        ],
    }
    discovered = [
        {
            "type": "function",
            "function": {
                "name": "mcp__exa__web_search_exa",
                "description": "Duplicate",
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
            },
        }
    ]
    out = responses_to_chat(body, "gemma-4-E4B", discovered_mcp_tools=discovered)
    names = [t["function"]["name"] for t in out["tools"]]
    assert names.count("mcp__exa__web_search_exa") == 1


def test_pre_discover_mcp_tools_returns_chat_format_with_prefixes(monkeypatch):
    mcp_search.invalidate_discovery_cache()
    monkeypatch.setattr(
        mcp_search,
        "_iter_server_urls",
        lambda: [("mcp__testpre", "https://example.test/mcp")],
    )

    async def fake_call(url):
        return [
            {"name": "do_thing", "description": "Do a thing", "inputSchema": {"type": "object"}},
            {"name": "mcp__testpre__other", "description": "Already prefixed"},
        ]

    monkeypatch.setattr(mcp_search, "call_mcp_tools_list", fake_call)
    import asyncio
    tools = asyncio.run(mcp_search.pre_discover_mcp_tools(force=True))
    assert tools[0]["function"]["name"] == "mcp__testpre__do_thing"
    assert tools[1]["function"]["name"] == "mcp__testpre__other"
    assert tools[0]["function"]["description"] == "Do a thing"
    assert tools[0]["function"]["parameters"] == {"type": "object"}
    mcp_search.invalidate_discovery_cache()


def test_pre_discover_mcp_tools_caches_results(monkeypatch):
    mcp_search.invalidate_discovery_cache()
    call_count = {"n": 0}

    async def fake_call(url):
        call_count["n"] += 1
        return [{"name": "t", "description": "d", "inputSchema": {"type": "object"}}]

    monkeypatch.setattr(mcp_search, "call_mcp_tools_list", fake_call)
    monkeypatch.setattr(
        mcp_search,
        "_iter_server_urls",
        lambda: [("mcp__cached", "https://cached.test/mcp")],
    )
    import asyncio
    a = asyncio.run(mcp_search.pre_discover_mcp_tools(force=True))
    b = asyncio.run(mcp_search.pre_discover_mcp_tools())
    assert call_count["n"] == 1
    assert a[0]["function"]["name"] == "mcp__cached__t"
    assert b[0]["function"]["name"] == "mcp__cached__t"
    mcp_search.invalidate_discovery_cache()


def test_inject_tool_search_if_mcp_injects_discovered():
    from codex_shim.server import _inject_tool_search_if_mcp
    body = {
        "tools": [
            {"type": "function", "function": {"name": "mcp__exa", "description": "server"}},
            {"type": "function", "function": {"name": "shell_command"}},
        ]
    }
    discovered = [
        {
            "type": "function",
            "function": {
                "name": "mcp__exa__web_search_exa",
                "description": "Web search",
                "parameters": {"type": "object"},
            },
        }
    ]
    _inject_tool_search_if_mcp(body, discovered)
    names = [
        (t.get("function") or {}).get("name") or t.get("name")
        for t in body["tools"]
    ]
    assert "mcp__exa__web_search_exa" in names
    assert "tool_search_call" in names
    assert "mcp__exa" not in names
    assert "shell_command" in names


def test_inject_tool_search_if_mcp_no_op_without_mcp():
    from codex_shim.server import _inject_tool_search_if_mcp
    body = {"tools": [{"type": "function", "function": {"name": "shell_command"}}]}
    _inject_tool_search_if_mcp(body, [{"type": "function", "function": {"name": "x"}}])
    names = [(t.get("function") or {}).get("name") for t in body["tools"]]
    assert names == ["shell_command"]


def test_is_mcp_tool_call_parses_full_name():
    assert mcp_search.is_mcp_tool_call("mcp__exa__web_search_exa") == "mcp__exa"
    assert mcp_search.is_mcp_tool_call("mcp__context7__get_docs") == "mcp__context7"
    assert mcp_search.is_mcp_tool_call("mcp__exa") is None
    assert mcp_search.is_mcp_tool_call("shell_command") is None
    assert mcp_search.is_mcp_tool_call("mcp__exa__") is None
    assert mcp_search.is_mcp_tool_call("") is None


def test_is_shim_resolved_tool():
    assert mcp_search.is_shim_resolved_tool("tool_search_call")
    assert not mcp_search.is_shim_resolved_tool("mcp__exa__web_search_exa")
    assert not mcp_search.is_shim_resolved_tool("exec_command")
    assert not mcp_search.is_shim_resolved_tool("mcp__exa")

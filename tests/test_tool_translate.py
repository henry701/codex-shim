from codex_shim import mcp_search, tool_translate


def test_parse_mcp_function_name():
    assert mcp_search.parse_mcp_function_name("mcp__exa__web_search_exa") == (
        "exa",
        "web_search_exa",
    )


def test_mcp_function_call_item_matches_passthrough_shape():
    item = tool_translate.mcp_function_call_item(
        "call_1",
        "exa",
        "web_search_exa",
        '{"query":"hello"}',
        "completed",
    )
    assert item == {
        "id": "call_1",
        "type": "function_call",
        "status": "completed",
        "call_id": "call_1",
        "name": "web_search_exa",
        "namespace": "mcp__exa",
        "arguments": '{"query":"hello"}',
    }


def test_mcp_function_call_from_name():
    item = tool_translate.mcp_function_call_from_name(
        "call_2",
        "mcp__exa__web_search_exa",
        '{"query":"ukraine"}',
        "in_progress",
    )
    assert item is not None
    assert item["namespace"] == "mcp__exa"
    assert item["name"] == "web_search_exa"


def test_is_shim_resolved_tool_only_tool_search():
    assert mcp_search.is_shim_resolved_tool("tool_search_call")
    assert not mcp_search.is_shim_resolved_tool("mcp__exa__web_search_exa")

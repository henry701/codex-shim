from codex_shim import mcp_search, tool_translate


def test_parse_mcp_function_name():
    assert mcp_search.parse_mcp_function_name("mcp__exa__web_search_exa") == (
        "exa",
        "web_search_exa",
    )


def test_parse_mcp_tool_reference_accepts_dot_notation():
    assert mcp_search.parse_mcp_tool_reference("mcp__exa.web_search_exa") == (
        "exa",
        "web_search_exa",
    )
    assert mcp_search.parse_mcp_tool_reference("mcp__exa__web_search_exa") == (
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


def test_tool_search_call_item_matches_passthrough_shape():
    item = tool_translate.tool_search_call_item(
        "call_search",
        {"query": "mcp__exa"},
        "completed",
    )
    assert item["type"] == "tool_search_call"
    assert item["execution"] == "client"
    assert item["arguments"] == {"query": "mcp__exa", "limit": 10}


def test_tool_search_call_from_raw():
    item = tool_translate.tool_search_call_from_raw(
        "call_search",
        '{"query":"mcp__jina"}',
        "in_progress",
    )
    assert item["arguments"]["query"] == "mcp__jina"
    assert item["arguments"]["limit"] == 10

"""Translate upstream MCP function calls into Codex-native Responses items.

Passthrough stores MCP calls as ``function_call`` items with a ``namespace``
field (e.g. ``namespace: "mcp__exa"``, ``name: "web_search_exa"``). Codex CLI
executes those locally and renders them as ``mcp_tool_call`` in ``--json`` output.
"""

from __future__ import annotations

from . import mcp_search


def mcp_namespace(server: str) -> str:
    return server if server.startswith("mcp__") else f"mcp__{server}"


def mcp_function_call_item(
    item_id: str,
    server: str,
    tool: str,
    raw_arguments: str,
    status: str,
) -> dict[str, Any]:
    return {
        "id": item_id,
        "type": "function_call",
        "status": status,
        "call_id": item_id,
        "name": tool,
        "namespace": mcp_namespace(server),
        "arguments": raw_arguments,
    }


def mcp_function_call_from_name(
    item_id: str,
    function_name: str,
    raw_arguments: str,
    status: str,
) -> dict[str, Any] | None:
    parsed = mcp_search.parse_mcp_function_name(function_name)
    if parsed is None:
        return None
    server, tool = parsed
    return mcp_function_call_item(item_id, server, tool, raw_arguments, status)

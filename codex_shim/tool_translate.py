"""Translate upstream tool calls into Codex-native Responses items.

Passthrough stores MCP invocations as namespaced ``function_call`` items and
MCP discovery as ``tool_search_call`` items. Codex CLI/Desktop executes both
locally (``mcp_tool_call`` / ``tool_search_output`` in ``--json`` output).
"""

from __future__ import annotations

import json
from typing import Any

from . import mcp_search


def mcp_namespace(server: str) -> str:
    return server if server.startswith("mcp__") else f"mcp__{server}"


def parse_tool_arguments(raw_args: Any) -> dict[str, Any]:
    if isinstance(raw_args, str):
        try:
            parsed = json.loads(raw_args) if raw_args.strip() else {}
        except json.JSONDecodeError:
            return {"_raw": raw_args}
        return parsed if isinstance(parsed, dict) else {"_value": parsed}
    if isinstance(raw_args, dict):
        return raw_args
    return {}


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


def tool_search_call_item(
    call_id: str,
    arguments: dict[str, Any],
    status: str,
) -> dict[str, Any]:
    args = dict(arguments)
    if "limit" not in args:
        args["limit"] = 10
    return {
        "id": call_id,
        "type": "tool_search_call",
        "status": status,
        "call_id": call_id,
        "execution": "client",
        "arguments": args,
    }


def tool_search_call_from_raw(
    call_id: str,
    raw_arguments: str,
    status: str,
) -> dict[str, Any]:
    return tool_search_call_item(call_id, parse_tool_arguments(raw_arguments), status)

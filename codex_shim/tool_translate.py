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


def responses_function_call_ids(raw_id: str | None) -> tuple[str, str]:
    """ChatGPT Responses Lite requires ``function_call.id`` to start with ``fc``.

    ``call_id`` stays on the Chat Completions ``call_`` prefix so Desktop can
    pair ``function_call_output.call_id`` with the originating tool call.
    """
    token = str(raw_id or "").strip() or "0"
    if token.startswith("call_"):
        suffix = token[5:] or "0"
    elif token.startswith("fc_"):
        suffix = token[3:] or "0"
        if suffix.startswith("call_"):
            suffix = suffix[5:] or "0"
    else:
        suffix = token
    return f"fc_{suffix}", f"call_{suffix}"


def apply_function_call_ids(item: dict[str, Any]) -> dict[str, Any]:
    if item.get("type") != "function_call":
        return item
    raw = item.get("call_id") or item.get("id")
    item_id, call_id = responses_function_call_ids(None if raw is None else str(raw))
    out = dict(item)
    out["id"] = item_id
    out["call_id"] = call_id
    return out


def strip_function_call_output_item_id(item: dict[str, Any]) -> dict[str, Any]:
    """Drop leaked ``call_``/``fc_`` ids from ``function_call_output`` items.

    Lite validates ``function_call.id`` as ``fc_*``. A copied ``call_*`` string
    on the output item's ``id`` trips the same prefix check; ``call_id`` is the
    correlator and must stay.
    """
    if item.get("type") != "function_call_output":
        return item
    oid = item.get("id")
    if isinstance(oid, str) and (oid.startswith("call_") or oid.startswith("fc_")):
        out = dict(item)
        out.pop("id", None)
        return out
    return item


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
    item_id, call_id = responses_function_call_ids(item_id)
    return {
        "id": item_id,
        "type": "function_call",
        "status": status,
        "call_id": call_id,
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

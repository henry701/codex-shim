"""MCP tool discovery helpers for the codex-shim.

Injects a virtual ``tool_search_call`` tool definition into BYOK upstream
requests when Codex exposes deferred MCP discovery (``tool_search`` or
``mcp__*`` server stubs).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

MCP_TOOL_SEARCH_NAME = "tool_search_call"

MCP_TOOL_SEARCH_DEFINITION: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": MCP_TOOL_SEARCH_NAME,
        "description": (
            "Look up MCP tool names before calling them. Pass `query` as a short "
            "server or tool token (e.g. 'exa' or 'web_search_exa') to search Codex's "
            "local deferred-tool index. Returns JSON with full mcp__<server>__<tool> "
            "names to call on the next turn — call that tool directly; do not repeat "
            "tool_search_call. Do NOT call bare server stubs like mcp__exa as a tool "
            "— that fails with 'unsupported call'. Avoid mcp__-prefixed search strings."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Short search token, e.g. 'exa' to list Exa tools or "
                        "'web_search_exa' for one tool's description."
                    ),
                }
            },
            "required": ["query"],
        },
    },
}

_FALLBACK_MCP_URLS: dict[str, str] = {
    "mcp__exa": "https://mcp.exa.ai/mcp?tools=web_search_exa",
}

_CONFIG_CACHE: dict[str, Any] | None = None


def responses_tools_need_tool_search(tools: Any) -> bool:
    if not isinstance(tools, list):
        return False
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        tool_type = str(tool.get("type") or "").strip().lower()
        if tool_type == "tool_search":
            return True
        fn = tool.get("function")
        name = tool.get("name") if isinstance(tool.get("name"), str) else None
        if not name and isinstance(fn, dict):
            name = fn.get("name")
        if isinstance(name, str) and name.startswith("mcp__"):
            return True
    return False


def _config_path() -> Path:
    return Path(os.environ.get("CODEX_CONFIG_PATH") or Path.home() / ".codex" / "config.toml")


def _read_codex_mcp_servers() -> dict[str, str]:
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None:
        return _CONFIG_CACHE

    path = _config_path()
    if not path.exists():
        _CONFIG_CACHE = {}
        return _CONFIG_CACHE

    try:
        text = path.read_text()
    except OSError:
        _CONFIG_CACHE = {}
        return _CONFIG_CACHE

    section = _extract_mcp_servers_section(text)
    result: dict[str, str] = {}
    for name, info in section.items():
        if isinstance(info, dict):
            url = info.get("url")
            if isinstance(url, str) and url:
                result[f"mcp__{name}"] = url
    _CONFIG_CACHE = result
    return result


def _extract_mcp_servers_section(text: str) -> dict[str, Any]:
    # Hand-rolled TOML-ish parse to avoid the tomllib 3.11+ dependency.
    result: dict[str, Any] = {}
    current_name: str | None = None
    current: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            if current_name is not None:
                result[current_name] = current
            header = line[1:-1].strip()
            prefix = "mcp_servers."
            if header.startswith(prefix) and "." not in header[len(prefix):]:
                current_name = header[len(prefix):].strip()
                current = {}
            else:
                current_name = None
                current = {}
            continue
        if current_name is None or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            value = value[1:-1]
        current[key] = value
    if current_name is not None:
        result[current_name] = current
    return result


def invalidate_config_cache() -> None:
    global _CONFIG_CACHE
    _CONFIG_CACHE = None


def resolve_mcp_url(server_name: str) -> str | None:
    if not isinstance(server_name, str) or not server_name.startswith("mcp__"):
        return None
    urls = _read_codex_mcp_servers()
    if server_name in urls:
        return urls[server_name]
    return _FALLBACK_MCP_URLS.get(server_name)


def known_mcp_servers() -> list[str]:
    urls = _read_codex_mcp_servers()
    return list({*urls.keys(), *_FALLBACK_MCP_URLS.keys()})


def is_mcp_tool_call(name: str) -> str | None:
    if not isinstance(name, str) or not name.startswith("mcp__"):
        return None
    body = name[len("mcp__"):]
    if "__" not in body:
        return None
    server, _, tool = body.partition("__")
    if not server or not tool:
        return None
    return f"mcp__{server}"


def parse_mcp_function_name(name: str) -> tuple[str, str] | None:
    """Split ``mcp__<server>__<tool>`` into Codex ``(server, tool)`` names."""
    server_key = is_mcp_tool_call(name)
    if not server_key:
        return None
    tool = name[len(server_key) + 2 :]
    if not tool:
        return None
    return server_key[len("mcp__") :], tool


def is_tool_search_call(name: str) -> bool:
    return name == MCP_TOOL_SEARCH_NAME


_UPSTREAM_TOOL_ALIASES: dict[str, str] = {
    "web_search_exa": "mcp__exa__web_search_exa",
    "web_search": "mcp__exa__web_search_exa",
}


def normalize_upstream_tool_name(name: str) -> str:
    stripped = name.strip()
    if not stripped:
        return stripped
    return _UPSTREAM_TOOL_ALIASES.get(stripped, stripped)


def format_tool_search_result(query: str, tools: list[dict[str, Any]]) -> str:
    cleaned: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        name = tool.get("name")
        if not isinstance(name, str) or not name:
            continue
        if not name.startswith(query + "__"):
            name = f"{query}__{name}"
        description = tool.get("description", "")
        if not isinstance(description, str):
            description = ""
        cleaned.append(
            {
                "name": name,
                "description": description.strip()[:300],
            }
        )
    return json.dumps(
        {
            "server": query,
            "tool_count": len(cleaned),
            "tools": cleaned,
            "usage": (
                f"Call these tools as `{query}__<tool>` (e.g. `{query}__read_url`). "
                "Do NOT call the bare server name."
            ),
        },
        indent=2,
    )


def format_tool_search_error(query: str, message: str) -> str:
    return json.dumps({"server": query, "error": message}, indent=2)


async def augment_response_with_tool_search(response: dict[str, Any]) -> dict[str, Any]:
    """Rewrite upstream tool calls into Codex-native Responses item shapes."""
    from . import tool_translate

    output = response.get("output")
    if not isinstance(output, list) or not output:
        return response

    rewritten: list[dict[str, Any]] = []
    changed = False
    for item in output:
        if not isinstance(item, dict):
            rewritten.append(item)
            continue
        if item.get("type") != "function_call":
            rewritten.append(item)
            continue
        name = item.get("name") or ""
        call_id = str(item.get("call_id") or item.get("id") or "call_0")
        raw_args = item.get("arguments", "{}")
        mcp_item = tool_translate.mcp_function_call_from_name(
            call_id,
            name,
            raw_args if isinstance(raw_args, str) else json.dumps(raw_args),
            "completed",
        )
        if mcp_item is not None:
            rewritten.append(mcp_item)
            changed = True
            continue
        if is_tool_search_call(name):
            rewritten.append(
                tool_translate.tool_search_call_from_raw(
                    call_id,
                    raw_args if isinstance(raw_args, str) else json.dumps(raw_args),
                    "completed",
                )
            )
            changed = True
            continue
        rewritten.append(item)
    if changed:
        response["output"] = rewritten
    return response


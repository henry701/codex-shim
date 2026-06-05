"""MCP server proxying for the codex-shim.

Implements a virtual ``tool_search_call`` tool that the shim injects into the
tool list. When the upstream model invokes it, the shim resolves the bare
MCP server name (e.g. ``mcp__jina``) to a transport URL by reading
``~/.codex/config.toml`` and dispatches a JSON-RPC ``tools/list`` call.
The discovered ``mcp__<server>__<tool>`` names are returned to the model so
the next turn can call them directly. Connections are made only on demand.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import aiohttp


MCP_TOOL_SEARCH_NAME = "tool_search_call"

MCP_TOOL_SEARCH_DEFINITION: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": MCP_TOOL_SEARCH_NAME,
        "description": (
            "Look up MCP tool names before calling them. Pass `query` as either "
            "a bare MCP server name (e.g. 'mcp__jina') to list every tool that "
            "server exposes, or a full tool name (e.g. 'mcp__jina__read_url') "
            "to fetch the description of one specific tool. Returns a JSON "
            "object with the mcp__<server>__<tool> names you should then call "
            "directly on the next turn. Do NOT call bare server names like "
            "'mcp__jina' as a tool — that fails with 'unsupported call'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Either a bare MCP server name (e.g. 'mcp__jina') to list its tools, or a full tool name (e.g. 'mcp__jina__read_url') to fetch one tool's description.",
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

_DISCOVERY_CACHE: dict[str, tuple[float, list[dict[str, Any]]]] = {}
_DISCOVERY_TTL_SECONDS = 300.0
_DISCOVERY_ERROR_TTL_SECONDS = 30.0


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


def _iter_server_urls() -> list[tuple[str, str]]:
    result: list[tuple[str, str]] = []
    seen: set[str] = set()
    for name, url in _read_codex_mcp_servers().items():
        if name in seen or not isinstance(url, str) or not url:
            continue
        seen.add(name)
        result.append((name, url))
    for name, url in _FALLBACK_MCP_URLS.items():
        if name in seen:
            continue
        seen.add(name)
        result.append((name, url))
    return result


def invalidate_discovery_cache() -> None:
    _DISCOVERY_CACHE.clear()


async def pre_discover_mcp_tools(force: bool = False) -> list[dict[str, Any]]:
    import time

    now = time.time()
    if not force:
        cached = _DISCOVERY_CACHE.get("__tools__")
        if cached is not None:
            cached_at, cached_tools = cached
            if cached_tools and now - cached_at < _DISCOVERY_TTL_SECONDS:
                return cached_tools
            if not cached_tools and now - cached_at < _DISCOVERY_ERROR_TTL_SECONDS:
                return []

    all_tools: list[dict[str, Any]] = []
    for server_name, url in _iter_server_urls():
        try:
            tools = await call_mcp_tools_list(url)
        except Exception:
            continue
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            name = tool.get("name", "")
            if not isinstance(name, str) or not name:
                continue
            if not name.startswith(f"{server_name}__"):
                name = f"{server_name}__{name}"
            description = tool.get("description", "")
            if not isinstance(description, str):
                description = ""
            parameters = tool.get("inputSchema")
            if not isinstance(parameters, dict):
                parameters = {"type": "object", "properties": {}}
            all_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": description.strip()[:500],
                        "parameters": parameters,
                    },
                }
            )

    _DISCOVERY_CACHE["__tools__"] = (now, all_tools)
    return all_tools


async def call_mcp_tools_list(url: str) -> list[dict[str, Any]]:
    body = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/list",
        "params": {},
    }
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
            async with session.post(url, json=body, headers=headers) as resp:
                if resp.status >= 400:
                    return []
                text = await resp.text()
    except (aiohttp.ClientError, TimeoutError, OSError):
        return []

    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            continue
        tools = _extract_tools_from_jsonrpc(data)
        if tools is not None:
            return tools
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []
    return _extract_tools_from_jsonrpc(data) or []


async def call_mcp_tool(url: str, tool_name: str, arguments: dict[str, Any]) -> str:
    body = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {
            "name": tool_name,
            "arguments": arguments,
        },
    }
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
            async with session.post(url, json=body, headers=headers) as resp:
                if resp.status >= 400:
                    return json.dumps({"error": f"MCP server returned HTTP {resp.status}"})
                text = await resp.text()
    except (aiohttp.ClientError, TimeoutError, OSError) as exc:
        return json.dumps({"error": f"MCP call failed: {exc}"})

    payload_data: Any = None
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        chunk = line[len("data:"):].strip()
        if not chunk or chunk == "[DONE]":
            continue
        try:
            payload_data = json.loads(chunk)
        except json.JSONDecodeError:
            continue
        if isinstance(payload_data, dict):
            break
    if payload_data is None:
        try:
            payload_data = json.loads(text)
        except json.JSONDecodeError:
            return json.dumps({"error": "MCP server returned non-JSON response", "raw": text[:500]})

    return _format_mcp_tool_result(payload_data)


def _format_mcp_tool_result(payload: Any) -> str:
    if not isinstance(payload, dict):
        return json.dumps({"error": "MCP response was not an object"})
    if "error" in payload:
        err = payload["error"]
        if isinstance(err, dict):
            message = err.get("message") or json.dumps(err)
            code = err.get("code")
            return json.dumps({"error": f"MCP error {code}: {message}"} if code is not None else {"error": message})
        return json.dumps({"error": str(err)})
    result = payload.get("result")
    if not isinstance(result, dict):
        return json.dumps(result)
    content = result.get("content")
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                else:
                    parts.append(json.dumps(item))
        if parts:
            return "\n".join(parts)
    if "text" in result and isinstance(result["text"], str):
        return result["text"]
    return json.dumps(result)


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


def _extract_tools_from_jsonrpc(data: Any) -> list[dict[str, Any]] | None:
    if not isinstance(data, dict):
        return None
    result = data.get("result")
    if not isinstance(result, dict):
        return None
    tools = result.get("tools")
    if isinstance(tools, list):
        return [t for t in tools if isinstance(t, dict)]
    return None


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


async def execute_tool_search(query: str) -> str:
    raw = query.strip()
    if raw.startswith("mcp__"):
        body = raw[len("mcp__"):]
        if "__" in body:
            server, _, tool = body.partition("__")
            server_name = f"mcp__{server}"
            tool_name = tool.strip()
        else:
            server_name = raw
            tool_name = ""
    else:
        server_name = raw
        tool_name = ""
    url = resolve_mcp_url(server_name)
    if not url:
        known = ", ".join(known_mcp_servers()) or "(none configured)"
        return format_tool_search_error(
            raw,
            f"Unknown MCP server '{server_name}'. Known servers: {known}.",
        )
    tools = await call_mcp_tools_list(url)
    if not tools:
        return format_tool_search_error(
            raw,
            f"MCP server '{server_name}' reachable at {url} but returned no tools.",
        )
    if tool_name:
        match = next((t for t in tools if isinstance(t, dict) and t.get("name") == tool_name), None)
        if not match:
            available = ", ".join(
                t.get("name", "") for t in tools if isinstance(t, dict) and t.get("name")
            ) or "(none)"
            return format_tool_search_error(
                raw,
                f"MCP server '{server_name}' does not expose a tool named '{tool_name}'. Available: {available}.",
            )
        return format_tool_search_result(server_name, [match])
    return format_tool_search_result(server_name, tools)


async def augment_response_with_tool_search(response: dict[str, Any]) -> dict[str, Any]:
    """Resolve any `tool_search_call` function calls in a Responses-API payload
    by running the MCP tools/list lookup and appending a paired
    `function_call_output` item. Returns the same dict (mutated) if no
    tool_search_call items are present."""
    output = response.get("output")
    if not isinstance(output, list) or not output:
        return response

    augmented: list[dict[str, Any]] = []
    found = False
    for item in output:
        augmented.append(item)
        if not (isinstance(item, dict) and item.get("type") == "function_call"):
            continue
        if item.get("name") != MCP_TOOL_SEARCH_NAME:
            continue
        found = True
        call_id = item.get("call_id") or item.get("id")
        raw_args = item.get("arguments", "{}")
        if isinstance(raw_args, str):
            try:
                args = json.loads(raw_args) if raw_args.strip() else {}
            except json.JSONDecodeError:
                args = {}
        elif isinstance(raw_args, dict):
            args = raw_args
        else:
            args = {}
        query = args.get("query", "") if isinstance(args, dict) else ""
        if not isinstance(query, str) or not query.strip():
            result = format_tool_search_error("", "tool_search_call requires a non-empty string 'query' argument")
        else:
            result = await execute_tool_search(query)
        augmented.append(
            {
                "type": "function_call_output",
                "call_id": call_id,
                "output": result,
                "status": "completed",
            }
        )
    if found:
        response["output"] = augmented
    return response


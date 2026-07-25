"""Cursor Agent → Codex tool bridge (suffix prompt + loopback shell API)."""

from __future__ import annotations

import asyncio
import json
import os
import re
import secrets
import string
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from .settings import DEFAULT_PORT
from .translate import (
    responses_tool_resolve_map,
    responses_tool_type_map,
    resolve_namespaced_tool_name,
    upstream_chat_tool_name,
)

BRIDGE_PATH = "/_cursor_bridge/v1/invoke"
BRIDGE_SUFFIX_TAG = "[CODEX_SHIM_CURSOR_BRIDGE v1]"
BRIDGE_ENV = "CODEX_SHIM_CURSOR_BRIDGE"
BRIDGE_TTL_ENV = "CODEX_SHIM_CURSOR_BRIDGE_TTL_S"
BRIDGE_TOOL_LIST_CAP_ENV = "CODEX_SHIM_CURSOR_BRIDGE_TOOL_LIST_CAP"

DEFAULT_BRIDGE_TTL_S = 1800.0
DEFAULT_SUFFIX_TOOL_LIST_CAP = 40
_BRIDGE_ID_ALPHABET = string.ascii_letters + string.digits
_BRIDGE_SHELL_MARKER = re.compile(r"/_cursor_bridge/v1/invoke", re.IGNORECASE)
_BRIDGE_TOOL_JSON_RE = re.compile(r'"tool"\s*:\s*"([^"\\]+)"')

# Cursor already has shell/file/search/MCP. Bridge only Codex-native control-plane tools
# (goals, collaboration/sub-agents, plan/UI helpers, and future peers).
_BRIDGE_DENIED_EXACT = frozenset(
    {
        "exec_command",
        "write_stdin",
        "apply_patch",
        "local_shell",
        "shell",
        "exec",
        "wait",
        "web_search",
        "web_search_preview",
        "image_generation",
        "computer_use",
        "computer_use_preview",
        "list_mcp_resources",
        "list_mcp_resource_templates",
        "read_mcp_resource",
        "tool_search",
        "tool_search_call",
    }
)
_BRIDGE_DENIED_PREFIXES = ("mcp__", "mcp_")
_BRIDGE_DENIED_TYPES = frozenset(
    {
        "apply_patch",
        "local_shell",
        "shell",
        "web_search",
        "web_search_preview",
        "image_generation",
        "computer_use",
        "computer_use_preview",
    }
)
_DESC_LIMIT = 420
_PARAM_DESC_LIMIT = 160


class BridgeError(Exception):
    """Base bridge error."""


class BridgeToolNotAllowedError(BridgeError):
    """Requested tool is denied or not in this turn's bridged tool set."""


class BridgeNotAttachedError(BridgeError):
    """No stream emitter or collector attached to the session."""


@dataclass(frozen=True)
class BridgeToolSpec:
    chat_name: str
    emit_name: str
    namespace: str | None
    description: str
    parameters: dict[str, Any]

    @property
    def display_name(self) -> str:
        if self.namespace:
            return f"{self.namespace}.{self.emit_name}"
        return self.emit_name


def cursor_bridge_enabled() -> bool:
    return os.environ.get(BRIDGE_ENV, "1").strip().lower() not in {"0", "false", "no", "off"}


def bridge_ttl_seconds() -> float:
    raw = os.environ.get(BRIDGE_TTL_ENV, "").strip()
    if not raw:
        return DEFAULT_BRIDGE_TTL_S
    try:
        return max(1.0, float(raw))
    except ValueError:
        return DEFAULT_BRIDGE_TTL_S


def bridge_tool_list_cap() -> int:
    raw = os.environ.get(BRIDGE_TOOL_LIST_CAP_ENV, "").strip()
    if not raw:
        return DEFAULT_SUFFIX_TOOL_LIST_CAP
    try:
        return max(1, int(raw))
    except ValueError:
        return DEFAULT_SUFFIX_TOOL_LIST_CAP


def generate_bridge_id(length: int = 16) -> str:
    return "".join(secrets.choice(_BRIDGE_ID_ALPHABET) for _ in range(length))


def is_bridge_denied_tool(
    *,
    chat_name: str,
    tool_type: str = "",
    namespace: str | None = None,
) -> bool:
    """True for Cursor-overlapping / MCP / hosted tools that must not go through the bridge."""
    clean = str(chat_name or "").strip().lower()
    ns = str(namespace or "").strip().lower()
    typ = str(tool_type or "").strip().lower()
    if not clean:
        return True
    if clean in _BRIDGE_DENIED_EXACT:
        return True
    if typ in _BRIDGE_DENIED_TYPES or any(typ.startswith(prefix) for prefix in ("web_search", "computer_use", "mcp")):
        return True
    if ns.startswith("mcp"):
        return True
    if any(clean.startswith(prefix) for prefix in _BRIDGE_DENIED_PREFIXES):
        return True
    return False


def _truncate(text: str, limit: int) -> str:
    value = " ".join(str(text or "").split())
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 1)].rstrip() + "…"


def _compact_parameters(parameters: Any) -> dict[str, Any]:
    if not isinstance(parameters, dict):
        return {"type": "object", "properties": {}}
    props_in = parameters.get("properties")
    props_out: dict[str, Any] = {}
    if isinstance(props_in, dict):
        for key, schema in props_in.items():
            if not isinstance(schema, dict):
                props_out[str(key)] = {"type": "string"}
                continue
            entry: dict[str, Any] = {}
            if schema.get("type") is not None:
                entry["type"] = schema.get("type")
            desc = schema.get("description")
            if desc:
                entry["description"] = _truncate(str(desc), _PARAM_DESC_LIMIT)
            enum = schema.get("enum")
            if isinstance(enum, list) and enum:
                entry["enum"] = enum[:12]
            props_out[str(key)] = entry or {"type": "string"}
    out: dict[str, Any] = {"type": "object", "properties": props_out}
    required = parameters.get("required")
    if isinstance(required, list) and required:
        out["required"] = [str(item) for item in required]
    return out


def _tool_sort_key(spec: BridgeToolSpec) -> tuple[int, str]:
    ns = (spec.namespace or "").lower()
    name = spec.emit_name.lower()
    chat = spec.chat_name.lower()
    if ns == "collaboration" or chat.startswith("collaboration_"):
        return (0, name)
    if ns == "goals" or name in {"create_goal", "update_goal", "get_goal"} or chat.startswith("goals_"):
        return (1, name)
    if name in {"update_plan", "request_user_input", "request_plugin_install"}:
        return (2, name)
    return (3, name)


def _iter_bridge_tool_candidates(tools: Any) -> list[BridgeToolSpec]:
    if not isinstance(tools, list):
        return []
    out: list[BridgeToolSpec] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        if tool.get("type") == "namespace":
            namespace = str(tool.get("name") or "").strip()
            ns_desc = str(tool.get("description") or "").strip()
            for sub_tool in tool.get("tools") or []:
                if not isinstance(sub_tool, dict) or sub_tool.get("type") != "function":
                    continue
                emit_name = str(sub_tool.get("name") or "").strip()
                if not namespace or not emit_name:
                    continue
                out.append(
                    BridgeToolSpec(
                        chat_name=upstream_chat_tool_name(namespace, emit_name),
                        emit_name=emit_name,
                        namespace=namespace,
                        description=_truncate(str(sub_tool.get("description") or ns_desc or ""), _DESC_LIMIT),
                        parameters=_compact_parameters(sub_tool.get("parameters")),
                    )
                )
            continue

        fn = tool.get("function") if isinstance(tool.get("function"), dict) else None
        if fn and fn.get("name"):
            emit_name = str(fn.get("name") or "").strip()
            desc = str(fn.get("description") or tool.get("description") or "")
            params = fn.get("parameters") or tool.get("parameters")
        else:
            tool_type = str(tool.get("type") or "").strip().lower()
            emit_name = str(tool.get("name") or "").strip()
            if not emit_name and tool_type not in {"", "function", "custom", "namespace"}:
                emit_name = tool_type
            desc = str(tool.get("description") or "")
            params = tool.get("parameters")
        if not emit_name:
            continue
        chat_name = re.sub(r"[^a-zA-Z0-9_-]+", "_", emit_name.strip())[:64].strip("_") or emit_name
        out.append(
            BridgeToolSpec(
                chat_name=chat_name,
                emit_name=emit_name,
                namespace=None,
                description=_truncate(desc, _DESC_LIMIT),
                parameters=_compact_parameters(params),
            )
        )
    return out


def bridge_tool_specs(body: dict[str, Any]) -> list[BridgeToolSpec]:
    """Codex tools from this turn minus the Cursor-overlap denylist."""
    tools = body.get("tools")
    tool_types = responses_tool_type_map(tools)
    resolve_map = responses_tool_resolve_map(tools)
    specs: list[BridgeToolSpec] = []
    for candidate in _iter_bridge_tool_candidates(tools):
        chat_name = candidate.chat_name
        namespace = candidate.namespace
        emit_name = candidate.emit_name
        if chat_name in resolve_map:
            namespace, emit_name = resolve_map[chat_name]
            chat_name = upstream_chat_tool_name(namespace, emit_name) if namespace else emit_name
        tool_type = str(tool_types.get(chat_name) or tool_types.get(candidate.chat_name) or "function")
        if is_bridge_denied_tool(chat_name=chat_name, tool_type=tool_type, namespace=namespace):
            continue
        specs.append(
            BridgeToolSpec(
                chat_name=chat_name,
                emit_name=emit_name,
                namespace=namespace,
                description=candidate.description,
                parameters=candidate.parameters,
            )
        )
    specs.sort(key=_tool_sort_key)
    seen: set[str] = set()
    unique: list[BridgeToolSpec] = []
    for spec in specs:
        if spec.chat_name in seen:
            continue
        seen.add(spec.chat_name)
        unique.append(spec)
    return unique


def bridge_allowed_tools(body: dict[str, Any]) -> frozenset[str]:
    return frozenset(spec.chat_name for spec in bridge_tool_specs(body))


def shim_port_from_request_host(host_header: str, *, default: int = DEFAULT_PORT) -> int:
    value = (host_header or "").strip()
    if not value:
        return default
    if value.startswith("["):
        end = value.find("]")
        if end != -1 and len(value) > end + 1 and value[end + 1] == ":":
            try:
                return int(value[end + 2 :])
            except ValueError:
                return default
        return default
    if value.count(":") == 1:
        try:
            return int(value.rsplit(":", 1)[1])
        except ValueError:
            return default
    return default


def build_bridge_invoke_url(port: int) -> str:
    return f"http://127.0.0.1:{port}{BRIDGE_PATH}"


def _format_tool_catalog(specs: list[BridgeToolSpec], *, cap: int) -> str:
    if not specs:
        return "(none — no Codex-native tools on this turn)"
    shown = specs[:cap]
    blocks: list[str] = []
    for spec in shown:
        invoke_hint = (
            f'tool="{spec.emit_name}", namespace="{spec.namespace}"'
            if spec.namespace
            else f'tool="{spec.emit_name}"'
        )
        alt = f' (or tool="{spec.chat_name}")' if spec.namespace else ""
        params_json = json.dumps(spec.parameters, separators=(",", ":"), ensure_ascii=False)
        if len(params_json) > 900:
            params_json = params_json[:899] + "…"
        desc = spec.description or "(no description from Codex)"
        blocks.append(
            f"- `{spec.display_name}`\n"
            f"  Invoke: {invoke_hint}{alt}\n"
            f"  Description: {desc}\n"
            f"  Parameters schema: {params_json}"
        )
    extra = ""
    if len(specs) > cap:
        extra = f"\n… ({len(specs) - cap} more bridged tools omitted from prompt; server still accepts them)"
    return "\n".join(blocks) + extra


def build_bridge_suffix(session: CursorBridgeSession, port: int, *, workspace: str | None = None) -> str:
    invoke_url = build_bridge_invoke_url(port)
    curl_body = (
        '{"bridge":"'
        + session.bridge_id
        + '","tool":"TOOL","namespace":"NAMESPACE_OR_EMPTY","arguments":ARGUMENTS}'
    )
    workspace_line = f"Codex workspace path: {workspace}\n\n" if workspace else ""
    catalog = _format_tool_catalog(session.tool_specs, cap=bridge_tool_list_cap())
    return (
        f"{BRIDGE_SUFFIX_TAG}\n"
        f"Bridge session: {session.bridge_id}\n"
        f"Invoke URL: {invoke_url}\n\n"
        f"{workspace_line}"
        "You are running under Cursor/Composer. Use Cursor's own tools for files, shell, search, and MCP.\n"
        "For Codex-native control-plane tools below (goals, sub-agents/collaboration, plan/UI helpers),\n"
        "you MUST call them through this bridge via your Shell tool with EXACTLY this pattern\n"
        "(substitute TOOL, NAMESPACE_OR_EMPTY, and ARGUMENTS only; do not alter bridge/URL/headers):\n"
        "\n"
        f"curl -sS -X POST '{invoke_url}' \\\n"
        "  -H 'Content-Type: application/json' \\\n"
        f"  -d '{curl_body}'\n"
        "\n"
        "For non-namespaced tools set \"namespace\":\"\". ARGUMENTS must be a JSON object.\n"
        "Wait for curl stdout JSON before continuing. Never pretend a bridged tool ran.\n"
        "\n"
        "Bridged Codex tools (denylist excludes shell/file/web/MCP runners):\n"
        f"{catalog}\n"
        "\n"
        "Sub-agent protocol (critical):\n"
        "- spawn_agent: required {\"task_name\":\"...\",\"message\":\"...\"}; keep task_name lowercase/digits/underscores.\n"
        "- send_message / followup_task: required {\"target\":\"...\",\"message\":\"...\"} using ids/paths from spawn_agent or list_agents.\n"
        "- wait_agent: optional {\"timeout_ms\":N}; use after spawn/send when waiting for mailbox updates — do not busy-loop list_agents alone.\n"
        "- list_agents: inspect live agents; interrupt_agent: {\"target\":\"...\"}.\n"
        "- Prefer send_message/followup_task to talk to children; do not use Cursor shell as a substitute for collaboration tools.\n"
        "\n"
        "Goal tools:\n"
        "- create_goal requires {\"objective\":\"...\"} (not name/description).\n"
        "- update_goal requires {\"status\":\"complete\"} or {\"status\":\"blocked\"} (include goal_id when known).\n"
        "- get_goal reads current goal state.\n"
    )


def is_bridge_shell_command(command: str) -> bool:
    return bool(_BRIDGE_SHELL_MARKER.search(command or ""))


def parse_bridge_tool_from_shell(command: str) -> str | None:
    if not is_bridge_shell_command(command):
        return None
    match = _BRIDGE_TOOL_JSON_RE.search(command or "")
    return match.group(1) if match else None


def is_loopback_peer(remote: str | None) -> bool:
    return (remote or "").strip() in {"127.0.0.1", "::1"}


StreamEmitFn = Callable[..., Awaitable[None]]


@dataclass
class CursorBridgeSession:
    bridge_id: str
    allowed_tools: frozenset[str]
    tool_types: dict[str, str]
    tool_resolve: dict[str, tuple[str | None, str]]
    tool_specs: list[BridgeToolSpec] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    ttl_s: float = field(default_factory=bridge_ttl_seconds)
    _invoke_count: int = 0
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _stream_emit: StreamEmitFn | None = field(default=None, repr=False)
    _collector: Any = field(default=None, repr=False)

    @classmethod
    def create(
        cls,
        *,
        allowed_tools: frozenset[str],
        tool_types: dict[str, str],
        tool_resolve: dict[str, tuple[str | None, str]],
        tool_specs: list[BridgeToolSpec] | None = None,
    ) -> CursorBridgeSession:
        specs = list(tool_specs or [])
        if not specs and allowed_tools:
            # Fallback for older call sites: names only.
            specs = [
                BridgeToolSpec(
                    chat_name=name,
                    emit_name=name,
                    namespace=None,
                    description="",
                    parameters={"type": "object", "properties": {}},
                )
                for name in sorted(allowed_tools)
            ]
        return cls(
            bridge_id=generate_bridge_id(),
            allowed_tools=allowed_tools,
            tool_types=tool_types,
            tool_resolve=tool_resolve,
            tool_specs=specs,
        )

    def is_expired(self) -> bool:
        return (time.time() - self.created_at) > self.ttl_s

    def attach_stream(self, emit_fn: StreamEmitFn) -> None:
        self._stream_emit = emit_fn

    def attach_collector(self, collector: Any) -> None:
        self._collector = collector

    def _resolve_tool(self, tool: str, namespace: str | None) -> tuple[str, str | None, str]:
        raw = str(tool or "").strip()
        if not raw:
            raise BridgeToolNotAllowedError("tool name is required")

        candidates: list[str] = []
        if namespace:
            candidates.append(upstream_chat_tool_name(namespace, raw))
        candidates.append(raw)
        if "." in raw:
            parts = raw.split(".", 1)
            if len(parts) == 2 and parts[0] and parts[1]:
                candidates.append(upstream_chat_tool_name(parts[0], parts[1]))

        chat_name = ""
        for candidate in candidates:
            normalized = candidate
            ns, tn = resolve_namespaced_tool_name(normalized, self.tool_resolve)
            if ns:
                normalized = upstream_chat_tool_name(ns, tn)
            if normalized in self.allowed_tools:
                chat_name = normalized
                break

        if not chat_name:
            raise BridgeToolNotAllowedError(
                f"tool not available via Codex bridge (denied or not on this turn): {raw}"
            )

        emit_namespace, emit_name = resolve_namespaced_tool_name(chat_name, self.tool_resolve)
        return chat_name, emit_namespace, emit_name

    async def invoke(
        self,
        *,
        tool: str,
        arguments: Any,
        namespace: str | None = None,
    ) -> dict[str, Any]:
        if not isinstance(arguments, dict):
            raise BridgeError("arguments must be a JSON object")

        async with self._lock:
            if self.is_expired():
                raise BridgeError("bridge session expired")
            chat_name, emit_namespace, emit_name = self._resolve_tool(tool, namespace)
            self._invoke_count += 1
            call_id = f"call_{self.bridge_id}_{self._invoke_count}"

            if self._stream_emit is not None:
                await self._stream_emit(
                    name=emit_name,
                    arguments=arguments,
                    call_id=call_id,
                    namespace=emit_namespace,
                    chat_name=chat_name,
                )
            elif self._collector is not None:
                self._collector.append_function_call(
                    name=emit_name,
                    arguments=arguments,
                    call_id=call_id,
                    namespace=emit_namespace,
                    chat_name=chat_name,
                    tool_types=self.tool_types,
                    tool_resolve=self.tool_resolve,
                )
            else:
                raise BridgeNotAttachedError("bridge session has no active passthrough sink")

            return {
                "ok": True,
                "bridge": self.bridge_id,
                "tool": emit_name if emit_namespace else chat_name,
                "namespace": emit_namespace,
                "codex_call_id": call_id,
            }


class CursorBridgeRegistry:
    def __init__(self) -> None:
        self._sessions: dict[str, CursorBridgeSession] = {}
        self._lock = asyncio.Lock()

    async def register(self, session: CursorBridgeSession) -> str:
        async with self._lock:
            self._sessions[session.bridge_id] = session
            return session.bridge_id

    def get(self, bridge_id: str) -> CursorBridgeSession | None:
        session = self._sessions.get(str(bridge_id or "").strip())
        if session is None:
            return None
        if session.is_expired():
            self._sessions.pop(session.bridge_id, None)
            return None
        return session

    def close(self, bridge_id: str) -> None:
        self._sessions.pop(str(bridge_id or "").strip(), None)


cursor_bridge_registry = CursorBridgeRegistry()

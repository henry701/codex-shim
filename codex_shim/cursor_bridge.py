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


class BridgeError(Exception):
    """Base bridge error."""


class BridgeToolNotAllowedError(BridgeError):
    """Requested tool is not in the session allowlist."""


class BridgeNotAttachedError(BridgeError):
    """No stream emitter or collector attached to the session."""


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


def bridge_allowed_tools(body: dict[str, Any]) -> frozenset[str]:
    return frozenset(responses_tool_type_map(body.get("tools")).keys())


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


def build_bridge_suffix(session: CursorBridgeSession, port: int, *, workspace: str | None = None) -> str:
    invoke_url = build_bridge_invoke_url(port)
    tools = sorted(session.allowed_tools)
    cap = bridge_tool_list_cap()
    if len(tools) <= cap:
        tools_line = ", ".join(tools)
    else:
        shown = ", ".join(tools[:cap])
        tools_line = f"{shown}, … ({len(tools)} tools total; server enforces full allowlist)"
    curl_body = (
        '{"bridge":"'
        + session.bridge_id
        + '","tool":"TOOL","arguments":ARGUMENTS}'
    )
    workspace_line = f"Codex workspace path: {workspace}\n\n" if workspace else ""
    return (
        f"{BRIDGE_SUFFIX_TAG}\n"
        f"Bridge session: {session.bridge_id}\n"
        f"Invoke URL: {invoke_url}\n\n"
        f"{workspace_line}"
        "When you MUST call a Codex tool from this list, use your Shell tool with EXACTLY this pattern\n"
        "(substitute TOOL and ARGUMENTS only; do not alter bridge, URL, or headers):\n"
        "\n"
        f"curl -sS -X POST '{invoke_url}' \\\n"
        "  -H 'Content-Type: application/json' \\\n"
        f"  -d '{curl_body}'\n"
        "\n"
        f"Allowed Codex tools: {tools_line}\n"
        "\n"
        "Rules:\n"
        "- Use a real Shell tool call; never pretend the tool ran.\n"
        "- ARGUMENTS must be valid JSON (object).\n"
        "- Wait for curl stdout JSON before continuing.\n"
        "- Do not run this curl for normal file/shell work; only for listed Codex tools.\n"
        "- Goal tools: create_goal requires {\"objective\":\"...\"}; update_goal requires {\"status\":\"complete\"} or {\"status\":\"blocked\"}.\n"
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
    ) -> CursorBridgeSession:
        return cls(
            bridge_id=generate_bridge_id(),
            allowed_tools=allowed_tools,
            tool_types=tool_types,
            tool_resolve=tool_resolve,
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
            raise BridgeToolNotAllowedError(f"tool not allowed: {raw}")

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

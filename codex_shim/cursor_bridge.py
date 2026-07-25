"""Cursor Agent → Codex tool bridge (suffix prompt + loopback shell API)."""

from __future__ import annotations

import asyncio
import json
import os
import re
import secrets
import string
import time
from collections.abc import AsyncIterator
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
BRIDGE_WAIT_PATH = "/_cursor_bridge/v1/wait"
BRIDGE_POLL_PATH = "/_cursor_bridge/v1/poll"
BRIDGE_SUFFIX_TAG = "[CODEX_SHIM_CURSOR_BRIDGE v1]"
BRIDGE_TURN_CLOSED_EVENT = "bridge_turn_closed"
BRIDGE_CALL_EVENT = "bridge_call"
BRIDGE_EMIT_TIMEOUT_S = 30.0
BRIDGE_ENV = "CODEX_SHIM_CURSOR_BRIDGE"
BRIDGE_TTL_ENV = "CODEX_SHIM_CURSOR_BRIDGE_TTL_S"
BRIDGE_TOOL_LIST_CAP_ENV = "CODEX_SHIM_CURSOR_BRIDGE_TOOL_LIST_CAP"
BRIDGE_WAIT_TIMEOUT_ENV = "CODEX_SHIM_CURSOR_BRIDGE_WAIT_TIMEOUT_MS"

DEFAULT_BRIDGE_TTL_S = 1800.0
DEFAULT_SUFFIX_TOOL_LIST_CAP = 40
DEFAULT_BRIDGE_WAIT_TIMEOUT_MS = 180_000
_BRIDGE_ID_ALPHABET = string.ascii_letters + string.digits
_BRIDGE_SHELL_MARKER = re.compile(r"/_cursor_bridge/v1/(invoke|wait|poll)", re.IGNORECASE)
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


class BridgeJobUnknownError(BridgeError):
    """job_id is unknown or already consumed."""


class BridgeJobTimeoutError(BridgeError):
    """Timed out waiting for a Codex tool result."""


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


def bridge_wait_timeout_ms() -> int:
    raw = os.environ.get(BRIDGE_WAIT_TIMEOUT_ENV, "").strip()
    if not raw:
        return DEFAULT_BRIDGE_WAIT_TIMEOUT_MS
    try:
        return max(1, int(raw))
    except ValueError:
        return DEFAULT_BRIDGE_WAIT_TIMEOUT_MS


def generate_bridge_id(length: int = 16) -> str:
    return "".join(secrets.choice(_BRIDGE_ID_ALPHABET) for _ in range(length))


def generate_job_id(length: int = 12) -> str:
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


def build_bridge_wait_url(port: int) -> str:
    return f"http://127.0.0.1:{port}{BRIDGE_WAIT_PATH}"


def build_bridge_poll_url(port: int) -> str:
    return f"http://127.0.0.1:{port}{BRIDGE_POLL_PATH}"


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
    wait_url = build_bridge_wait_url(port)
    poll_url = build_bridge_poll_url(port)
    curl_body = (
        '{"bridge":"'
        + session.bridge_id
        + '","tool":"TOOL","namespace":"NAMESPACE_OR_EMPTY","arguments":ARGUMENTS}'
    )
    wait_body = '{"bridge":"' + session.bridge_id + '","job_id":"JOB_ID","timeout_ms":TIMEOUT_MS}'
    poll_body = '{"bridge":"' + session.bridge_id + '","timeout_ms":TIMEOUT_MS}'
    workspace_line = f"Codex workspace path: {workspace}\n\n" if workspace else ""
    catalog = _format_tool_catalog(session.tool_specs, cap=bridge_tool_list_cap())
    default_timeout = bridge_wait_timeout_ms()
    return (
        f"{BRIDGE_SUFFIX_TAG}\n"
        f"Bridge session: {session.bridge_id}\n"
        f"Invoke URL: {invoke_url}\n"
        f"Wait URL: {wait_url}\n"
        f"Poll URL: {poll_url}\n\n"
        f"{workspace_line}"
        "You are running under Cursor/Composer. Use Cursor's own tools for files, shell, search, and MCP.\n"
        "For Codex-native control-plane tools below (goals, sub-agents/collaboration, plan/UI helpers),\n"
        "you MUST use this async bridge via your Shell tool.\n"
        "\n"
        "Protocol (mandatory — do not skip wait/poll):\n"
        "1) Invoke (returns immediately with job_id; does NOT include Codex tool output):\n"
        f"curl -sS -X POST '{invoke_url}' \\\n"
        "  -H 'Content-Type: application/json' \\\n"
        f"  -d '{curl_body}'\n"
        "2) Wait for that job's Codex result (blocks until ready; substitute JOB_ID from step 1):\n"
        f"curl -sS -X POST '{wait_url}' \\\n"
        "  -H 'Content-Type: application/json' \\\n"
        f"  -d '{wait_body}'\n"
        f"   Default TIMEOUT_MS={default_timeout}. Use the returned JSON `output` (success or error text).\n"
        "3) Pull channel (optional): poll returns any ready results and removes them so they are not duplicated:\n"
        f"curl -sS -X POST '{poll_url}' \\\n"
        "  -H 'Content-Type: application/json' \\\n"
        f"  -d '{poll_body}'\n"
        "\n"
        "Rules:\n"
        "- For non-namespaced tools set \"namespace\":\"\". ARGUMENTS must be a JSON object.\n"
        "- Never pretend a bridged tool ran. Never re-invoke the same logical action until wait/poll returned its result.\n"
        "- After the first wait/poll, this Codex stream is completed so tools can run; batch needed invokes first,\n"
        "  then wait/poll. Further invokes on the same bridge session will fail until the next Codex turn.\n"
        "- Do not busy-loop invoke/list_agents. Wait or poll instead.\n"
        "- If a call returns {\"error\":\"unknown_bridge\"}, the session is gone (shim restart or ended turn):\n"
        "  do not retry that bridge id, and do not mark the goal blocked — use the bridge id from the\n"
        "  current turn's instructions, or answer as text if none is present.\n"
        "\n"
        "Bridged Codex tools (denylist excludes shell/file/web/MCP runners):\n"
        f"{catalog}\n"
        "\n"
        "Sub-agent protocol (critical):\n"
        "- spawn_agent: required {\"task_name\":\"...\",\"message\":\"...\"}; keep task_name lowercase/digits/underscores.\n"
        "- send_message / followup_task: required {\"target\":\"...\",\"message\":\"...\"} using ids/paths from spawn_agent or list_agents.\n"
        "- wait_agent: optional {\"timeout_ms\":N}; after invoke, use bridge wait/poll for the Codex result — do not spam list_agents.\n"
        "- list_agents: inspect live agents; interrupt_agent: {\"target\":\"...\"}.\n"
        "- Prefer send_message/followup_task to talk to children; do not use Cursor shell as a substitute for collaboration tools.\n"
        "\n"
        "Goal tools:\n"
        "- create_goal requires {\"objective\":\"...\"} (not name/description).\n"
        "- update_goal requires {\"status\":\"complete\"} or {\"status\":\"blocked\"} (include goal_id when known).\n"
        "- get_goal reads current goal state.\n"
    )


def bridge_unknown_session_payload(bridge_id: str, action: str) -> dict[str, Any]:
    """Actionable body for a stale bridge id.

    Bridge state is per-process and per-turn, so a shim restart or a finished turn
    invalidates old ids. A bare 404 makes agents retry in a reconnect storm and then
    mark the goal blocked, so spell out the recovery path instead.
    """
    return {
        "ok": False,
        "error": "unknown_bridge",
        "action": action,
        "bridge": str(bridge_id or ""),
        "retryable": False,
        "hint": (
            "This bridge session no longer exists (the shim restarted or that Codex turn ended). "
            "Do not retry this bridge id. Use the bridge id printed in the current turn's "
            f"{BRIDGE_SUFFIX_TAG} block; if there is none, finish with a normal text answer instead "
            "of reporting the goal as blocked."
        ),
    }


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
TurnCompleteFn = Callable[[], Awaitable[None]]


@dataclass
class BridgeJob:
    job_id: str
    call_id: str
    tool: str
    namespace: str | None
    created_at: float = field(default_factory=time.time)
    status: str = "pending"  # pending | ready | consumed
    output: Any = None
    ready: asyncio.Event = field(default_factory=asyncio.Event)

    def snapshot(self, *, consumed: bool = False) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "codex_call_id": self.call_id,
            "tool": self.tool,
            "namespace": self.namespace,
            "status": "consumed" if consumed else self.status,
            "output": self.output,
        }


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
    _turn_complete: TurnCompleteFn | None = field(default=None, repr=False)
    _jobs_by_id: dict[str, BridgeJob] = field(default_factory=dict, repr=False)
    _jobs_by_call_id: dict[str, BridgeJob] = field(default_factory=dict, repr=False)
    _turn_complete_requested: bool = False
    _turn_closed: bool = False
    _waiter_count: int = 0
    _passthrough_finished: asyncio.Event = field(default_factory=asyncio.Event)
    _events: asyncio.Queue[dict[str, Any] | None] = field(
        default_factory=asyncio.Queue, repr=False
    )
    _agent_task: asyncio.Task[None] | None = field(default=None, repr=False)
    _agent_finished: bool = False
    _post_terminal_text: list[str] = field(default_factory=list, repr=False)
    session_key: str = ""
    _handoff_disconnect_expected: bool = False

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

    def attach_turn_complete(self, complete_fn: TurnCompleteFn) -> None:
        self._turn_complete = complete_fn

    def detach_passthrough(self) -> None:
        self._stream_emit = None
        self._collector = None
        self._turn_complete = None
        self._turn_complete_requested = False

    def mark_turn_closed(self) -> None:
        self._turn_closed = True

    def mark_handoff_disconnect_expected(self) -> None:
        """The next client disconnect is from our early-complete, not user cancel."""
        self._handoff_disconnect_expected = True

    def consume_handoff_disconnect_expected(self) -> bool:
        expected = self._handoff_disconnect_expected
        self._handoff_disconnect_expected = False
        return expected

    def cancel_agent_on_disconnect(self, reason: str) -> None:
        """Kill the agent unless the disconnect was caused by our handoff finish."""
        if self.consume_handoff_disconnect_expected():
            return
        if not self.agent_alive():
            return
        print(f"[cursor-bridge] cancel bridge={self.bridge_id} ({reason})", flush=True)
        self.cancel_agent()

    @property
    def turn_closed(self) -> bool:
        return self._turn_closed

    def reopen_turn(self) -> None:
        """Let the same agent emit more Codex tool calls on the next turn.

        Early-complete closes the turn so Codex can run what was already emitted; the
        agent itself is untouched and will keep invoking once its results arrive.
        """
        self._turn_closed = False
        self._turn_complete_requested = False
        self._handoff_disconnect_expected = False

    def start_agent(self, events: AsyncIterator[dict[str, Any]]) -> None:
        """Own the cursor-agent for the life of the session, not of one HTTP turn.

        Codex can only run tool calls once the response stream ends, so a bridged turn
        is cut short while the agent is still mid-thought. Draining it here keeps that
        one process alive across turns and buffers whatever it emits until the next
        turn adopts it; respawning instead would restart the agent from a rebuilt
        prompt and lose everything it had already worked out.
        """
        if self._agent_task is not None:
            return

        async def _pump() -> None:
            try:
                async for event in events:
                    await self._events.put(event)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                await self._events.put(
                    {"type": "error", "message": str(exc).strip() or repr(exc)}
                )
            finally:
                self._agent_finished = True
                await self._events.put(None)

        self._agent_task = asyncio.create_task(_pump())

    def agent_alive(self) -> bool:
        """True while the agent can still produce output (running or buffered)."""
        if self._agent_task is None:
            return False
        return not self._agent_finished or not self._events.empty()

    async def next_event(self, *, timeout_s: float) -> dict[str, Any] | None:
        """Next buffered or live agent event; ``None`` once the agent is done.

        Raises ``TimeoutError`` when the agent goes quiet.
        """
        return await asyncio.wait_for(self._events.get(), timeout=max(0.1, timeout_s))

    def cancel_agent(self) -> None:
        task = self._agent_task
        self._agent_task = None
        if task is not None and not task.done():
            task.cancel()

    async def _emit_call_in_stream_order(self, payload: dict[str, Any]) -> None:
        """Queue a tool call behind the agent text that preceded it.

        The agent produces text and tool calls as one sequence. Writing a call straight
        to the response while earlier text is still queued reorders the turn, and the
        stranded text is then cut off when the call closes the turn.
        """
        delivered = asyncio.Event()
        self._events.put_nowait(
            {"type": BRIDGE_CALL_EVENT, "emit": payload, "delivered": delivered}
        )
        try:
            await asyncio.wait_for(delivered.wait(), timeout=BRIDGE_EMIT_TIMEOUT_S)
        except TimeoutError as exc:
            raise BridgeNotAttachedError(
                "no Codex turn is streaming this bridge session right now"
            ) from exc

    async def deliver_queued_call(self, event: dict[str, Any]) -> None:
        """Write a queued tool call into the turn that is currently streaming."""
        emit = event.get("emit") or {}
        delivered = event.get("delivered")
        try:
            if self._stream_emit is not None:
                await self._stream_emit(**emit)
        finally:
            if isinstance(delivered, asyncio.Event):
                delivered.set()

    def has_pending_jobs(self) -> bool:
        return any(job.status in {"pending", "ready"} for job in self._jobs_by_id.values())

    def has_active_waiters(self) -> bool:
        return self._waiter_count > 0

    def mark_passthrough_started(self) -> None:
        self._passthrough_finished = asyncio.Event()
        self._post_terminal_text = []

    def append_post_terminal_text(self, text: str) -> None:
        value = str(text or "")
        if value:
            self._post_terminal_text.append(value)

    def take_post_terminal_text(self) -> str:
        text = "".join(self._post_terminal_text).strip()
        self._post_terminal_text.clear()
        return text

    def mark_passthrough_finished(self) -> None:
        self._passthrough_finished.set()

    async def wait_passthrough_finished(self, *, timeout_s: float) -> bool:
        try:
            await asyncio.wait_for(self._passthrough_finished.wait(), timeout=max(0.1, timeout_s))
            return True
        except TimeoutError:
            return False

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

    async def _request_turn_complete(self) -> None:
        if self._turn_complete_requested:
            return
        self._turn_complete_requested = True
        if self._turn_complete is not None:
            try:
                await self._turn_complete()
            except Exception as exc:
                print(f"[cursor-bridge] turn-complete failed bridge={self.bridge_id}: {exc}", flush=True)
        self._turn_closed = True
        if self._agent_task is not None:
            # The turn's consumer is parked on the event queue and the agent cannot emit
            # again until Codex returns these results, so wake it to close the response.
            self._events.put_nowait({"type": BRIDGE_TURN_CLOSED_EVENT})

    def complete_call(self, call_id: str, output: Any) -> bool:
        """Mark a pending job ready. Idempotent: already-ready/consumed → False."""
        job = self._jobs_by_call_id.get(str(call_id or "").strip())
        if job is None or job.status != "pending":
            return False
        job.output = output
        job.status = "ready"
        job.ready.set()
        return True

    def _consume_job(self, job: BridgeJob) -> dict[str, Any]:
        payload = job.snapshot(consumed=True)
        payload["ok"] = True
        job.status = "consumed"
        self._jobs_by_id.pop(job.job_id, None)
        self._jobs_by_call_id.pop(job.call_id, None)
        return payload

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
            if self._turn_closed:
                raise BridgeError(
                    "bridge turn already completed for this session; "
                    "finish wait/poll for in-flight jobs, then invoke on the next Codex turn"
                )
            chat_name, emit_namespace, emit_name = self._resolve_tool(tool, namespace)
            self._invoke_count += 1
            call_id = f"call_{self.bridge_id}_{self._invoke_count}"
            job_id = generate_job_id()
            display_tool = emit_name if emit_namespace else chat_name
            job = BridgeJob(
                job_id=job_id,
                call_id=call_id,
                tool=display_tool,
                namespace=emit_namespace,
            )
            self._jobs_by_id[job_id] = job
            self._jobs_by_call_id[call_id] = job
            cursor_bridge_registry.index_call(call_id, self.bridge_id)
            cursor_bridge_registry.remember_call_tool(call_id, chat_name)

            if self._stream_emit is not None:
                emit_payload = {
                    "name": emit_name,
                    "arguments": arguments,
                    "call_id": call_id,
                    "namespace": emit_namespace,
                    "chat_name": chat_name,
                }
                if self._agent_task is not None:
                    await self._emit_call_in_stream_order(emit_payload)
                else:
                    await self._stream_emit(**emit_payload)
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
                self._jobs_by_id.pop(job_id, None)
                self._jobs_by_call_id.pop(call_id, None)
                cursor_bridge_registry.unindex_call(call_id)
                raise BridgeNotAttachedError("bridge session has no active passthrough sink")

            return {
                "ok": True,
                "status": "accepted",
                "bridge": self.bridge_id,
                "job_id": job_id,
                "tool": display_tool,
                "namespace": emit_namespace,
                "codex_call_id": call_id,
            }

    async def wait_job(self, job_id: str, *, timeout_s: float | None = None) -> dict[str, Any]:
        clean_id = str(job_id or "").strip()
        if not clean_id:
            return {"ok": False, "error": "job_id_required"}
        job = self._jobs_by_id.get(clean_id)
        if job is None:
            return {"ok": False, "error": "unknown_job", "job_id": clean_id}

        if job.status == "pending":
            await self._request_turn_complete()

        self._waiter_count += 1
        try:
            if timeout_s is None:
                timeout_s = bridge_wait_timeout_ms() / 1000.0
            timeout_s = max(0.0, float(timeout_s))

            if job.status == "pending":
                if timeout_s <= 0:
                    return {
                        "ok": False,
                        "error": "timeout",
                        "job_id": clean_id,
                        "codex_call_id": job.call_id,
                        "tool": job.tool,
                        "namespace": job.namespace,
                        "status": job.status,
                    }
                try:
                    await asyncio.wait_for(job.ready.wait(), timeout=timeout_s)
                except TimeoutError:
                    return {
                        "ok": False,
                        "error": "timeout",
                        "job_id": clean_id,
                        "codex_call_id": job.call_id,
                        "tool": job.tool,
                        "namespace": job.namespace,
                        "status": job.status,
                    }

            if job.status == "ready":
                async with self._lock:
                    if job.status == "ready":
                        return self._consume_job(job)
            return {"ok": False, "error": "unknown_job", "job_id": clean_id}
        finally:
            self._waiter_count = max(0, self._waiter_count - 1)

    async def poll_jobs(self, *, timeout_s: float = 0.0) -> dict[str, Any]:
        if self.has_pending_jobs() or any(job.status == "ready" for job in self._jobs_by_id.values()):
            await self._request_turn_complete()
        timeout_s = max(0.0, float(timeout_s))

        self._waiter_count += 1
        try:
            if timeout_s > 0 and not any(job.status == "ready" for job in self._jobs_by_id.values()):
                pending = [job for job in self._jobs_by_id.values() if job.status == "pending"]
                if pending:
                    tasks = [asyncio.create_task(job.ready.wait()) for job in pending]
                    try:
                        done, outstanding = await asyncio.wait(
                            tasks,
                            timeout=timeout_s,
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                        for task in outstanding:
                            task.cancel()
                        if done:
                            await asyncio.gather(*done, return_exceptions=True)
                    except Exception:
                        for task in tasks:
                            task.cancel()

            async with self._lock:
                jobs = [
                    self._consume_job(job)
                    for job in list(self._jobs_by_id.values())
                    if job.status == "ready"
                ]
            pending = sum(1 for job in self._jobs_by_id.values() if job.status == "pending")
            result: dict[str, Any] = {
                "ok": True,
                "bridge": self.bridge_id,
                "jobs": jobs,
                "pending": pending,
            }
            if not jobs and not pending:
                # Agents otherwise re-poll an empty session forever; say so explicitly.
                result["idle"] = True
                result["hint"] = (
                    "No jobs on this bridge session. Stop polling: invoke a tool first, "
                    "or continue your answer if there is nothing left to collect."
                )
            return result
        finally:
            self._waiter_count = max(0, self._waiter_count - 1)


_CALL_TOOL_NAME_MEMO_CAP = 512


class CursorBridgeRegistry:
    def __init__(self) -> None:
        self._sessions: dict[str, CursorBridgeSession] = {}
        self._call_index: dict[str, str] = {}
        self._call_tool_names: dict[str, str] = {}
        self._lock = asyncio.Lock()

    async def register(
        self,
        session: CursorBridgeSession,
        *,
        session_key: str = "",
    ) -> str:
        session.session_key = str(session_key or "").strip()
        async with self._lock:
            self._sessions[session.bridge_id] = session
        self.prune_expired()
        return session.bridge_id

    def cancel_live_agents_for_session(
        self,
        session_key: str,
        *,
        except_bridge_id: str | None = None,
    ) -> int:
        """Kill cursor-agents still running for a Codex session (cancel / new turn).

        After early-complete the agent waits for a tool-output adopt. If the user
        cancels or starts a fresh turn instead, that agent must die immediately —
        otherwise it can keep running shell/file tools in the background.
        """
        key = str(session_key or "").strip()
        if not key:
            return 0
        skip = str(except_bridge_id or "").strip()
        cancelled = 0
        for session in list(self._sessions.values()):
            if session.session_key != key:
                continue
            if skip and session.bridge_id == skip:
                continue
            if not session.agent_alive():
                continue
            print(
                f"[cursor-bridge] cancel bridge={session.bridge_id} "
                f"(superseded by new Codex turn session={key})",
                flush=True,
            )
            session.cancel_agent()
            cancelled += 1
        return cancelled

    def prune_expired(self) -> int:
        """Drop sessions past their TTL.

        A turn that invokes three tools but only waits on one leaves the rest ``ready``
        forever, so ``release_if_idle`` never reclaims the session. Without this sweep
        those sessions accumulate for the life of the process.
        """
        stale = [
            session.bridge_id
            for session in list(self._sessions.values())
            if session.is_expired() and not session.has_active_waiters()
        ]
        for bridge_id in stale:
            print(f"[cursor-bridge] prune-expired bridge={bridge_id}", flush=True)
            self.close(bridge_id)
        return len(stale)

    def get(self, bridge_id: str) -> CursorBridgeSession | None:
        session = self._sessions.get(str(bridge_id or "").strip())
        if session is None:
            return None
        if session.is_expired() and not session.has_pending_jobs():
            self.close(session.bridge_id)
            return None
        return session

    def index_call(self, call_id: str, bridge_id: str) -> None:
        clean = str(call_id or "").strip()
        if clean:
            self._call_index[clean] = bridge_id

    def remember_call_tool(self, call_id: str, chat_name: str) -> None:
        """Retain call_id -> tool name after the job is consumed.

        Bridge results reach Codex detached from the synthetic `function_call`, so the
        input pipeline needs the real name to avoid labelling them `unknown_tool`.
        """
        clean = str(call_id or "").strip()
        if not clean or not chat_name:
            return
        self._call_tool_names[clean] = chat_name
        while len(self._call_tool_names) > _CALL_TOOL_NAME_MEMO_CAP:
            self._call_tool_names.pop(next(iter(self._call_tool_names)))

    def tool_name_for_call(self, call_id: str) -> str | None:
        return self._call_tool_names.get(str(call_id or "").strip())

    def unindex_call(self, call_id: str) -> None:
        self._call_index.pop(str(call_id or "").strip(), None)

    def ingest_function_call_outputs(self, items: Any) -> int:
        """Complete bridge jobs from Codex `function_call_output` input items. Returns count."""
        if not isinstance(items, list):
            return 0
        completed = 0
        for item in items:
            if not isinstance(item, dict):
                continue
            item_type = str(item.get("type") or "").strip()
            if item_type not in {"function_call_output", "custom_tool_call_output"}:
                continue
            call_id = str(item.get("call_id") or "").strip()
            if not call_id:
                continue
            bridge_id = self._call_index.get(call_id)
            session = self.get(bridge_id) if bridge_id else None
            if session is None:
                # Slow path: scan sessions (call index miss after restart-in-process).
                for candidate in list(self._sessions.values()):
                    if call_id in candidate._jobs_by_call_id:
                        session = candidate
                        break
            if session is None:
                continue
            output = item.get("output")
            if session.complete_call(call_id, output):
                completed += 1
                print(
                    f"[cursor-bridge] job-ready bridge={session.bridge_id} call_id={call_id}",
                    flush=True,
                )
        return completed

    def close(self, bridge_id: str) -> None:
        session = self._sessions.pop(str(bridge_id or "").strip(), None)
        if session is None:
            return
        for call_id in list(session._jobs_by_call_id):
            self.unindex_call(call_id)
        session.cancel_agent()
        session.detach_passthrough()

    def release_if_idle(self, bridge_id: str) -> None:
        session = self.get(bridge_id)
        if session is None:
            return
        # A live agent is mid-thought between Codex turns; closing here would strand it
        # and force the next turn to respawn from a rebuilt prompt.
        if session.agent_alive() or session.has_pending_jobs() or session.has_active_waiters():
            session.detach_passthrough()
            return
        self.close(bridge_id)

    def inflight_session_for_tool_outputs(self, items: Any) -> CursorBridgeSession | None:
        """Session whose cursor-agent is still running and owns these tool outputs."""
        if not isinstance(items, list):
            return None
        for item in items:
            if not isinstance(item, dict):
                continue
            if str(item.get("type") or "").strip() not in {
                "function_call_output",
                "custom_tool_call_output",
            }:
                continue
            call_id = str(item.get("call_id") or "").strip()
            if not call_id:
                continue
            bridge_id = self._call_index.get(call_id)
            session = self.get(bridge_id) if bridge_id else None
            if session is None:
                for candidate in list(self._sessions.values()):
                    if call_id in candidate._jobs_by_call_id:
                        session = candidate
                        break
            if session is not None and session.agent_alive():
                return session
        return None

    def cancel_sessions_for_call_ids(self, items: Any) -> int:
        """Kill live agents that own call_ids in ``items`` (steer/interrupt orphans).

        Pure tool-output follow-ups adopt those agents instead; mixed follow-ups
        (interrupted outputs + new user text) must not leave them blocked on wait.
        """
        if not isinstance(items, list):
            return 0
        cancelled = 0
        seen: set[str] = set()
        for item in items:
            if not isinstance(item, dict):
                continue
            if str(item.get("type") or "").strip() not in {
                "function_call_output",
                "custom_tool_call_output",
            }:
                continue
            call_id = str(item.get("call_id") or "").strip()
            if not call_id:
                continue
            bridge_id = self._call_index.get(call_id)
            session = self.get(bridge_id) if bridge_id else None
            if session is None:
                continue
            if session.bridge_id in seen:
                continue
            seen.add(session.bridge_id)
            if session.agent_alive():
                print(
                    f"[cursor-bridge] cancel bridge={session.bridge_id} "
                    f"(orphaned by steer/interrupt call_id={call_id})",
                    flush=True,
                )
                session.cancel_agent()
                cancelled += 1
        return cancelled

    def has_active_waiters(self) -> bool:
        return any(session.has_active_waiters() for session in self._sessions.values())

    def early_closed_sessions(self) -> list[CursorBridgeSession]:
        return [session for session in self._sessions.values() if session._turn_closed]

    async def wait_for_delivery_passthroughs(self, *, timeout_s: float = 180.0) -> str:
        """Wait for waiters to drain and early-completed Cursor turns to finish; return leftover text."""
        deadline = time.time() + max(0.1, timeout_s)
        while self.has_active_waiters() and time.time() < deadline:
            await asyncio.sleep(0.05)
        chunks: list[str] = []
        for session in self.early_closed_sessions():
            remaining = max(0.1, deadline - time.time())
            await session.wait_passthrough_finished(timeout_s=remaining)
            text = session.take_post_terminal_text()
            if text:
                chunks.append(text)
        return "\n".join(chunks).strip()


def input_items_are_only_tool_outputs(items: Any) -> bool:
    """True when Codex follow-up input is tool results (+ optional reasoning), no new user text."""
    if not isinstance(items, list) or not items:
        return False
    saw_tool_output = False
    for item in items:
        if not isinstance(item, dict):
            return False
        item_type = str(item.get("type") or "").strip()
        if item_type in {"function_call_output", "custom_tool_call_output", "tool_search_output"}:
            saw_tool_output = True
            continue
        # Expand/cache may include paired function_call items; still delivery-only.
        if item_type in {"function_call", "custom_tool_call", "item_reference"}:
            continue
        # Desktop often replays reasoning blocks alongside tool outputs.
        if item_type in {"reasoning", "reasoning_text"} or item_type.startswith("reasoning"):
            continue
        return False
    return saw_tool_output


# Must never appear as assistant output — ends Desktop turns and causes goal loops.
BRIDGE_DELIVERY_STUB_MARKER = "[codex-shim] Bridge delivered"


def decide_tool_output_followup(
    *,
    ingested: int,
    delivery_path: bool,
    input_items: Any,
    leftover: str,
) -> str:
    """How to handle a Cursor passthrough follow-up after ingesting bridge job outputs.

    ``delivery_path`` is True when waiters were active *before* ingest and/or an
    early-closed bridge session still has leftover Cursor text to reclaim.

    Returns:
      - ``\"reuse_leftover\"`` — complete with in-flight Cursor leftover text (no new agent)
      - ``\"continue_cursor\"`` — wake waiters (already done) then run normal Cursor passthrough
      - ``\"noop\"`` — no waiter-delivery path; proceed as usual

    Never returns a synthetic stub message — that ends Desktop turns and causes goal loops.
    """
    if ingested <= 0 or not delivery_path:
        return "noop"
    if not input_items_are_only_tool_outputs(input_items):
        return "noop"
    if str(leftover or "").strip():
        return "reuse_leftover"
    return "continue_cursor"


cursor_bridge_registry = CursorBridgeRegistry()

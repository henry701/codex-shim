from __future__ import annotations

import time
from typing import Any

from .sse import write_anthropic_sse, write_bytes, write_sse

_RESPONSES_TERMINAL = frozenset({"response.completed", "response.failed", "response.incomplete"})
RESPONSES_TERMINAL_EVENTS = _RESPONSES_TERMINAL
_ANTHROPIC_TERMINAL = frozenset({"message_stop", "error"})


# Codex (codex-api responses_websocket.rs map_wrapped_websocket_error_event) only
# turns a WS ``error`` frame into an ApiError for this code or a non-2xx top-level
# ``status``/``status_code``; any other ``error`` frame is parsed as a generic
# event and Codex keeps waiting for a terminal that never comes.
WS_CONNECTION_LIMIT_REACHED_CODE = "websocket_connection_limit_reached"
UPSTREAM_ERROR_FALLBACK_CODE = "upstream_error"
_ERROR_LOG_MESSAGE_CHARS = 300


def _error_body(event: dict[str, Any]) -> dict[str, Any]:
    err = event.get("error")
    return err if isinstance(err, dict) else {}


def ws_error_code_message(event: dict[str, Any]) -> tuple[str, str]:
    """``(code, message)`` from a wrapped (``error: {...}``) or flat error frame."""
    err = _error_body(event)
    code = err.get("code") or err.get("type") or event.get("code") or UPSTREAM_ERROR_FALLBACK_CODE
    message = err.get("message") or event.get("message") or event.get("detail") or ""
    return str(code), str(message)


def codex_maps_ws_error(event: dict[str, Any]) -> bool:
    """True when Codex ends the turn on this ``error`` frame by itself."""
    if event.get("type") != "error":
        return False
    if _error_body(event).get("code") == WS_CONNECTION_LIMIT_REACHED_CODE:
        return True
    status = event.get("status", event.get("status_code"))
    return isinstance(status, int) and not isinstance(status, bool) and not 200 <= status < 300


def describe_ws_error(event: dict[str, Any]) -> str:
    """One-line log form of an error frame: type, code, status, truncated message."""
    code, message = ws_error_code_message(event)
    err_type = _error_body(event).get("type")
    status = event.get("status", event.get("status_code"))
    if len(message) > _ERROR_LOG_MESSAGE_CHARS:
        message = message[:_ERROR_LOG_MESSAGE_CHARS] + "..."
    return f"type={err_type!r} code={code!r} status={status!r} message={message!r}"


def _synthetic_responses_object(last: dict[str, Any] | None, model: str, status: str) -> dict[str, Any]:
    base = dict(last) if last else {}
    base.setdefault("id", f"resp_shim_{int(time.time() * 1000)}")
    base.setdefault("object", "response")
    base.setdefault("model", model)
    base.setdefault("output", [])
    base["status"] = status
    return base


class ResponsesEmitter:
    def __init__(self, state: Any):
        self.state = state

    @property
    def already_emitted(self) -> bool:
        return bool(getattr(self.state, "failed", False) or getattr(self.state, "terminal_emitted", False))

    @property
    def terminal_event(self) -> str | None:
        return getattr(self.state, "terminal_event", None)

    def finish_reason(self) -> Any:
        return getattr(self.state, "upstream_finish_reason", None)

    async def complete(self, response: Any, *, upstream_saw_done: bool) -> str:
        if self.already_emitted:
            return self.terminal_event or "NONE"
        await self.state.finish(response, upstream_saw_done=upstream_saw_done)
        return self.terminal_event or "NONE"

    async def fail(self, response: Any, message: str, *, code: str) -> str:
        if self.already_emitted:
            return self.terminal_event or "response.failed"
        await self.state.fail(response, message, code=code)
        return "response.failed"


class RawChatEmitter:
    def __init__(self) -> None:
        self.already_emitted = False
        self.terminal_event: str | None = None

    async def complete(self, response: Any, *, upstream_saw_done: bool) -> str:
        del upstream_saw_done
        if self.already_emitted:
            return self.terminal_event or "done"
        await write_bytes(response, b"data: [DONE]\n\n")
        self.already_emitted = True
        self.terminal_event = "done"
        return "done"

    async def fail(self, response: Any, message: str, *, code: str) -> str:
        if self.already_emitted:
            return self.terminal_event or "error"
        await write_sse(response, {"error": {"code": code, "message": message}})
        await write_bytes(response, b"data: [DONE]\n\n")
        self.already_emitted = True
        self.terminal_event = "error"
        return "error"


class AnthropicMessagesEmitter:
    def __init__(self, state: Any):
        self.state = state

    @property
    def already_emitted(self) -> bool:
        return bool(getattr(self.state, "failed", False) or getattr(self.state, "terminal_emitted", False))

    @property
    def terminal_event(self) -> str | None:
        if getattr(self.state, "failed", False):
            return "error"
        if getattr(self.state, "terminal_emitted", False):
            return "message_stop"
        return None

    async def complete(self, response: Any, *, upstream_saw_done: bool) -> str:
        del upstream_saw_done
        if self.already_emitted:
            return self.terminal_event or "message_stop"
        await self.state.finish(response)
        return "message_stop"

    async def fail(self, response: Any, message: str, *, code: str) -> str:
        if self.already_emitted:
            return self.terminal_event or "error"
        await self.state.fail(response, message, code=code)
        return "error"


class ChatgptRelayEmitter:
    """Passthrough for native Responses SSE. Synthesizes a terminal if upstream omits one."""

    def __init__(self, *, model: str = "chatgpt"):
        self.model = model
        self.saw_terminal = False
        self.last_response: dict[str, Any] | None = None
        self.already_emitted = False
        self.terminal_event: str | None = None
        self._done_written = False

    def observe(self, payload: dict[str, Any]) -> None:
        event_type = payload.get("type")
        if event_type in _RESPONSES_TERMINAL:
            self.saw_terminal = True
            self.terminal_event = str(event_type)
        response = payload.get("response")
        if isinstance(response, dict):
            self.last_response = response

    async def complete(self, response: Any, *, upstream_saw_done: bool) -> str:
        del upstream_saw_done
        if self.saw_terminal:
            await self._write_done(response)
            self.already_emitted = True
            return self.terminal_event or "response.completed"
        event_type = "response.incomplete"
        print(
            f"[stream] {self.model} upstream ended without a terminal event; synthesizing {event_type}",
            flush=True,
        )
        await write_sse(response, {"type": event_type, "response": _synthetic_responses_object(self.last_response, self.model, "incomplete")})
        await self._write_done(response)
        self.already_emitted = True
        self.terminal_event = event_type
        return event_type

    async def fail(self, response: Any, message: str, *, code: str) -> str:
        if self.saw_terminal:
            await self._write_done(response)
            self.already_emitted = True
            return self.terminal_event or "response.failed"
        failed = _synthetic_responses_object(self.last_response, self.model, "failed")
        failed["error"] = {"code": code, "message": message}
        await write_sse(response, {"type": "error", "code": code, "message": message})
        await write_sse(response, {"type": "response.failed", "response": failed})
        await self._write_done(response)
        self.already_emitted = True
        self.saw_terminal = True
        self.terminal_event = "response.failed"
        return "response.failed"

    async def _write_done(self, response: Any) -> None:
        if self._done_written:
            return
        await write_bytes(response, b"data: [DONE]\n\n")
        self._done_written = True


class AnthropicRelayEmitter:
    """Native Anthropic Messages SSE relay. Synthesizes ``message_stop`` if missing."""

    def __init__(self, *, model: str = "anthropic"):
        self.model = model
        self.saw_terminal = False
        self.already_emitted = False
        self.terminal_event: str | None = None

    def observe(self, event: dict[str, Any]) -> None:
        event_type = event.get("type")
        if event_type in _ANTHROPIC_TERMINAL:
            self.saw_terminal = True
            self.terminal_event = str(event_type)

    async def complete(self, response: Any, *, upstream_saw_done: bool) -> str:
        del upstream_saw_done
        if self.saw_terminal:
            self.already_emitted = True
            return self.terminal_event or "message_stop"
        print(
            f"[stream] {self.model} upstream ended without message_stop; synthesizing terminal",
            flush=True,
        )
        await write_anthropic_sse(response, "message_stop", {"type": "message_stop"})
        self.already_emitted = True
        self.terminal_event = "message_stop"
        return "message_stop"

    async def fail(self, response: Any, message: str, *, code: str) -> str:
        if self.already_emitted:
            return self.terminal_event or "error"
        await write_anthropic_sse(
            response,
            "error",
            {"type": "error", "error": {"type": code, "message": message}},
        )
        await write_anthropic_sse(response, "message_stop", {"type": "message_stop"})
        self.already_emitted = True
        self.saw_terminal = True
        self.terminal_event = "error"
        return "error"


class WsRelayEmitter:
    """Synthesize a Responses terminal frame on a client WebSocket.

    An upstream ``error`` frame Codex would not map (see ``codex_maps_ws_error``)
    is not terminal for Codex, so it is held in ``upstream_error`` and ``complete``
    closes the turn with a ``response.failed`` carrying the same code/message.
    """

    def __init__(self, write_event, *, model: str = "chatgpt"):
        self._write_event = write_event
        self.model = model
        self.saw_terminal = False
        self.last_response: dict[str, Any] | None = None
        self.already_emitted = False
        self.terminal_event: str | None = None
        self.last_emitted: dict[str, Any] | None = None
        self.upstream_error: dict[str, Any] | None = None

    def observe(self, payload: dict[str, Any]) -> None:
        event_type = payload.get("type")
        if event_type == "error" and not codex_maps_ws_error(payload):
            self.upstream_error = payload
        elif event_type in _RESPONSES_TERMINAL | {"error"}:
            self.saw_terminal = True
            self.terminal_event = str(event_type)
            self.already_emitted = True
        response = payload.get("response")
        if isinstance(response, dict):
            self.last_response = response

    async def complete(self, response: Any = None, *, upstream_saw_done: bool = False) -> str:
        del response, upstream_saw_done
        if self.saw_terminal:
            return self.terminal_event or "response.completed"
        if self.upstream_error is not None:
            code, message = ws_error_code_message(self.upstream_error)
            return await self._emit_failed(code, message, error_frame=False)
        event = {
            "type": "response.incomplete",
            "response": _synthetic_responses_object(self.last_response, self.model, "incomplete"),
        }
        await self._write_event(event)
        self.last_emitted = event
        self.already_emitted = True
        self.saw_terminal = True
        self.terminal_event = "response.incomplete"
        return "response.incomplete"

    async def fail(self, response: Any, message: str, *, code: str) -> str:
        del response
        if self.already_emitted:
            return self.terminal_event or "error"
        return await self._emit_failed(code, message, error_frame=True)

    async def _emit_failed(self, code: str, message: str, *, error_frame: bool) -> str:
        failed = _synthetic_responses_object(self.last_response, self.model, "failed")
        failed["error"] = {"code": code, "message": message}
        event = {"type": "response.failed", "response": failed}
        if error_frame:
            await self._write_event({"type": "error", "code": code, "message": message})
        await self._write_event(event)
        self.last_emitted = event
        self.already_emitted = True
        self.saw_terminal = True
        self.terminal_event = "response.failed"
        return "response.failed"

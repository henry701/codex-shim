from __future__ import annotations

import time
from typing import Any

from .sse import write_anthropic_sse, write_bytes, write_sse

_RESPONSES_TERMINAL = frozenset({"response.completed", "response.failed", "response.incomplete"})
_ANTHROPIC_TERMINAL = frozenset({"message_stop", "error"})


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
    """Synthesize a Responses terminal frame on a client WebSocket."""

    def __init__(self, write_event, *, model: str = "chatgpt"):
        self._write_event = write_event
        self.model = model
        self.saw_terminal = False
        self.last_response: dict[str, Any] | None = None
        self.already_emitted = False
        self.terminal_event: str | None = None
        self.last_emitted: dict[str, Any] | None = None

    def observe(self, payload: dict[str, Any]) -> None:
        event_type = payload.get("type")
        if event_type in _RESPONSES_TERMINAL | {"error"}:
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
        failed = _synthetic_responses_object(self.last_response, self.model, "failed")
        failed["error"] = {"code": code, "message": message}
        await self._write_event({"type": "error", "code": code, "message": message})
        await self._write_event({"type": "response.failed", "response": failed})
        self.already_emitted = True
        self.terminal_event = "response.failed"
        return "response.failed"

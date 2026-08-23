from __future__ import annotations

from typing import Any

MAX_PREFILL_CONTINUES = 3
CONCLUSIVE_FINISH_REASONS = frozenset({"stop", "tool_calls"})
PREFILL_REJECT_STATUS = frozenset({400, 404, 422})


def is_conclusive_finish(finish_reason: str | None, *, saw_done: bool) -> bool:
    if saw_done:
        return True
    return finish_reason in CONCLUSIVE_FINISH_REASONS


def should_prefill_continue(
    *,
    finish_reason: str | None,
    saw_done: bool,
    continues: int,
    as_responses: bool = True,
) -> bool:
    if not as_responses:
        return False
    if continues >= MAX_PREFILL_CONTINUES:
        return False
    if finish_reason == "length":
        return False
    if is_conclusive_finish(finish_reason, saw_done=saw_done):
        return False
    return True


def assistant_prefill_message(state: Any) -> dict[str, Any] | None:
    content = str(getattr(state, "message_text", "") or "")
    reasoning_parts: list[str] = []
    for block in (getattr(state, "reasoning_blocks", None) or {}).values():
        text = str((block or {}).get("text") or "")
        if text:
            reasoning_parts.append(text)
    reasoning = "".join(reasoning_parts)
    tool_calls: list[dict[str, Any]] = []
    for index, tc in _iter_indexed_tools(state):
        name = str(tc.get("name") or "")
        call_id = str(tc.get("call_id") or tc.get("id") or f"call_{index}")
        arguments = str(tc.get("arguments") or "")
        if not name and not arguments:
            continue
        tool_calls.append(
            {
                "id": call_id,
                "type": "function",
                "index": index,
                "function": {"name": name, "arguments": arguments},
            }
        )
    if not content and not reasoning and not tool_calls:
        return None
    message: dict[str, Any] = {
        "role": "assistant",
        "content": content if content or not tool_calls else "",
    }
    if reasoning:
        message["reasoning_content"] = reasoning
    if tool_calls:
        message["tool_calls"] = tool_calls
        if message["content"] is None:
            message["content"] = ""
    return message


def with_assistant_prefill(
    messages: list[Any],
    prefill: dict[str, Any],
) -> list[Any]:
    out = list(messages)
    if out and isinstance(out[-1], dict) and out[-1].get("role") == "assistant":
        out[-1] = prefill
        return out
    out.append(prefill)
    return out


def skip_replayed_prefix(prefix: str, incoming: str, skipped: int) -> tuple[str, int]:
    if skipped < 0 or not prefix or skipped >= len(prefix):
        return incoming, skipped
    index = skipped
    offset = 0
    while index < len(prefix) and offset < len(incoming) and prefix[index] == incoming[offset]:
        index += 1
        offset += 1
    if offset < len(incoming) and index < len(prefix):
        return incoming, skipped if offset == 0 else -1
    return incoming[offset:], index


class ReplaySkipper:
    def __init__(
        self,
        text_prefix: str = "",
        tool_prefixes: dict[int, str] | None = None,
    ):
        self.text_prefix = text_prefix
        self.text_skipped = 0
        self.tool_prefixes = tool_prefixes or {}
        self.tool_skipped = {index: 0 for index in self.tool_prefixes}

    @classmethod
    def from_state(cls, state: Any) -> ReplaySkipper:
        tools: dict[int, str] = {}
        for index, tc in _iter_indexed_tools(state):
            args = str(tc.get("arguments") or "")
            if args:
                tools[index] = args
        return cls(
            text_prefix=str(getattr(state, "message_text", "") or ""),
            tool_prefixes=tools,
        )

    def filter_chunk(self, event: dict[str, Any]) -> dict[str, Any] | None:
        choice = (event.get("choices") or [{}])[0]
        if not isinstance(choice, dict):
            return event
        delta = choice.get("delta")
        if not isinstance(delta, dict):
            return event
        new_delta = dict(delta)
        content = new_delta.get("content")
        if isinstance(content, str) and content:
            emitted, self.text_skipped = skip_replayed_prefix(
                self.text_prefix, content, self.text_skipped
            )
            if emitted:
                new_delta["content"] = emitted
            else:
                new_delta.pop("content", None)
        tools = new_delta.get("tool_calls")
        if isinstance(tools, list):
            filtered_tools: list[Any] = []
            for call in tools:
                filtered_tools.append(self._filter_tool_call(call))
            new_delta["tool_calls"] = [call for call in filtered_tools if call is not None]
            if not new_delta["tool_calls"]:
                new_delta.pop("tool_calls", None)
        if not new_delta and not choice.get("finish_reason") and not event.get("usage"):
            return None
        new_choice = {**choice, "delta": new_delta}
        choices = list(event.get("choices") or [{}])
        choices[0] = new_choice
        return {**event, "choices": choices}

    def _filter_tool_call(self, call: Any) -> Any:
        if not isinstance(call, dict):
            return call
        try:
            index = int(call.get("index", 0))
        except (TypeError, ValueError):
            return call
        prefix = self.tool_prefixes.get(index, "")
        fn = call.get("function")
        if not isinstance(fn, dict):
            return call
        args = fn.get("arguments")
        if not isinstance(args, str) or not args or not prefix:
            return call
        skipped = self.tool_skipped.get(index, 0)
        emitted, skipped = skip_replayed_prefix(prefix, args, skipped)
        self.tool_skipped[index] = skipped
        new_fn = dict(fn)
        if emitted:
            new_fn["arguments"] = emitted
        else:
            new_fn.pop("arguments", None)
        if not new_fn and not call.get("id") and not call.get("type"):
            return None
        return {**call, "function": new_fn}


def _iter_indexed_tools(state: Any):
    buckets = (
        getattr(state, "tool_calls", None) or {},
        getattr(state, "mcp_tool_calls", None) or {},
        getattr(state, "tool_search_calls", None) or {},
    )
    seen: set[int] = set()
    sequential = 0
    for bucket in buckets:
        for key, tc in bucket.items():
            if not isinstance(tc, dict):
                continue
            try:
                index = int(key)
            except (TypeError, ValueError):
                index = sequential
            while index in seen:
                sequential += 1
                index = sequential
            seen.add(index)
            sequential = max(sequential, index + 1)
            yield index, tc

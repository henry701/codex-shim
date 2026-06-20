from __future__ import annotations

import json

from codex_shim.server import ResponsesStreamState
from codex_shim.translate import chat_completion_to_response


class FakeResponse:
    def __init__(self):
        self.chunks: list[bytes] = []

    async def write(self, data: bytes):
        self.chunks.append(data)


def _sse_events(raw: bytes) -> list[dict]:
    events = []
    for block in raw.decode().split("\n\n"):
        if not block.startswith("data: "):
            continue
        payload = block.removeprefix("data: ")
        if payload == "[DONE]":
            continue
        events.append(json.loads(payload))
    return events


def test_chat_completion_maps_apply_patch_tool_to_custom_tool_call():
    payload = {
        "id": "chatcmpl_apply",
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "id": "call_patch",
                            "type": "function",
                            "function": {
                                "name": "apply_patch",
                                "arguments": "*** Begin Patch\n*** End Patch",
                            },
                        }
                    ]
                }
            }
        ],
    }

    out = chat_completion_to_response(payload, "local")

    assert out["output"][0]["type"] == "custom_tool_call"
    assert out["output"][0]["name"] == "apply_patch"
    assert out["output"][0]["input"] == "*** Begin Patch\n*** End Patch"


def test_chat_completion_maps_web_search_tool_to_web_search_call():
    payload = {
        "id": "chatcmpl_web",
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "id": "call_web",
                            "type": "function",
                            "function": {
                                "name": "web_search",
                                "arguments": '{"query":"codex shim"}',
                            },
                        }
                    ]
                }
            }
        ],
    }

    out = chat_completion_to_response(payload, "local")

    assert out["output"][0]["type"] == "web_search_call"
    assert out["output"][0]["action"] == {"type": "search", "query": "codex shim"}


async def test_streaming_apply_patch_tool_emits_custom_tool_call():
    downstream = FakeResponse()
    state = ResponsesStreamState("local")
    await state.start(downstream)
    await state.write_chat_delta(
        downstream,
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_patch",
                                "function": {
                                    "name": "apply_patch",
                                    "arguments": "*** Begin Patch\n",
                                },
                            }
                        ]
                    }
                }
            ]
        },
    )
    await state.write_chat_delta(
        downstream,
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "*** End Patch"}}]}}]},
    )
    await state.finish(downstream, upstream_saw_done=True)

    events = _sse_events(b"".join(downstream.chunks))
    added = [
        event
        for event in events
        if event.get("type") == "response.output_item.added"
        and (event.get("item") or {}).get("type") == "custom_tool_call"
    ]
    assert len(added) == 1
    completed = [event for event in events if event.get("type") == "response.completed"][-1]
    custom_items = [item for item in completed["response"]["output"] if item.get("type") == "custom_tool_call"]
    assert custom_items[0]["input"] == "*** Begin Patch\n*** End Patch"


async def test_streaming_web_search_tool_emits_web_search_call():
    downstream = FakeResponse()
    state = ResponsesStreamState("local")
    await state.start(downstream)
    await state.write_chat_delta(
        downstream,
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_web",
                                "function": {
                                    "name": "web_search",
                                    "arguments": '{"query":"codex shim"}',
                                },
                            }
                        ]
                    }
                }
            ]
        },
    )
    await state.finish(downstream, upstream_saw_done=True)

    events = _sse_events(b"".join(downstream.chunks))
    added = [
        event
        for event in events
        if event.get("type") == "response.output_item.added"
        and (event.get("item") or {}).get("type") == "web_search_call"
    ]
    assert len(added) == 1
    completed = [event for event in events if event.get("type") == "response.completed"][-1]
    web_items = [item for item in completed["response"]["output"] if item.get("type") == "web_search_call"]
    assert web_items[0]["action"] == {"type": "search", "query": "codex shim"}

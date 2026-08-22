from __future__ import annotations

import json

from codex_shim.server import ResponsesStreamState
from codex_shim.translate import chat_completion_to_response, original_responses_tool_type, unwrap_custom_tool_input


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


def test_apply_patch_stays_custom_when_client_advertises_function_or_custom():
    call = {
        "id": "call_patch",
        "type": "function",
        "function": {
            "name": "apply_patch",
            "arguments": json.dumps({"input": "*** Begin Patch\n*** Add File: snake.html\n+hi\n*** End Patch"}),
        },
    }
    payload = {"id": "chatcmpl_apply", "choices": [{"message": {"tool_calls": [call]}}]}

    for advertised in ("function", "custom", "freeform"):
        out = chat_completion_to_response(
            payload,
            "local",
            tool_types={"apply_patch": advertised},
        )
        item = out["output"][0]
        assert item["type"] == "custom_tool_call", advertised
        assert item["name"] == "apply_patch"
        assert item["input"].startswith("*** Begin Patch")
        assert not item["input"].lstrip().startswith("{")


def test_unwrap_custom_tool_input_strips_json_envelope():
    raw = "*** Begin Patch\n*** End Patch"
    assert unwrap_custom_tool_input(json.dumps({"input": raw})) == raw
    assert unwrap_custom_tool_input(json.dumps({"patch": raw})) == raw
    assert unwrap_custom_tool_input(raw) == raw
    assert original_responses_tool_type("apply_patch", {"apply_patch": "function"}) == "apply_patch"
    assert original_responses_tool_type("apply_patch", {"apply_patch": "custom"}) == "apply_patch"


async def test_streaming_apply_patch_with_function_tool_type_emits_custom_tool_call():
    downstream = FakeResponse()
    state = ResponsesStreamState("local", tool_types={"apply_patch": "function"})
    await state.start(downstream)
    envelope = json.dumps(
        {"input": "*** Begin Patch\n*** Add File: snake.html\n+hi\n*** End Patch"}
    )
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
                                "function": {"name": "apply_patch", "arguments": envelope[:12]},
                            }
                        ]
                    }
                }
            ]
        },
    )
    await state.write_chat_delta(
        downstream,
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": envelope[12:]}}]}}]},
    )
    await state.finish(downstream, upstream_saw_done=True)

    events = _sse_events(b"".join(downstream.chunks))
    added = [
        event
        for event in events
        if event.get("type") == "response.output_item.added"
        and (event.get("item") or {}).get("name") == "apply_patch"
    ]
    assert added, "expected apply_patch output item"
    assert added[0]["item"]["type"] == "custom_tool_call"
    assert not any(event.get("type") == "response.function_call_arguments.delta" for event in events)
    completed = [event for event in events if event.get("type") == "response.completed"][-1]
    custom_items = [item for item in completed["response"]["output"] if item.get("type") == "custom_tool_call"]
    assert custom_items[0]["input"].startswith("*** Begin Patch")
    assert not custom_items[0]["input"].lstrip().startswith("{")


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

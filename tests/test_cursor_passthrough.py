from __future__ import annotations

import json

from codex_shim.cursor_passthrough import (
    CursorResponseCollector,
    CursorStreamParser,
    _parse_cursor_list_models_output,
    build_cursor_prompt,
    cursor_upstream_model,
    format_cursor_tool_completed_markdown,
    format_cursor_tool_started_markdown,
    is_cursor_passthrough_slug,
    iter_cursor_agent_events,
)

def test_parse_cursor_list_models_output():
    models = _parse_cursor_list_models_output(
        "auto - Auto\ncomposer-2.5 - Composer 2.5\ngpt-5.3-codex - Codex 5.3\n"
    )
    assert models["cursor-composer-2-5"].upstream_id == "composer-2.5"
    assert models["cursor-composer-2-5"].display_name == "Cursor - Composer 2.5"
    assert models["cursor-gpt-5-3-codex"].upstream_id == "gpt-5.3-codex"


def test_is_cursor_passthrough_slug(monkeypatch):
    monkeypatch.setattr(
        "codex_shim.cursor_passthrough._load_cursor_catalog_models",
        lambda **_: _parse_cursor_list_models_output("composer-2.5 - Composer 2.5\n"),
    )
    assert is_cursor_passthrough_slug("cursor-composer-2-5")
    assert is_cursor_passthrough_slug("composer-2-5")
    assert is_cursor_passthrough_slug("composer-2.5")
    assert not is_cursor_passthrough_slug("codex-gpt-5-5")
    assert cursor_upstream_model("cursor-composer-2-5") == "composer-2.5"
    assert cursor_upstream_model("composer-2-5") == "composer-2.5"


def test_build_cursor_prompt_from_responses_body():
    body = {
        "model": "cursor-composer-2-5",
        "instructions": "You are Codex.",
        "input": [{"role": "user", "content": "Hello"}],
    }
    prompt = build_cursor_prompt(body)
    assert "You are Codex." in prompt
    assert "Hello" in prompt


def test_cursor_stream_parser_emits_deltas():
    parser = CursorStreamParser()
    line1 = json.dumps(
        {
            "type": "assistant",
            "message": {"role": "assistant", "content": [{"type": "text", "text": "Hel"}]},
            "timestamp_ms": 1,
        }
    )
    line2 = json.dumps(
        {
            "type": "assistant",
            "message": {"role": "assistant", "content": [{"type": "text", "text": "lo"}]},
            "timestamp_ms": 2,
        }
    )
    assert parser.feed_line(line1) == "Hel"
    assert parser.feed_line(line2) == "lo"


def test_cursor_stream_parser_skips_buffered_flush_before_tool():
    parser = CursorStreamParser()
    streaming = json.dumps(
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "Root cause: global OOM"}],
            },
            "timestamp_ms": 1,
        }
    )
    buffered = json.dumps(
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "Root cause: global OOM"}],
            },
            "timestamp_ms": 2,
            "model_call_id": "call-1",
        }
    )
    assert parser.feed_events(streaming) == [
        {"type": "text_delta", "delta": "Root cause: global OOM"}
    ]
    assert parser.feed_events(buffered) == [{"type": "segment_boundary"}]


def test_cursor_stream_parser_skips_final_flush():
    parser = CursorStreamParser()
    streaming = json.dumps(
        {
            "type": "assistant",
            "message": {"role": "assistant", "content": [{"type": "text", "text": "Done"}]},
            "timestamp_ms": 1,
        }
    )
    final_flush = json.dumps(
        {
            "type": "assistant",
            "message": {"role": "assistant", "content": [{"type": "text", "text": "Done"}]},
        }
    )
    parser.feed_events(streaming)
    assert parser.feed_events(final_flush) == [{"type": "segment_boundary"}]


def test_cursor_stream_parser_tool_call_sequence():
    parser = CursorStreamParser()
    started = json.dumps(
        {
            "type": "tool_call",
            "subtype": "started",
            "call_id": "tool-1",
            "tool_call": {"readToolCall": {"args": {"path": "README.md"}}},
        }
    )
    completed = json.dumps(
        {
            "type": "tool_call",
            "subtype": "completed",
            "call_id": "tool-1",
            "tool_call": {
                "readToolCall": {
                    "args": {"path": "README.md"},
                    "result": {"success": {"content": "# Title\n"}},
                }
            },
        }
    )
    start_events = parser.feed_events(started)
    assert start_events[0]["type"] == "segment_boundary"
    assert start_events[1]["type"] == "tool_started"
    assert "README.md" in start_events[1]["markdown"]

    complete_events = parser.feed_events(completed)
    assert complete_events[0]["type"] == "tool_completed"
    assert "# Title" in complete_events[0]["markdown"]


def test_format_cursor_tool_started_markdown():
    markdown = format_cursor_tool_started_markdown(
        {"tool_call": {"shellToolCall": {"args": {"command": "sysctl vm.swappiness"}}}}
    )
    assert "**cursor-agent · shell**" in markdown
    assert "sysctl vm.swappiness" in markdown
    assert markdown.strip().startswith("**cursor-agent")


def test_format_cursor_tool_completed_markdown():
    markdown = format_cursor_tool_completed_markdown(
        {
            "tool_call": {
                "readToolCall": {
                    "result": {"success": {"content": "file body"}},
                }
            }
        }
    )
    assert "**Result**" in markdown
    assert "file body" in markdown


def test_delete_tool_call_markdown():
    started = format_cursor_tool_started_markdown(
        {"tool_call": {"deleteToolCall": {"args": {"path": "/tmp/to-delete.txt"}}}}
    )
    assert "**cursor-agent · delete**" in started
    assert "to-delete.txt" in started


def test_glob_tool_call_markdown():
    started = format_cursor_tool_started_markdown(
        {
            "tool_call": {
                "globToolCall": {
                    "args": {"globPattern": "**/*.txt", "targetDirectory": "/workspace"},
                }
            }
        }
    )
    assert "**cursor-agent · glob**" in started
    assert "**/*.txt" in started
    assert "/workspace" in started


def test_grep_tool_call_markdown():
    started = format_cursor_tool_started_markdown(
        {
            "tool_call": {
                "grepToolCall": {
                    "args": {"pattern": "TOKEN", "path": "/workspace"},
                }
            }
        }
    )
    assert "**cursor-agent · grep**" in started
    assert "TOKEN" in started


def test_unknown_tool_renders_json_fence():
    started = format_cursor_tool_started_markdown(
        {
            "tool_call": {
                "fooToolCall": {
                    "args": {"alpha": 1, "beta": "two"},
                }
            }
        }
    )
    assert "**cursor-agent · unknown**" in started
    assert "```json" in started
    assert '"tool": "fooToolCall"' in started
    assert '"phase": "started"' in started
    assert '"alpha": 1' in started

    completed = format_cursor_tool_completed_markdown(
        {
            "tool_call": {
                "fooToolCall": {
                    "args": {"alpha": 1},
                    "result": {"success": {"status": "ok"}},
                }
            }
        }
    )
    assert "```json" in completed
    assert '"phase": "completed"' in completed
    assert '"status": "ok"' in completed


def test_cursor_response_collector_alternates_message_and_reasoning():
    collector = CursorResponseCollector()
    collector.consume({"type": "text_delta", "delta": "Inspecting memory."})
    collector.consume(
        {
            "type": "tool_started",
            "call_id": "t1",
            "markdown": format_cursor_tool_started_markdown(
                {"tool_call": {"readToolCall": {"args": {"path": "/etc/sysctl.conf"}}}}
            ),
        }
    )
    collector.consume(
        {
            "type": "tool_completed",
            "call_id": "t1",
            "markdown": format_cursor_tool_completed_markdown(
                {"tool_call": {"readToolCall": {"result": {"success": {"content": "vm.swappiness=60"}}}}}
            ),
        }
    )
    collector.consume({"type": "text_delta", "delta": "Applying fixes."})
    output = collector.build_output()
    assert len(output) == 3
    assert output[0]["type"] == "message"
    assert "Inspecting memory." in output[0]["content"][0]["text"]
    assert output[1]["type"] == "reasoning"
    assert "cursor-agent" in output[1]["summary"][0]["text"]
    assert "/etc/sysctl.conf" in output[1]["summary"][0]["text"]
    assert output[2]["type"] == "message"
    assert "Applying fixes." in output[2]["content"][0]["text"]


async def test_iter_cursor_agent_events_does_not_kill_normal_completion(monkeypatch):
    class FakeStdin:
        def write(self, data):
            self.data = data

        async def drain(self):
            pass

        def close(self):
            pass

    class FakeReader:
        def __init__(self, chunks):
            self.chunks = list(chunks)

        async def read(self, _size):
            if self.chunks:
                return self.chunks.pop(0)
            return b""

    class FakeProc:
        def __init__(self):
            self.stdin = FakeStdin()
            self.stdout = FakeReader(
                [
                    json.dumps(
                        {
                            "type": "result",
                            "subtype": "success",
                            "result": "Hello",
                        }
                    ).encode()
                    + b"\n",
                    b"",
                ]
            )
            self.stderr = FakeReader([b""])
            self.returncode = None
            self.killed = False

        def kill(self):
            self.killed = True
            self.returncode = -9

        async def wait(self):
            if self.returncode is None:
                self.returncode = 0
            return self.returncode

    proc = FakeProc()

    async def fake_create_subprocess_exec(*_args, **_kwargs):
        return proc

    monkeypatch.setattr(
        "codex_shim.cursor_passthrough.asyncio.create_subprocess_exec",
        fake_create_subprocess_exec,
    )

    events = [event async for event in iter_cursor_agent_events("prompt", "composer-2.5")]

    assert proc.killed is False
    assert proc.returncode == 0
    assert events[-1] == {"type": "completed", "text": "Hello"}

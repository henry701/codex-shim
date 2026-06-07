from __future__ import annotations

import json

from codex_shim.cursor_passthrough import (
    CursorStreamParser,
    _parse_cursor_list_models_output,
    build_cursor_prompt,
    cursor_upstream_model,
    is_cursor_passthrough_slug,
    iter_cursor_agent_events,
)


def test_parse_cursor_list_models_output():
    models = _parse_cursor_list_models_output(
        "auto - Auto\ncomposer-2.5 - Composer 2.5\ngpt-5.3-codex - Codex 5.3\n"
    )
    assert models["cursor-composer-2-5"].upstream_id == "composer-2.5"
    assert models["cursor-composer-2-5"].display_name == "Composer 2.5"
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
            "message": {"role": "assistant", "content": [{"type": "text", "text": "Hello"}]},
            "timestamp_ms": 2,
        }
    )
    assert parser.feed_line(line1) == "Hel"
    assert parser.feed_line(line2) == "lo"


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

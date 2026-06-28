from __future__ import annotations

import json
from pathlib import Path

import pytest

from codex_shim.cursor_passthrough import (
    CursorResponseCollector,
    CursorStreamParser,
    format_cursor_tool_started_markdown,
    replay_cursor_ndjson,
)
from codex_shim.cursor_stream_visualizer import visualize_ndjson_file

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "cursor_stream"
FIXTURE_NAMES = [
    "sleep_shell_read.ndjson",
    "write_edit_patch.ndjson",
    "move_rename.ndjson",
    "delete_file.ndjson",
    "write_new.ndjson",
    "list_glob.ndjson",
    "grep_search.ndjson",
]


def _load_fixture(name: str) -> str:
    return (FIXTURES / name).read_text()


def test_fixture_sleep_shell_read_replay_has_no_duplicate_prose():
    events, parser, collector = replay_cursor_ndjson(_load_fixture("sleep_shell_read.ndjson"), collect_output=True)
    assert collector is not None
    assembled = "".join(event["delta"] for event in events if event["type"] == "text_delta")
    assert assembled == parser.final_text
    assert "done-after-sleep" in assembled
    output = collector.build_output(fallback_text=parser.final_text)
    assert output[-1]["type"] == "message"
    reasoning = [item for item in output if item["type"] == "reasoning"]
    assert len(reasoning) >= 2
    shell_blocks = [item for item in reasoning if "shell" in item["summary"][0]["text"]]
    read_blocks = [item for item in reasoning if "read" in item["summary"][0]["text"]]
    assert shell_blocks and read_blocks
    thinking_blocks = [item for item in reasoning if "thinking" in item["summary"][0]["text"]]
    assert len(thinking_blocks) == 1
    assert "Running sleep 1" in thinking_blocks[0]["summary"][0]["text"]
    assert "reading" in thinking_blocks[0]["summary"][0]["text"]


def test_fixture_replay_event_counts():
    events, _, _ = replay_cursor_ndjson(_load_fixture("sleep_shell_read.ndjson"))
    counts = {}
    for event in events:
        counts[event["type"]] = counts.get(event["type"], 0) + 1
    assert counts.get("tool_started") == 2
    assert counts.get("tool_completed") == 2
    assert counts.get("thinking_delta", 0) >= 3
    assert counts.get("thinking_completed") == 1
    assert counts.get("segment_boundary", 0) >= 1


def test_incremental_assistant_fragments_and_cumulative_prefix():
    parser = CursorStreamParser()
    incremental = [
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "Hel"}]}, "timestamp_ms": 1},
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "lo"}]}, "timestamp_ms": 2},
    ]
    deltas = []
    for obj in incremental:
        deltas.extend(parser.feed_events(json.dumps(obj)))
    assert [event["delta"] for event in deltas if event["type"] == "text_delta"] == ["Hel", "lo"]
    assert parser.segment_text == "Hello"

    parser2 = CursorStreamParser()
    cumulative = [
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "Hel"}]}, "timestamp_ms": 1},
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "Hello"}]}, "timestamp_ms": 2},
    ]
    deltas2 = []
    for obj in cumulative:
        deltas2.extend(parser2.feed_events(json.dumps(obj)))
    assert [event["delta"] for event in deltas2 if event["type"] == "text_delta"] == ["Hel", "lo"]


def test_buffered_cumulative_assistant_flush_is_skipped_not_re_emitted():
    parser = CursorStreamParser()
    all_events: list[dict] = []
    for obj in [
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "Done."}]}, "timestamp_ms": 1},
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "Done."}]}},
    ]:
        all_events.extend(parser.feed_events(json.dumps(obj)))
    text_events = [event for event in all_events if event["type"] == "text_delta"]
    boundary = [event for event in all_events if event["type"] == "segment_boundary"]
    assert len(text_events) == 1
    assert text_events[0]["delta"] == "Done."
    assert len(boundary) == 1


def test_write_tool_call_markdown():
    started = format_cursor_tool_started_markdown(
        {
            "tool_call": {
                "writeToolCall": {
                    "args": {
                        "path": "patch-note.txt",
                        "fileText": "patched content line\n",
                    }
                }
            }
        }
    )
    assert "write" in started
    assert "patch-note.txt" in started
    assert "patched content" in started


def test_edit_tool_call_markdown():
    started = format_cursor_tool_started_markdown(
        {
            "tool_call": {
                "editToolCall": {
                    "args": {
                        "path": "marker.txt",
                        "streamContent": "patched-marker",
                    }
                }
            }
        }
    )
    assert "edit" in started
    assert "marker.txt" in started
    assert "patched-marker" in started


def test_fixture_write_edit_patch_replay():
    events, parser, collector = replay_cursor_ndjson(
        _load_fixture("write_edit_patch.ndjson"), collect_output=True
    )
    assert collector is not None
    edit_started = [event for event in events if event["type"] == "tool_started" and "edit" in event.get("markdown", "")]
    assert len(edit_started) >= 2
    assembled = "".join(event["delta"] for event in events if event["type"] == "text_delta")
    assert "patched-marker" in assembled or "patched-marker" in parser.final_text
    output = collector.build_output(fallback_text=parser.final_text)
    edit_blocks = [
        item
        for item in output
        if item["type"] == "reasoning" and "edit" in item["summary"][0]["text"]
    ]
    assert len(edit_blocks) >= 2


def test_fixture_delete_file_replay():
    events, _, collector = replay_cursor_ndjson(_load_fixture("delete_file.ndjson"), collect_output=True)
    assert collector is not None
    delete_started = [event for event in events if event["type"] == "tool_started" and "delete" in event.get("markdown", "")]
    assert delete_started
    output = collector.build_output()
    delete_blocks = [
        item for item in output if item["type"] == "reasoning" and "delete" in item["summary"][0]["text"]
    ]
    assert delete_blocks


def test_fixture_list_glob_replay():
    events, _, collector = replay_cursor_ndjson(_load_fixture("list_glob.ndjson"), collect_output=True)
    assert collector is not None
    glob_started = [event for event in events if event["type"] == "tool_started" and "glob" in event.get("markdown", "")]
    assert glob_started
    output = collector.build_output()
    glob_blocks = [
        item for item in output if item["type"] == "reasoning" and "glob" in item["summary"][0]["text"]
    ]
    assert glob_blocks


def test_fixture_grep_search_replay():
    events, _, collector = replay_cursor_ndjson(_load_fixture("grep_search.ndjson"), collect_output=True)
    assert collector is not None
    grep_started = [event for event in events if event["type"] == "tool_started" and "grep" in event.get("markdown", "")]
    assert grep_started
    output = collector.build_output()
    grep_blocks = [
        item for item in output if item["type"] == "reasoning" and "grep" in item["summary"][0]["text"]
    ]
    assert grep_blocks


def test_all_fixtures_replay_without_error():
    for name in FIXTURE_NAMES:
        events, parser, _ = replay_cursor_ndjson(_load_fixture(name))
        assert isinstance(events, list)
        assert parser.error is None


def test_tmux_smoke_script_is_valid_bash():
    import subprocess

    script = Path(__file__).resolve().parents[1] / "scripts" / "cursor-passthrough-smoke-tmux.sh"
    assert script.is_file()
    assert script.stat().st_mode & 0o111
    subprocess.run(["bash", "-n", str(script)], check=True)


def test_visualizer_cli_module_entrypoint(tmp_path):
    import subprocess

    fixture_copy = tmp_path / "stream.ndjson"
    fixture_copy.write_text(_load_fixture("sleep_shell_read.ndjson"))
    proc = subprocess.run(
        [
            "python3",
            "-m",
            "codex_shim.cursor_stream_visualizer",
            str(fixture_copy),
            "--workdir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=True,
        cwd=str(Path(__file__).resolve().parents[1]),
    )
    assert "[prose]" in proc.stdout
    assert "assembled prose:" in proc.stdout


def test_thinking_deltas_merge_in_collector():
    collector = CursorResponseCollector()
    collector.consume({"type": "thinking_delta", "delta": "Planning "})
    collector.consume({"type": "thinking_delta", "delta": "the patch."})
    collector.consume({"type": "thinking_completed"})
    output = collector.build_output()
    assert len([item for item in output if item["type"] == "reasoning"]) == 1
    assert "Planning the patch." in output[0]["summary"][0]["text"]


def test_connection_interrupted_marks_pending_tools():
    collector = CursorResponseCollector()
    collector.consume(
        {
            "type": "tool_started",
            "call_id": "t1",
            "markdown": format_cursor_tool_started_markdown(
                {"tool_call": {"shellToolCall": {"args": {"command": "sleep 5"}}}}
            ),
        }
    )
    collector.consume({"type": "connection_interrupted", "message": "> interrupted\n"})
    output = collector.build_output()
    assert output[0]["type"] == "reasoning"
    assert "interrupted" in output[0]["summary"][0]["text"]


def test_visualizer_renders_fixture_summary(tmp_path):
    fixture_copy = tmp_path / "stream.ndjson"
    fixture_copy.write_text(_load_fixture("sleep_shell_read.ndjson"))
    import io

    buf = io.StringIO()
    summary = visualize_ndjson_file(fixture_copy, stream=buf, workdir=str(tmp_path))
    text = buf.getvalue()
    assert "[prose]" in text
    assert "[think]" in text
    assert "[tool" in text
    assert "reasoning block:" in text
    assert summary["assembled"] == summary["final_text"]


def test_tmux_visualizer_renders_in_pane_when_available():
    import shutil
    import subprocess
    import uuid

    if not shutil.which("tmux"):
        pytest.skip("tmux not installed")

    root = Path(__file__).resolve().parents[1]
    fixture = FIXTURES / "sleep_shell_read.ndjson"
    workdir = Path(pytest.importorskip("tempfile").mkdtemp(prefix="cursor-viz-"))
    session = f"codex-shim-viz-test-{uuid.uuid4().hex[:8]}"
    out_file = workdir / "viz.out"
    signal = f"viz-done-{uuid.uuid4().hex[:8]}"
    try:
        subprocess.run(
            [
                "tmux",
                "new-session",
                "-d",
                "-s",
                session,
                "bash",
                "-lc",
                (
                    f"cd {root!s} && python3 -m codex_shim.cursor_stream_visualizer "
                    f"{fixture!s} --workdir {workdir!s} >{out_file!s} 2>&1; "
                    f"tmux wait-for -S {signal}"
                ),
            ],
            check=True,
            timeout=30,
        )
        subprocess.run(["tmux", "wait-for", signal], check=True, timeout=30)
        text = out_file.read_text()
        assert "[think]" in text
        assert "[tool" in text
        assert "[prose]" in text
        assert "assembled prose:" in text
    finally:
        subprocess.run(["tmux", "kill-session", "-t", session], check=False, timeout=10)


@pytest.mark.integration
def test_live_cursor_agent_smoke_optional():
    """Run only when explicitly requested: pytest -m integration."""
    pytest.importorskip("shutil")
    import shutil
    import subprocess

    if not shutil.which("cursor-agent"):
        pytest.skip("cursor-agent not installed")
    proc = subprocess.run(["cursor-agent", "status"], capture_output=True, text=True, timeout=20)
    if "logged in" not in (proc.stdout + proc.stderr).lower():
        pytest.skip("cursor-agent not logged in")

    workdir = Path(pytest.importorskip("tempfile").mkdtemp(prefix="cursor-smoke-"))
    (workdir / "marker.txt").write_text("initial\n")
    prompt = "Append done-smoke to marker.txt with shell, then read it. One sentence reply."
    proc = subprocess.run(
        [
            "cursor-agent",
            "--print",
            "--output-format",
            "stream-json",
            "--stream-partial-output",
            "--force",
            "--trust",
            "--workspace",
            str(workdir),
            "--model",
            "composer-2.5",
            prompt,
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    events, parser, collector = replay_cursor_ndjson(lines, collect_output=True)
    assert collector is not None
    assert any(event["type"] == "tool_started" for event in events)
    assert parser.final_text or any(event["type"] == "text_delta" for event in events)
    assert "done-smoke" in (workdir / "marker.txt").read_text()

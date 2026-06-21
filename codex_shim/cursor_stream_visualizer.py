"""Terminal visualization for cursor-agent NDJSON replay (smoke / debug)."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, TextIO

from .cursor_passthrough import CursorResponseCollector, CursorStreamParser, replay_cursor_ndjson

_ANSI = {
    "reset": "\033[0m",
    "dim": "\033[2m",
    "bold": "\033[1m",
    "cyan": "\033[36m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "magenta": "\033[35m",
    "blue": "\033[34m",
}


def _c(code: str, text: str) -> str:
    return f"{_ANSI.get(code, '')}{text}{_ANSI['reset']}"


def format_event_line(event: dict[str, Any]) -> str:
    event_type = str(event.get("type") or "?")
    if event_type == "text_delta":
        return _c("green", f"[prose] {event.get('delta', '')!r}")
    if event_type == "thinking_delta":
        return _c("magenta", f"[think] {event.get('delta', '')!r}")
    if event_type == "thinking_completed":
        return _c("magenta", "[think] (completed)")
    if event_type == "tool_started":
        preview = str(event.get("markdown") or "").splitlines()[0]
        return _c("yellow", f"[tool▶] {preview}")
    if event_type == "tool_completed":
        return _c("yellow", "[tool✓] (result appended)")
    if event_type == "segment_boundary":
        return _c("dim", "[segment boundary]")
    if event_type == "connection_interrupted":
        return _c("blue", "[connection interrupted]")
    return _c("dim", json.dumps(event, ensure_ascii=False))


def visualize_ndjson_file(
    path: Path,
    *,
    stream: TextIO | None = None,
    delay_sec: float = 0.0,
    workdir: str | None = None,
) -> dict[str, Any]:
    """Replay a captured NDJSON file with colored event lines."""
    out = stream or sys.stdout
    text = path.read_text()
    if workdir:
        text = text.replace("{{WORKDIR}}", workdir)
    parser = CursorStreamParser()
    collector = CursorResponseCollector()
    counts: dict[str, int] = {}
    assembled = ""
    for line in text.splitlines():
        if not line.strip():
            continue
        for event in parser.feed_events(line):
            counts[event["type"]] = counts.get(event["type"], 0) + 1
            collector.consume(event)
            if event.get("type") == "text_delta":
                assembled += str(event.get("delta") or "")
            out.write(format_event_line(event) + "\n")
            out.flush()
            if delay_sec:
                time.sleep(delay_sec)
    output = collector.build_output(fallback_text=parser.final_text)
    out.write("\n")
    out.write(_c("bold", "--- summary ---") + "\n")
    out.write(f"assembled prose: {assembled!r}\n")
    out.write(f"final result: {parser.final_text!r}\n")
    out.write(f"event counts: {counts}\n")
    out.write(f"output items: {[item.get('type') for item in output]}\n")
    for item in output:
        if item.get("type") == "reasoning":
            body = (item.get("summary") or [{}])[0].get("text", "")
            out.write(_c("cyan", f"\nreasoning block:\n{body}\n"))
    return {
        "assembled": assembled,
        "final_text": parser.final_text,
        "counts": counts,
        "output": output,
    }


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Visualize cursor-agent NDJSON capture")
    parser.add_argument("ndjson", type=Path, help="Path to stream.ndjson capture")
    parser.add_argument("--delay", type=float, default=0.0, help="Pause between events (seconds)")
    parser.add_argument("--workdir", default="", help="Replace {{WORKDIR}} placeholder in fixture")
    args = parser.parse_args(argv)
    visualize_ndjson_file(
        args.ndjson,
        delay_sec=args.delay,
        workdir=args.workdir or None,
    )


if __name__ == "__main__":
    main()

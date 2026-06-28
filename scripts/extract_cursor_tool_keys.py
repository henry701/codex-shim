#!/usr/bin/env python3
"""Inventory cursor-agent stream-json tool_call keys from NDJSON captures."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def _tool_keys(tool_call: dict[str, Any]) -> list[str]:
    keys: list[str] = []
    for key, value in tool_call.items():
        if key in {"hookAdditionalContexts", "toolCallId", "startedAtMs", "completedAtMs"}:
            continue
        if isinstance(value, dict) and ("args" in value or "result" in value or key.endswith("ToolCall")):
            keys.append(key)
    return keys


def _args_keys(tool_call: dict[str, Any], tool_key: str) -> list[str]:
    payload = tool_call.get(tool_key)
    if not isinstance(payload, dict):
        return []
    args = payload.get("args")
    if not isinstance(args, dict):
        return []
    return sorted(args.keys())


def scan_file(path: Path) -> dict[str, Any]:
    counts: dict[str, int] = defaultdict(int)
    args_samples: dict[str, set[str]] = defaultdict(set)
    phases: dict[str, set[str]] = defaultdict(set)
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if obj.get("type") != "tool_call":
            continue
        subtype = str(obj.get("subtype") or "")
        tool_call = obj.get("tool_call")
        if not isinstance(tool_call, dict):
            continue
        for key in _tool_keys(tool_call):
            counts[key] += 1
            phases[key].add(subtype or "?")
            for arg_key in _args_keys(tool_call, key):
                args_samples[key].add(arg_key)
    return {
        "file": str(path),
        "tools": {
            key: {
                "count": counts[key],
                "phases": sorted(phases[key]),
                "sample_args_keys": sorted(args_samples[key]),
            }
            for key in sorted(counts)
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ndjson", nargs="+", type=Path, help="NDJSON capture file(s)")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text summary")
    args = parser.parse_args(argv)

    reports = [scan_file(path) for path in args.ndjson]
    if args.json:
        json.dump(reports if len(reports) > 1 else reports[0], sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0

    for report in reports:
        print(f"=== {report['file']} ===")
        if not report["tools"]:
            print("  (no tool_call events)")
            continue
        for key, info in report["tools"].items():
            phases = ", ".join(info["phases"])
            arg_keys = ", ".join(info["sample_args_keys"]) or "(none)"
            print(f"  {key}: count={info['count']} phases=[{phases}] args=[{arg_keys}]")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

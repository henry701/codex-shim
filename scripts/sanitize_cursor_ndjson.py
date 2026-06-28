#!/usr/bin/env python3
"""Replace absolute workspace paths with {{WORKDIR}} in cursor-agent NDJSON captures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def sanitize_text(text: str, workdir: str) -> str:
    workdir = workdir.rstrip("/")
    if not workdir:
        return text
    return text.replace(workdir, "{{WORKDIR}}")


def sanitize_obj(obj: object, workdir: str) -> object:
    if isinstance(obj, str):
        return sanitize_text(obj, workdir)
    if isinstance(obj, list):
        return [sanitize_obj(item, workdir) for item in obj]
    if isinstance(obj, dict):
        return {key: sanitize_obj(value, workdir) for key, value in obj.items()}
    return obj


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", required=True, help="Absolute workspace path to replace")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    lines_out: list[str] = []
    for line in args.input.read_text().splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError:
            lines_out.append(sanitize_text(line, args.workdir))
            continue
        sanitized = sanitize_obj(obj, args.workdir)
        lines_out.append(json.dumps(sanitized, ensure_ascii=False))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines_out) + ("\n" if lines_out else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

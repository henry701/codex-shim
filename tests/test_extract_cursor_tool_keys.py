from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.extract_cursor_tool_keys import scan_file

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "cursor_stream"


def test_scan_file_finds_delete_tool():
    report = scan_file(FIXTURES / "delete_file.ndjson")
    assert "deleteToolCall" in report["tools"]
    assert report["tools"]["deleteToolCall"]["count"] == 2
    assert "path" in report["tools"]["deleteToolCall"]["sample_args_keys"]


def test_scan_file_finds_grep_tool():
    report = scan_file(FIXTURES / "grep_search.ndjson")
    assert "grepToolCall" in report["tools"]
    assert "pattern" in report["tools"]["grepToolCall"]["sample_args_keys"]


def test_main_json_output(tmp_path, capsys):
    sample = tmp_path / "sample.ndjson"
    sample.write_text(
        json.dumps(
            {
                "type": "tool_call",
                "subtype": "started",
                "tool_call": {"globToolCall": {"args": {"globPattern": "*.txt"}}},
            }
        )
        + "\n"
    )
    from scripts.extract_cursor_tool_keys import main

    assert main([str(sample), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["tools"]["globToolCall"]["count"] == 1

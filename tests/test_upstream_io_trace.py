from __future__ import annotations

import json

from codex_shim.upstream_io_trace import (
    is_verbose_error_surface,
    record_upstream_error,
    summarize_upstream_body,
)


def test_summarize_upstream_body_lists_all_input_item_types():
    body = {
        "model": "codex-gpt-5-5",
        "stream": True,
        "input": [
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "one"}]},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "two"}]},
            {"type": "compaction_trigger"},
        ],
    }
    summary = summarize_upstream_body(body)
    assert summary["input_items"] == 3
    assert summary["input_item_types"][-1] == "compaction_trigger"
    assert "chars=3" in summary["input_item_types"][0]


def test_summarize_upstream_body_truncates_long_input_types_without_debug(monkeypatch):
    monkeypatch.delenv("CODEX_SHIM_REQUEST_LOG", raising=False)
    monkeypatch.delenv("CODEX_SHIM_PASSTHROUGH_TRACE", raising=False)
    monkeypatch.delenv("CODEX_SHIM_STREAM_LOG", raising=False)
    body = {
        "model": "codex-gpt-5-5",
        "input": [{"type": "message", "role": "user", "content": "x"} for _ in range(40)],
    }
    summary = summarize_upstream_body(body)
    assert summary["input_items"] == 40
    assert summary["input_item_types"][0].startswith("…(")
    assert len(summary["input_item_types"]) == 21  # marker + tail 20


def test_is_verbose_error_surface_for_turns_not_models():
    assert is_verbose_error_surface("chatgpt-passthrough")
    assert is_verbose_error_surface("chatgpt-compact")
    assert is_verbose_error_surface("byok-openai-responses:local")
    assert not is_verbose_error_surface("models-catalog")
    assert not is_verbose_error_surface("picker-api")
    assert not is_verbose_error_surface("discover-refresh")


def test_record_upstream_error_verbose_for_response_surfaces(monkeypatch, tmp_path):
    monkeypatch.delenv("CODEX_SHIM_REQUEST_LOG", raising=False)
    monkeypatch.delenv("CODEX_SHIM_PASSTHROUGH_TRACE", raising=False)
    monkeypatch.delenv("CODEX_SHIM_STREAM_LOG", raising=False)
    monkeypatch.setattr("codex_shim.upstream_io_trace.UPSTREAM_IO_DIR", tmp_path)

    body = {"model": "x", "input": [{"type": "message", "role": "user", "content": "hello-world"}]}
    record_upstream_error("chatgpt-passthrough", "https://example", 429, "limit", request_body=body)

    assert (tmp_path / "last-error.json").exists()
    assert list(tmp_path.glob("error-*.json")) == []
    payload = json.loads((tmp_path / "last-error.json").read_text())
    assert payload["request_body"]["input"][0]["content"] == "hello-world"
    assert "limit" in payload["response_text"]


def test_record_upstream_error_quiet_for_model_surfaces(monkeypatch, tmp_path):
    monkeypatch.delenv("CODEX_SHIM_REQUEST_LOG", raising=False)
    monkeypatch.delenv("CODEX_SHIM_PASSTHROUGH_TRACE", raising=False)
    monkeypatch.delenv("CODEX_SHIM_STREAM_LOG", raising=False)
    monkeypatch.setattr("codex_shim.upstream_io_trace.UPSTREAM_IO_DIR", tmp_path)

    huge = {"model": "x", "input": [{"type": "message", "role": "user", "content": "y" * 50_000}]}
    record_upstream_error("models-catalog", "https://example", 500, "boom", request_body=huge)

    payload = json.loads((tmp_path / "last-error.json").read_text())
    assert payload["request_body"]["input_items"] == 1
    assert "input" not in payload["request_body"]
    assert list(tmp_path.glob("error-*.json")) == []


def test_record_upstream_error_writes_timestamped_dump_when_debug(monkeypatch, tmp_path):
    monkeypatch.setenv("CODEX_SHIM_REQUEST_LOG", "1")
    monkeypatch.setattr("codex_shim.upstream_io_trace.UPSTREAM_IO_DIR", tmp_path)

    record_upstream_error("chatgpt", "https://example", 500, "boom", request_body={"model": "x", "input": []})
    assert (tmp_path / "last-error.json").exists()
    assert list(tmp_path.glob("error-*.json"))

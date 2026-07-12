from __future__ import annotations

import gzip
import json
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from codex_shim.quota_dashboard import (
    LONG_CONTEXT_INPUT,
    UsageTurn,
    discover_log_files,
    estimate_credits,
    format_report,
    main,
    parse_log_files,
    summarize,
)


def _write_sample_log(path: Path) -> None:
    path.write_text(
        "[req] /v1/responses/ws transport=ws model='codex-gpt-5-6-terra' stream=True "
        "previous_response_id=None tools=0 ([]) input=1 (['message(role=user)']) "
        "prompt_cache_key='thread-1'\n"
        "[upstream-headers] source=chatgpt-passthrough-ws "
        "usage={'input_tokens':280000,'input_tokens_details':{'cache_write_tokens':0,'cached_tokens':279000},"
        "'output_tokens':40,'output_tokens_details':{'reasoning_tokens':5},'total_tokens':280040}\n"
        "[req] /v1/responses/ws transport=ws model='codex-gpt-5-6-terra' stream=True "
        "previous_response_id='resp_1' tools=2 ([]) input=1 (['custom_tool_call_output'])\n"
        "[upstream-headers] source=chatgpt-passthrough-ws "
        "usage={'input_tokens':100000,'input_tokens_details':{'cache_write_tokens':0,'cached_tokens':99000},"
        "'output_tokens':20,'output_tokens_details':{'reasoning_tokens':0},'total_tokens':100020}\n"
        "[req] /v1/responses transport=http model='codex-gpt-5-6-sol' stream=True "
        "previous_response_id=None tools=0 ([]) input=120 (['message(role=user)'])\n"
    )


def test_parse_log_files_extracts_req_and_usage(tmp_path: Path):
    log = tmp_path / "shim.log"
    _write_sample_log(log)
    report = parse_log_files([log])
    assert len(report.req_events) == 3
    assert report.req_events[0].model == "codex-gpt-5-6-terra"
    assert report.req_events[0].prompt_cache_key == "thread-1"
    assert report.req_events[0].input_count == 1
    assert report.req_events[1].input_summary == "['custom_tool_call_output']"
    assert report.req_events[2].model == "codex-gpt-5-6-sol"
    assert report.req_events[2].input_count == 120
    assert len(report.usage_turns) == 2
    assert report.usage_turns[0].input_tokens == 280000
    assert report.usage_turns[0].model == "codex-gpt-5-6-terra"
    summary = summarize(report)
    assert summary["latest_req_model"] == "codex-gpt-5-6-sol"
    assert summary["req_delta_input_1"] == 2
    assert summary["req_full_history"] == 1
    assert summary["req_tool_continuations"] == 1
    assert summary["req_user_messages"] == 1
    assert summary["usage_by_model"]["codex-gpt-5-6-terra"]["long_context_turns"] == 1
    assert summary["long_context_threshold_input"] == LONG_CONTEXT_INPUT


def test_discover_log_files_reads_gz_and_filters_by_mtime(tmp_path: Path):
    import os

    plain = tmp_path / "shim.log"
    plain.write_text("[req] noop\n")
    gz = tmp_path / "shim.log.1.gz"
    with gzip.open(gz, "wt", encoding="utf-8") as handle:
        handle.write(
            "[req] /v1/responses/ws transport=ws model='codex-gpt-5-6-sol' stream=True "
            "previous_response_id=None tools=0 ([]) input=1 (['message(role=user)'])\n"
        )

    target = datetime(2026, 7, 10, 12, 0, 0, tzinfo=timezone.utc).timestamp()
    os.utime(plain, (target, target))
    os.utime(gz, (target, target))

    old = tmp_path / "shim.log.2.gz"
    with gzip.open(old, "wt", encoding="utf-8") as handle:
        handle.write("old\n")
    os.utime(old, (datetime(2026, 7, 1, tzinfo=timezone.utc).timestamp(),) * 2)

    mtime = date(2026, 7, 10)
    paths = discover_log_files(tmp_path, since=mtime, until=mtime)
    assert plain in paths
    assert gz in paths
    assert old not in paths


def test_parse_gzip_log_contents(tmp_path: Path):
    gz = tmp_path / "shim.log.1.gz"
    with gzip.open(gz, "wt", encoding="utf-8") as handle:
        handle.write(
            "[req] /v1/responses/ws transport=ws model='codex-gpt-5-6-sol' stream=True "
            "previous_response_id=None tools=0 ([]) input=1 (['message(role=user)'])\n"
            "[upstream-headers] source=chatgpt-passthrough-ws "
            "usage={'input_tokens':50000,'input_tokens_details':{'cached_tokens':40000},"
            "'output_tokens':10,'output_tokens_details':{},'total_tokens':50010}\n"
        )
    report = parse_log_files([gz])
    assert report.req_events[0].model == "codex-gpt-5-6-sol"
    assert report.usage_turns[0].cached_tokens == 40000


def test_estimate_credits_terra_short_vs_long_context():
    short = UsageTurn(
        model="codex-gpt-5-6-terra",
        input_tokens=100_000,
        cached_tokens=90_000,
        cache_write_tokens=0,
        output_tokens=100,
        reasoning_tokens=0,
    )
    long = UsageTurn(
        model="codex-gpt-5-6-terra",
        input_tokens=280_000,
        cached_tokens=270_000,
        cache_write_tokens=0,
        output_tokens=100,
        reasoning_tokens=0,
    )
    assert estimate_credits(long) > estimate_credits(short)
    # Terra short: cached 90k*6.25 + uncached 10k*62.5 + out 100*375
    assert estimate_credits(short) == pytest.approx((90_000 * 6.25 + 10_000 * 62.5 + 100 * 375) / 1_000_000)


def test_format_report_and_main_json(tmp_path: Path, capsys):
    log = tmp_path / "shim.log"
    _write_sample_log(log)
    summary = summarize(parse_log_files([log]))
    text = format_report(summary)
    assert "codex-gpt-5-6-terra" in text
    assert "Long-context API threshold" in text
    assert json.dumps(summary)

    code = main(["--log-dir", str(tmp_path), "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["req_total"] == 3


def test_usage_attributed_to_preceding_req_model_not_only_56(tmp_path: Path):
    log = tmp_path / "shim.log"
    log.write_text(
        "[req] /v1/responses/ws transport=ws model='codex-gpt-5-5' stream=True "
        "previous_response_id=None tools=0 ([]) input=1 (['message(role=user)'])\n"
        "[upstream-headers] source=chatgpt-passthrough-ws "
        "usage={'input_tokens':1000,'input_tokens_details':{'cached_tokens':900},"
        "'output_tokens':10,'output_tokens_details':{},'total_tokens':1010}\n"
        "[req] /v1/responses/ws transport=ws model='codex-gpt-5-6-terra' stream=True "
        "previous_response_id=None tools=0 ([]) input=1 (['message(role=user)'])\n"
        "[upstream-headers] source=chatgpt-passthrough-ws "
        "usage={'input_tokens':2000,'input_tokens_details':{'cached_tokens':1900},"
        "'output_tokens':10,'output_tokens_details':{},'total_tokens':2010}\n"
    )
    turns = parse_log_files([log]).usage_turns
    assert [(t.model, t.input_tokens) for t in turns] == [
        ("codex-gpt-5-5", 1000),
        ("codex-gpt-5-6-terra", 2000),
    ]

def test_main_returns_error_when_no_logs(tmp_path: Path, capsys):
    code = main(["--log-dir", str(tmp_path / "missing"), "--since", "2026-07-10"])
    assert code == 1
    assert "No shim.log files found" in capsys.readouterr().out

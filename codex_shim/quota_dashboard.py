"""Parse codex-shim logs and summarize GPT usage / quota signals."""

from __future__ import annotations

import argparse
import ast
import gzip
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, TextIO

LONG_CONTEXT_INPUT = 272_000

CREDIT_RATES = {
    "codex-gpt-5-6-sol": {"in": 125.0, "cached": 12.5, "out": 750.0, "lin": 250.0, "lcached": 25.0, "lout": 1125.0},
    "codex-gpt-5-6-terra": {"in": 62.5, "cached": 6.25, "out": 375.0, "lin": 125.0, "lcached": 12.5, "lout": 562.5},
    "codex-gpt-5-6-luna": {"in": 25.0, "cached": 2.5, "out": 150.0, "lin": 50.0, "lcached": 5.0, "lout": 225.0},
    "codex-gpt-5-5": {"in": 125.0, "cached": 12.5, "out": 750.0, "lin": 250.0, "lcached": 25.0, "lout": 1125.0},
}

REQ_RE = re.compile(
    r"\[req\] (?P<endpoint>\S+) transport=(?P<transport>\S+) model=(?P<model>'[^']+') "
    r"stream=(?P<stream>\S+) previous_response_id=(?P<prev>\S+) tools=(?P<tools>\d+)"
    r"(?: input=(?P<input>\d+))?"
)
USAGE_RE = re.compile(r"usage=(\{.*\})")
TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2})")


@dataclass
class UsageTurn:
    model: str
    input_tokens: int
    cached_tokens: int
    cache_write_tokens: int
    output_tokens: int
    reasoning_tokens: int
    source: str = ""


@dataclass
class ReqEvent:
    model: str
    endpoint: str
    transport: str
    input_count: int | None
    tools: int
    previous_response_id: str | None
    prompt_cache_key: str | None = None
    input_summary: str = ""


@dataclass
class DashboardReport:
    files: list[str] = field(default_factory=list)
    req_events: list[ReqEvent] = field(default_factory=list)
    usage_turns: list[UsageTurn] = field(default_factory=list)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize codex-shim shim.log usage for quota and cost monitoring.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path.home() / ".codex-shim",
        help="Directory containing shim.log and rotated archives (default: ~/.codex-shim)",
    )
    parser.add_argument(
        "--since",
        type=str,
        default=None,
        help="Include log files modified on/after this date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--until",
        type=str,
        default=None,
        help="Include log files modified on/before this date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of text report",
    )
    return parser.parse_args(argv)


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    return datetime.strptime(value, "%Y-%m-%d").date()


def discover_log_files(log_dir: Path, *, since: date | None, until: date | None) -> list[Path]:
    candidates = sorted(log_dir.glob("shim.log*"))
    if not candidates:
        return []
    selected: list[Path] = []
    for path in candidates:
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).date()
        if since and mtime < since:
            continue
        if until and mtime > until:
            continue
        selected.append(path)
    return selected


def iter_log_lines(paths: Iterable[Path]) -> Iterator[tuple[Path, str]]:
    for path in paths:
        opener: Any
        if path.suffix == ".gz" or path.name.endswith(".gz"):
            opener = lambda p=path: gzip.open(p, "rt", encoding="utf-8", errors="ignore")
        else:
            opener = lambda p=path: open(p, encoding="utf-8", errors="ignore")
        with opener() as handle:
            for line in handle:
                yield path, line.rstrip("\n")


def parse_log_files(paths: list[Path]) -> DashboardReport:
    report = DashboardReport(files=[str(path) for path in paths])
    model_ctx: str | None = None
    for path, line in iter_log_lines(paths):
        req = REQ_RE.search(line)
        if req:
            model = req.group("model").strip("'")
            input_raw = req.group("input")
            if input_raw is None:
                input_match = re.search(r"input=(\d+)", line)
                input_raw = input_match.group(1) if input_match else None
            pck = None
            pck_match = re.search(r"prompt_cache_key='([^']*)'", line)
            if pck_match:
                pck = pck_match.group(1)
            summary_match = re.search(r"input=\d+ \((.*)\)", line)
            report.req_events.append(
                ReqEvent(
                    model=model,
                    endpoint=req.group("endpoint"),
                    transport=req.group("transport"),
                    input_count=int(input_raw) if input_raw else None,
                    tools=int(req.group("tools")),
                    previous_response_id=req.group("prev").strip("'") if req.group("prev") != "None" else None,
                    prompt_cache_key=pck,
                    input_summary=summary_match.group(1) if summary_match else "",
                )
            )
            model_ctx = model

        if "[upstream-headers]" in line and "usage=" in line:
            um = USAGE_RE.search(line)
            if not um:
                continue
            try:
                usage = ast.literal_eval(um.group(1))
            except (SyntaxError, ValueError):
                continue
            if not isinstance(usage, dict):
                continue
            details = usage.get("input_tokens_details") or {}
            if not isinstance(details, dict):
                details = {}
            out_details = usage.get("output_tokens_details") or {}
            if not isinstance(out_details, dict):
                out_details = {}
            model = model_ctx or "unknown"
            report.usage_turns.append(
                UsageTurn(
                    model=model,
                    input_tokens=int(usage.get("input_tokens") or 0),
                    cached_tokens=int(details.get("cached_tokens") or 0),
                    cache_write_tokens=int(details.get("cache_write_tokens") or 0),
                    output_tokens=int(usage.get("output_tokens") or 0),
                    reasoning_tokens=int(out_details.get("reasoning_tokens") or 0),
                    source=line.split("source=", 1)[1].split(" ", 1)[0] if "source=" in line else "",
                )
            )
            model_ctx = None
    return report


def _credit_rates(model: str) -> dict[str, float]:
    if model in CREDIT_RATES:
        return CREDIT_RATES[model]
    if "5-6-sol" in model or model.endswith("gpt-5.6-sol"):
        return CREDIT_RATES["codex-gpt-5-6-sol"]
    if "5-6-terra" in model or model.endswith("gpt-5.6-terra"):
        return CREDIT_RATES["codex-gpt-5-6-terra"]
    if "5-6-luna" in model or model.endswith("gpt-5.6-luna"):
        return CREDIT_RATES["codex-gpt-5-6-luna"]
    if "5-5" in model:
        return CREDIT_RATES["codex-gpt-5-5"]
    return CREDIT_RATES["codex-gpt-5-6-terra"]


def estimate_credits(turn: UsageTurn) -> float:
    rates = _credit_rates(turn.model)
    long = turn.input_tokens > LONG_CONTEXT_INPUT
    if long:
        cached_rate, in_rate, out_rate = rates["lcached"], rates["lin"], rates["lout"]
    else:
        cached_rate, in_rate, out_rate = rates["cached"], rates["in"], rates["out"]
    uncached = max(0, turn.input_tokens - turn.cached_tokens - turn.cache_write_tokens)
    return (
        turn.cached_tokens * cached_rate
        + uncached * in_rate
        + turn.cache_write_tokens * in_rate * 1.25
        + turn.output_tokens * out_rate
    ) / 1_000_000


def summarize(report: DashboardReport) -> dict[str, Any]:
    by_model_usage: dict[str, list[UsageTurn]] = defaultdict(list)
    for turn in report.usage_turns:
        by_model_usage[turn.model].append(turn)

    usage_summary: dict[str, Any] = {}
    for model, turns in sorted(by_model_usage.items()):
        inputs = [t.input_tokens for t in turns if t.input_tokens > 0]
        long_count = sum(1 for t in turns if t.input_tokens > LONG_CONTEXT_INPUT)
        total_in = sum(t.input_tokens for t in turns)
        total_cached = sum(t.cached_tokens for t in turns)
        credits = sum(estimate_credits(t) for t in turns)
        usage_summary[model] = {
            "turns": len(turns),
            "input_tokens": total_in,
            "cached_tokens": total_cached,
            "cache_hit_pct": (total_cached / total_in * 100) if total_in else 0.0,
            "cache_write_tokens": sum(t.cache_write_tokens for t in turns),
            "output_tokens": sum(t.output_tokens for t in turns),
            "reasoning_tokens": sum(t.reasoning_tokens for t in turns),
            "long_context_turns": long_count,
            "long_context_pct": (long_count / len(turns) * 100) if turns else 0.0,
            "avg_input_tokens": (total_in / len(turns)) if turns else 0,
            "p50_input_tokens": sorted(inputs)[len(inputs) // 2] if inputs else 0,
            "max_input_tokens": max(inputs) if inputs else 0,
            "est_credits": round(credits, 2),
            "est_credits_per_turn": round(credits / len(turns), 3) if turns else 0.0,
        }

    req_by_model = Counter(event.model for event in report.req_events)
    delta_reqs = sum(1 for e in report.req_events if e.input_count == 1)
    full_reqs = sum(1 for e in report.req_events if e.input_count and e.input_count > 50)
    tool_reqs = sum(1 for e in report.req_events if "custom_tool_call_output" in e.input_summary)
    user_reqs = sum(1 for e in report.req_events if "message(role=user)" in e.input_summary and e.input_count == 1)

    latest_model = report.req_events[-1].model if report.req_events else None
    return {
        "files": report.files,
        "req_total": len(report.req_events),
        "req_by_model": dict(req_by_model),
        "latest_req_model": latest_model,
        "req_delta_input_1": delta_reqs,
        "req_full_history": full_reqs,
        "req_tool_continuations": tool_reqs,
        "req_user_messages": user_reqs,
        "usage_by_model": usage_summary,
        "long_context_threshold_input": LONG_CONTEXT_INPUT,
    }


def format_report(summary: dict[str, Any]) -> str:
    lines = [
        "Codex shim quota dashboard",
        f"Log files ({len(summary['files'])}):",
    ]
    for path in summary["files"]:
        lines.append(f"  - {path}")
    lines.extend(
        [
            "",
            f"Requests: {summary['req_total']} total",
            f"  latest model: {summary.get('latest_req_model')}",
            f"  by model: {summary.get('req_by_model')}",
            f"  delta (input=1): {summary.get('req_delta_input_1')}",
            f"  full history (input>50): {summary.get('req_full_history')}",
            f"  tool continuations: {summary.get('req_tool_continuations')}",
            f"  user message deltas: {summary.get('req_user_messages')}",
            "",
            f"Long-context API threshold: {summary['long_context_threshold_input']:,} input tokens",
            "",
            "Upstream usage:",
        ]
    )
    for model, stats in summary.get("usage_by_model", {}).items():
        lines.append(f"  {model}:")
        lines.append(
            f"    turns={stats['turns']} input={stats['input_tokens']:,} "
            f"cached={stats['cached_tokens']:,} ({stats['cache_hit_pct']:.1f}%) "
            f"write={stats['cache_write_tokens']:,}"
        )
        lines.append(
            f"    avg_in={stats['avg_input_tokens']:,.0f} p50={stats['p50_input_tokens']:,} "
            f"max={stats['max_input_tokens']:,} long_ctx={stats['long_context_turns']} ({stats['long_context_pct']:.1f}%)"
        )
        lines.append(
            f"    out={stats['output_tokens']:,} reasoning={stats['reasoning_tokens']:,} "
            f"est_credits={stats['est_credits']} (~{stats['est_credits_per_turn']}/turn)"
        )
    if not summary.get("usage_by_model"):
        lines.append("  (no upstream usage lines in selected logs)")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    since = _parse_date(args.since)
    until = _parse_date(args.until)
    paths = discover_log_files(args.log_dir.expanduser(), since=since, until=until)
    if not paths:
        print(f"No shim.log files found under {args.log_dir} for the selected date range.")
        return 1
    report = parse_log_files(paths)
    summary = summarize(report)
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(format_report(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Invariant lint for keepalive / throttle networking (TDD fixtures + tree scan)."""

from __future__ import annotations

from pathlib import Path

from lint.net_invariants import Violation, scan_source, scan_tree

ROOT = Path(__file__).resolve().parents[1]


def _rules(source: str) -> set[str]:
    return {item.rule for item in scan_source(source, path="<mem>")}


def test_lint_flags_ping_fn_retry_parameter():
    source = """
async def throttle_sleep(delay, *, ping_fn=None, disconnect_fn=None):
    if ping_fn is not None:
        await ping_fn()
    await asyncio.sleep(delay)
"""
    assert "retry-ping-fn-param" in _rules(source)


def test_lint_flags_ping_fn_keyword_argument():
    source = """
await retry_aiohttp_post(session, url, ping_fn=writer.ping, json=body)
"""
    assert "retry-ping-fn-kwarg" in _rules(source)


def test_lint_allows_retry_without_ping_fn():
    source = """
async def throttle_sleep(delay, *, disconnect_fn=None):
    await asyncio.sleep(delay)

await retry_aiohttp_post(session, url, json=body, disconnect_fn=disconnected)
"""
    assert _rules(source) == set()


def test_lint_flags_time_sleep_inside_async_def():
    source = """
import time

async def handle():
    time.sleep(1.0)
    await asyncio.sleep(0)
"""
    assert "async-time-sleep" in _rules(source)


def test_lint_allows_time_sleep_in_sync_def():
    source = """
import time

def throttle_sleep_sync(delay):
    time.sleep(delay)
"""
    rules = _rules(source)
    assert "async-time-sleep" not in rules


def test_lint_flags_unguarded_throttle_sleep_sync():
    source = """
import time

def throttle_sleep_sync(delay, sleep_fn=None):
    if sleep_fn is not None:
        sleep_fn(delay)
        return
    time.sleep(delay)
"""
    assert "throttle-sleep-sync-unguarded" in _rules(source)


def test_lint_allows_loop_guarded_throttle_sleep_sync():
    source = """
import time
import asyncio

def throttle_sleep_sync(delay, sleep_fn=None):
    if sleep_fn is not None:
        sleep_fn(delay)
        return
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        time.sleep(delay)
        return
    raise RuntimeError("sync throttle wait on the aiohttp event loop")
"""
    assert "throttle-sleep-sync-unguarded" not in _rules(source)


def test_lint_flags_pinger_that_emits_before_sleep():
    source = """
class DownstreamPinger:
    async def _run(self):
        while True:
            await self._emit()
            await asyncio.sleep(self.interval)
"""
    assert "pinger-sleep-before-ping" in _rules(source)


def test_lint_allows_pinger_that_sleeps_before_emit():
    source = """
class DownstreamPinger:
    async def _run(self):
        while True:
            await asyncio.sleep(self.interval)
            await self._emit()
"""
    assert "pinger-sleep-before-ping" not in _rules(source)


def test_lint_flags_keepalive_max_at_or_above_desktop_idle():
    source = """
KEEPALIVE_MIN = 4.0
KEEPALIVE_MAX = 15.0
"""
    assert "keepalive-max-vs-desktop" in _rules(source)


def test_lint_allows_keepalive_max_below_desktop_idle():
    source = """
KEEPALIVE_MIN = 4.0
KEEPALIVE_MAX = 6.0
"""
    assert "keepalive-max-vs-desktop" not in _rules(source)


def test_lint_flags_urlopen_inside_async_def():
    source = """
import urllib.request

async def search():
    urllib.request.urlopen("https://example.com")
"""
    assert "async-blocking-http" in _rules(source)


def test_lint_flags_request_urllib_inside_async_def():
    source = """
async def refresh():
    request_urllib("https://example.com")
"""
    assert "async-blocking-http" in _rules(source)


def test_lint_allows_request_urllib_via_to_thread():
    source = """
async def refresh():
    await asyncio.to_thread(request_urllib, "https://example.com")
"""
    assert "async-blocking-http" not in _rules(source)


def test_lint_allows_blocking_http_in_nested_sync_helper():
    source = """
async def refresh():
    def load():
        request_urllib("https://example.com")
        throttle_sleep_sync(1.0)
    await asyncio.to_thread(load)
"""
    rules = _rules(source)
    assert "async-blocking-http" not in rules
    assert "async-time-sleep" not in rules


def test_lint_flags_throttle_sleep_sync_inside_async_def():
    source = """
async def handle():
    throttle_sleep_sync(60.0)
"""
    assert "async-time-sleep" in _rules(source)


def test_lint_flags_run_until_complete():
    source = """
def intercept():
    loop.run_until_complete(coro)
"""
    assert "event-loop-run-until-complete" in _rules(source)


def test_lint_flags_shared_origin_throttle_map():
    source = """
_ORIGIN_BACKOFF = {}

async def wait_http_origin_gate(url, policy):
    return None
"""
    assert "shared-origin-throttle" in _rules(source)


def test_lint_allows_per_request_throttle():
    source = """
class RequestThrottle:
    exponent = 0

async def retry_aiohttp_post(session, url):
    throttle = RequestThrottle()
    await throttle_sleep(throttle.last_delay)
"""
    assert "shared-origin-throttle" not in _rules(source)
    violations = scan_tree(ROOT)
    assert violations == [], "\n".join(
        f"{item.path}:{item.line}: [{item.rule}] {item.message}" for item in violations
    )


def test_violation_is_hashable_dataclass():
    item = Violation(path="x.py", line=1, rule="async-time-sleep", message="nope")
    assert item.rule == "async-time-sleep"


def test_ast_grep_rule_files_exist():
    names = {path.name for path in (ROOT / "lint" / "rules").glob("*.yml")}
    assert "no-retry-ping-fn.yml" in names
    assert "no-retry-ping-fn-kwarg.yml" in names
    assert "no-async-time-sleep.yml" in names
    assert "pinger-sleep-before-ping.yml" in names
    assert "no-async-urlopen.yml" in names
    assert "no-async-request-urllib.yml" in names
    assert "no-async-throttle-sleep-sync.yml" in names
    assert "no-run-until-complete.yml" in names


def test_ci_enforces_net_invariants_ruff_and_coverage():
    text = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    assert "lint/net_invariants.py" in text
    assert "ruff check" in text
    assert "--cov-fail-under=95" in text
    assert "scripts/smoke_net_surfaces.sh" in text
    assert "scripts/codex_exec_smoke.sh" in text


def test_lint_cli_exits_zero_on_this_tree():
    from lint.net_invariants import main

    assert main([]) == 0

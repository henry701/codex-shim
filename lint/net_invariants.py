"""Lint invariants for shim networking: keepalives vs throttle, event-loop sleeps.

Codex Desktop idle-timeouts `stream.next()` / `ws_stream.next()` at 15s. Upstream
429 waits belong in `throttle_sleep`; downstream `{"type":"ping"}` belongs in
`DownstreamPinger` only. Sync `time.sleep` must not run on the aiohttp loop.
"""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass
from pathlib import Path

DESKTOP_WS_IDLE_TIMEOUT_S = 15.0
SCAN_RELATIVE = (
    "codex_shim/net",
    "codex_shim/ws_passthrough.py",
    "codex_shim/chatgpt_edge.py",
    "codex_shim/server.py",
)
FORBIDDEN_RETRY_PARAMS = frozenset({"ping_fn"})
FORBIDDEN_RETRY_KWARGS = frozenset({"ping_fn"})
FORBIDDEN_SHARED_ORIGIN_NAMES = frozenset(
    {
        "_ORIGIN_BACKOFF",
        "wait_http_origin_gate",
        "wait_http_origin_gate_sync",
        "configure_origin_backoff",
        "mark_origin_success",
        "get_origin_backoff",
        "reset_origin_backoff",
        "note_origin_failure",
    }
)


@dataclass(frozen=True)
class Violation:
    path: str
    line: int
    rule: str
    message: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: [{self.rule}] {self.message}"


def scan_tree(root: Path) -> list[Violation]:
    violations: list[Violation] = []
    for relative in SCAN_RELATIVE:
        target = root / relative
        if target.is_dir():
            paths = sorted(target.glob("*.py"))
        elif target.is_file():
            paths = [target]
        else:
            violations.append(
                Violation(
                    path=str(target),
                    line=1,
                    rule="missing-scan-target",
                    message=f"expected networking file {relative}",
                )
            )
            continue
        for path in paths:
            violations.extend(scan_source(path.read_text(encoding="utf-8"), path=str(path)))
    return violations


def scan_source(source: str, path: str = "<mem>") -> list[Violation]:
    tree = ast.parse(source)
    imported_sleep = _imported_time_sleep_names(tree)
    violations: list[Violation] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            violations.extend(_param_violations(node, path))
            if node.name == "throttle_sleep_sync":
                violations.extend(_throttle_sleep_sync_violations(node, path))
            if (
                isinstance(node, ast.AsyncFunctionDef)
                and node.name == "_run"
                and _enclosing_class_name(tree, node) == "DownstreamPinger"
            ):
                violations.extend(_pinger_run_violations(node, path))
            if isinstance(node, ast.AsyncFunctionDef):
                violations.extend(_async_sleep_violations(node, path, imported_sleep))
                violations.extend(_async_blocking_http_violations(node, path))
        elif isinstance(node, ast.Call):
            violations.extend(_kwarg_violations(node, path))
            violations.extend(_run_until_complete_violations(node, path))
        elif isinstance(node, ast.Assign):
            violations.extend(_keepalive_max_violations(node, path))
    violations.extend(_shared_origin_throttle_violations(tree, path))
    if Path(path).name == "server.py":
        violations.extend(_websocket_pinger_violations(tree, path))
    return violations


def _imported_time_sleep_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "time":
                    names.add(alias.asname or "time")
        elif isinstance(node, ast.ImportFrom) and node.module == "time":
            for alias in node.names:
                if alias.name == "sleep":
                    names.add(alias.asname or "sleep")
    return names


def _param_violations(node: ast.FunctionDef | ast.AsyncFunctionDef, path: str) -> list[Violation]:
    args = [
        *node.args.posonlyargs,
        *node.args.args,
        *node.args.kwonlyargs,
    ]
    if node.args.vararg is not None:
        args.append(node.args.vararg)
    if node.args.kwarg is not None:
        args.append(node.args.kwarg)
    violations: list[Violation] = []
    for arg in args:
        if arg.arg in FORBIDDEN_RETRY_PARAMS:
            violations.append(
                Violation(
                    path=path,
                    line=node.lineno,
                    rule="retry-ping-fn-param",
                    message=(
                        f"{node.name} accepts {arg.arg}=; keepalives belong on "
                        "DownstreamPinger, not retry/throttle"
                    ),
                )
            )
    return violations


def _kwarg_violations(node: ast.Call, path: str) -> list[Violation]:
    violations: list[Violation] = []
    for keyword in node.keywords:
        if keyword.arg in FORBIDDEN_RETRY_KWARGS:
            violations.append(
                Violation(
                    path=path,
                    line=node.lineno,
                    rule="retry-ping-fn-kwarg",
                    message=(
                        f"call passes {keyword.arg}=; keepalives belong on "
                        "DownstreamPinger, not retry/throttle"
                    ),
                )
            )
    return violations


def _is_time_sleep_call(node: ast.AST, imported_sleep: set[str]) -> bool:
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if isinstance(func, ast.Attribute) and func.attr == "sleep":
        if isinstance(func.value, ast.Name) and func.value.id in imported_sleep:
            return True
        if isinstance(func.value, ast.Name) and func.value.id in {"time", "_time"}:
            return True
    if isinstance(func, ast.Name) and func.id in imported_sleep and func.id != "time":
        return True
    return False


def _async_sleep_violations(
    node: ast.AsyncFunctionDef,
    path: str,
    imported_sleep: set[str],
) -> list[Violation]:
    violations: list[Violation] = []

    def walk(child: ast.AST, in_nested_sync: bool) -> None:
        if isinstance(child, ast.FunctionDef):
            return
        if isinstance(child, ast.AsyncFunctionDef) and child is not node:
            return
        if not in_nested_sync and isinstance(child, ast.Call) and (
            _is_time_sleep_call(child, imported_sleep)
            or _is_named_call(child, {"throttle_sleep_sync"})
        ):
            violations.append(
                Violation(
                    path=path,
                    line=child.lineno,
                    rule="async-time-sleep",
                    message=(
                        f"time.sleep in async def {node.name} blocks the aiohttp "
                        "event loop (Desktop WS connect timeout is 15s)"
                    ),
                )
            )
        for grandchild in ast.iter_child_nodes(child):
            walk(grandchild, in_nested_sync)

    for stmt in node.body:
        walk(stmt, False)
    return violations


_BLOCKING_SYNC_HTTP_NAMES = frozenset({"urlopen", "request_urllib"})


def _is_named_call(node: ast.AST, names: set[str] | frozenset[str]) -> bool:
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if isinstance(func, ast.Name) and func.id in names:
        return True
    return isinstance(func, ast.Attribute) and func.attr in names


def _is_urlopen_call(node: ast.AST) -> bool:
    return _is_named_call(node, _BLOCKING_SYNC_HTTP_NAMES)


def _async_blocking_http_violations(node: ast.AsyncFunctionDef, path: str) -> list[Violation]:
    violations: list[Violation] = []

    def walk(child: ast.AST) -> None:
        if isinstance(child, ast.FunctionDef):
            return
        if isinstance(child, ast.AsyncFunctionDef) and child is not node:
            return
        if isinstance(child, ast.Call) and _is_urlopen_call(child):
            violations.append(
                Violation(
                    path=path,
                    line=child.lineno,
                    rule="async-blocking-http",
                    message=(
                        f"urlopen/request_urllib in async def {node.name} blocks "
                        "the aiohttp event loop; use asyncio.to_thread"
                    ),
                )
            )
        for grandchild in ast.iter_child_nodes(child):
            walk(grandchild)

    for stmt in node.body:
        walk(stmt)
    return violations


def _run_until_complete_violations(node: ast.Call, path: str) -> list[Violation]:
    func = node.func
    if not isinstance(func, ast.Attribute) or func.attr != "run_until_complete":
        return []
    return [
        Violation(
            path=path,
            line=node.lineno,
            rule="event-loop-run-until-complete",
            message=(
                "loop.run_until_complete on the serve path nests a second loop "
                "(or raises if one is already running); await the coroutine instead"
            ),
        )
    ]


def _throttle_sleep_sync_violations(node: ast.FunctionDef | ast.AsyncFunctionDef, path: str) -> list[Violation]:
    names: set[str] = set()
    has_raise = False
    sleep_line: int | None = None
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            names.add(child.id)
        elif isinstance(child, ast.Attribute):
            names.add(child.attr)
        elif isinstance(child, ast.Raise):
            has_raise = True
        elif isinstance(child, ast.Call) and _is_time_sleep_call(child, {"time", "_time", "sleep"}):
            sleep_line = child.lineno
    if sleep_line is None:
        return []
    guarded = has_raise and (
        "get_running_loop" in names or "_on_running_event_loop" in names
    )
    if guarded:
        return []
    return [
        Violation(
            path=path,
            line=node.lineno,
            rule="throttle-sleep-sync-unguarded",
            message=(
                "throttle_sleep_sync must refuse time.sleep on a running "
                "asyncio event loop"
            ),
        )
    ]


def _enclosing_class_name(tree: ast.AST, target: ast.AST) -> str | None:
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        if any(child is target for child in ast.walk(node)):
            return node.name
    return None


def _first_await(stmts: list[ast.stmt]) -> ast.Await | None:
    for stmt in stmts:
        if isinstance(stmt, ast.Try):
            found = _first_await(stmt.body)
            if found is not None:
                return found
            continue
        if isinstance(stmt, ast.While):
            found = _first_await(stmt.body)
            if found is not None:
                return found
            continue
        if isinstance(stmt, ast.If):
            continue
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Await):
            return stmt.value
        if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Await):
            return stmt.value
    return None


def _await_call_name(await_node: ast.Await) -> str | None:
    value = await_node.value
    if not isinstance(value, ast.Call):
        return None
    func = value.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return None


def _pinger_run_violations(node: ast.AsyncFunctionDef, path: str) -> list[Violation]:
    first = _first_await(node.body)
    name = _await_call_name(first) if first is not None else None
    if name == "sleep":
        return []
    return [
        Violation(
            path=path,
            line=node.lineno,
            rule="pinger-sleep-before-ping",
            message=(
                "DownstreamPinger._run must await asyncio.sleep before the first "
                "ping so keepalives are background-only"
            ),
        )
    ]


def _keepalive_max_violations(node: ast.Assign, path: str) -> list[Violation]:
    names = [target.id for target in node.targets if isinstance(target, ast.Name)]
    if "KEEPALIVE_MAX" not in names:
        return []
    value = node.value
    if isinstance(value, ast.Constant) and isinstance(value.value, (int, float)):
        if float(value.value) < DESKTOP_WS_IDLE_TIMEOUT_S:
            return []
        return [
            Violation(
                path=path,
                line=node.lineno,
                rule="keepalive-max-vs-desktop",
                message=(
                    f"KEEPALIVE_MAX={value.value} must be < {DESKTOP_WS_IDLE_TIMEOUT_S:g}s "
                    "(Codex Desktop WS/SSE idle timeout)"
                ),
            )
        ]
    return [
        Violation(
            path=path,
            line=node.lineno,
            rule="keepalive-max-vs-desktop",
            message="KEEPALIVE_MAX must be a numeric constant below the Desktop idle timeout",
        )
    ]


def _websocket_pinger_violations(tree: ast.AST, path: str) -> list[Violation]:
    for node in ast.walk(tree):
        if not isinstance(node, ast.AsyncFunctionDef) or node.name != "responses_websocket":
            continue
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                func = child.func
                if isinstance(func, ast.Name) and func.id == "DownstreamPinger":
                    return []
        return [
            Violation(
                path=path,
                line=node.lineno,
                rule="ws-handler-missing-pinger",
                message=(
                    "responses_websocket must start DownstreamPinger after "
                    "ws.prepare(); do not ping from retry/throttle"
                ),
            )
        ]
    return [
        Violation(
            path=path,
            line=1,
            rule="ws-handler-missing-pinger",
            message="server.py is missing responses_websocket",
        )
    ]


def _shared_origin_throttle_violations(tree: ast.AST, path: str) -> list[Violation]:
    violations: list[Violation] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_SHARED_ORIGIN_NAMES:
            violations.append(
                Violation(
                    path=path,
                    line=node.lineno,
                    rule="shared-origin-throttle",
                    message=(
                        f"{node.id} is a process-wide origin throttle; backoff must stay "
                        "on the HTTP request or WS connection that hit 429"
                    ),
                )
            )
        elif (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name in FORBIDDEN_SHARED_ORIGIN_NAMES
        ):
            violations.append(
                Violation(
                    path=path,
                    line=node.lineno,
                    rule="shared-origin-throttle",
                    message=(
                        f"{node.name} is a process-wide origin throttle; backoff must stay "
                        "on the HTTP request or WS connection that hit 429"
                    ),
                )
            )
    return violations


def main(argv: list[str] | None = None) -> int:
    del argv
    root = Path(__file__).resolve().parents[1]
    violations = scan_tree(root)
    for item in violations:
        print(item)
    if violations:
        print(f"{len(violations)} net invariant violation(s)", file=sys.stderr)
        return 1
    print("net invariants: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

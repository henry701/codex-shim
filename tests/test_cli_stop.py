from __future__ import annotations

import signal

from codex_shim import cli


def test_stop_escalates_to_sigkill_when_sigterm_times_out(monkeypatch, tmp_path):
    pid_path = tmp_path / "shim.pid"
    pid_path.write_text("4242")
    monkeypatch.setattr(cli, "PID_PATH", pid_path)

    checks = {"count": 0}

    def pid_running(pid: int | None) -> bool:
        assert pid == 4242
        checks["count"] += 1
        return checks["count"] < 4

    kills: list[tuple] = []

    monkeypatch.setattr(cli, "_pid_running", pid_running)
    monkeypatch.setattr(cli, "_terminate_pid", lambda pid: kills.append(("term", pid)))
    monkeypatch.setattr(cli, "_wait_for_port_free", lambda port, timeout_s: True)
    monkeypatch.setattr(
        cli.os,
        "killpg",
        lambda pid, sig: kills.append(("killpg", pid, sig)),
    )
    monkeypatch.setattr(
        cli.os,
        "kill",
        lambda pid, sig: kills.append(("kill", pid, sig)),
    )
    monkeypatch.setattr(cli, "_SHUTDOWN_TERM_WAIT_S", 0.01)
    monkeypatch.setattr(cli, "_SHUTDOWN_KILL_WAIT_S", 0.05)
    monkeypatch.setattr(cli, "_SHUTDOWN_POLL_INTERVAL_S", 0.01)

    assert cli.stop() == 0
    assert ("term", 4242) in kills
    assert ("killpg", 4242, signal.SIGKILL) in kills or ("kill", 4242, signal.SIGKILL) in kills
    assert not pid_path.exists()


def test_stop_returns_error_when_process_survives_sigkill(monkeypatch, tmp_path):
    pid_path = tmp_path / "shim.pid"
    pid_path.write_text("5150")
    monkeypatch.setattr(cli, "PID_PATH", pid_path)
    monkeypatch.setattr(cli, "_pid_running", lambda pid: True)
    monkeypatch.setattr(cli, "_terminate_pid", lambda pid: None)
    monkeypatch.setattr(cli.os, "killpg", lambda pid, sig: None)
    monkeypatch.setattr(cli.os, "kill", lambda pid, sig: None)
    monkeypatch.setattr(cli, "_SHUTDOWN_TERM_WAIT_S", 0.01)
    monkeypatch.setattr(cli, "_SHUTDOWN_KILL_WAIT_S", 0.01)
    monkeypatch.setattr(cli, "_SHUTDOWN_POLL_INTERVAL_S", 0.01)

    assert cli.stop() == 1
    assert pid_path.exists()


def test_status_reports_healthy_foreground_service_without_pid(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, "PID_PATH", tmp_path / "missing.pid")
    monkeypatch.setattr(cli, "_health", lambda port: {"models": 12})

    assert cli.status(8765) == 0
    out = capsys.readouterr().out
    assert "Shim is running on http://127.0.0.1:8765 (12 models)." in out

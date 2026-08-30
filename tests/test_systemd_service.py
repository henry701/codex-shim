from __future__ import annotations

from pathlib import Path

from codex_shim import cli
from codex_shim import server as shim_server


def test_systemd_unit_runs_sync_desktop_then_foreground(monkeypatch, tmp_path):
    load_env = tmp_path / "load-env.sh"
    load_env.write_text("")
    monkeypatch.setattr(cli, "LOAD_ENV_SCRIPT", load_env)
    unit = cli._systemd_unit_content(
        codex_shim_bin="/usr/bin/codex-shim",
        settings_path=Path("/home/user/.codex-shim/models.json"),
        port=8765,
    )
    pre = next(line for line in unit.splitlines() if line.startswith("ExecStartPre="))
    start = next(line for line in unit.splitlines() if line.startswith("ExecStart="))
    assert "sync-desktop" in pre
    assert " serve" in start
    assert " run" not in start
    assert f'source "{load_env}"' in unit
    assert "WantedBy=graphical-session.target" in unit
    assert "network-online.target" not in unit
    assert "network-ready-user.service" not in unit
    assert "StandardOutput=append:" in unit
    assert "StandardError=append:" in unit
    assert str(cli.SERVICE_LOG_PATH) in unit
    timeout = next(line for line in unit.splitlines() if line.startswith("TimeoutStartSec="))
    assert timeout == f"TimeoutStartSec={cli.SYSTEMD_TIMEOUT_START_SEC}"
    assert cli.SYSTEMD_TIMEOUT_START_SEC >= 180


def test_startup_refresh_timeout_is_bounded_so_serve_can_bind():
    assert shim_server._STARTUP_REFRESH_TIMEOUT_SEC <= 15
    assert shim_server._CATALOG_REFRESH_INITIAL_DELAY_DEFAULT_SEC == 15.0


def _completed(returncode=0, stdout="", stderr=""):
    return type("R", (), {"returncode": returncode, "stdout": stdout, "stderr": stderr})()


def test_stop_stops_systemd_unit_when_active(monkeypatch, tmp_path):
    unit = tmp_path / "codex-shim.service"
    unit.write_text("[Service]\n")
    monkeypatch.setattr(cli, "SYSTEMD_USER_UNIT", unit)
    monkeypatch.setattr(cli, "PID_PATH", tmp_path / "shim.pid")
    monkeypatch.setattr(cli, "_read_pid", lambda: None)
    monkeypatch.setattr(cli, "_pid_running", lambda _pid: False)
    monkeypatch.setattr(cli, "_health", lambda _port: None)
    monkeypatch.setattr(cli.shutil, "which", lambda name: "/usr/bin/systemctl" if name == "systemctl" else None)

    calls: list[list[str]] = []

    def fake_run(cmd, check=False, capture_output=False, text=False):
        del check, capture_output, text
        calls.append(list(cmd))
        if cmd[:4] == ["systemctl", "--user", "show", "-p"]:
            return _completed(stdout="active\n")
        return _completed()

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    monkeypatch.setattr(cli, "_wait_for_port_free", lambda *_args: True)
    assert cli.stop() == 0
    assert ["systemctl", "--user", "stop", "codex-shim.service"] in calls


def test_restart_restarts_systemd_unit_when_installed(monkeypatch, tmp_path):
    unit = tmp_path / "codex-shim.service"
    unit.write_text("[Service]\n")
    monkeypatch.setattr(cli, "SYSTEMD_USER_UNIT", unit)
    monkeypatch.setattr(cli.shutil, "which", lambda name: "/usr/bin/systemctl" if name == "systemctl" else "/usr/bin/codex-shim")

    calls: list[list[str]] = []

    def fake_run(cmd, check=False, capture_output=False, text=False):
        del check, capture_output, text
        calls.append(list(cmd))
        return _completed()

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    monkeypatch.setattr(cli, "_healthy", lambda _port: True)
    monkeypatch.setattr(
        cli,
        "start",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("pid-file start must not run")),
    )
    monkeypatch.setattr(
        cli,
        "stop",
        lambda: (_ for _ in ()).throw(AssertionError("pid-file stop must not run")),
    )
    assert cli.restart(tmp_path / "models.json", 8765) == 0
    assert ["systemctl", "--user", "daemon-reload"] in calls
    assert ["systemctl", "--user", "restart", "codex-shim.service"] in calls


def test_restart_on_alternate_port_does_not_bounce_systemd(monkeypatch, tmp_path):
    unit = tmp_path / "codex-shim.service"
    unit.write_text("[Service]\n")
    monkeypatch.setattr(cli, "SYSTEMD_USER_UNIT", unit)
    monkeypatch.setattr(cli.shutil, "which", lambda name: "/usr/bin/systemctl" if name == "systemctl" else "/usr/bin/codex-shim")

    calls: list[list[str]] = []

    def fake_run(cmd, check=False, capture_output=False, text=False):
        del check, capture_output, text
        calls.append(list(cmd))
        return _completed()

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    monkeypatch.setattr(cli, "_listener_pid", lambda _port: None)

    def fail_stop() -> int:
        raise AssertionError("systemd/default stop must not run for 8766")

    def fail_generate(*_args, **_kwargs) -> None:
        raise AssertionError("generate must not rewrite Desktop catalog for 8766")

    started: list[int] = []

    def fake_start(_settings, port: int) -> int:
        started.append(port)
        return 0

    monkeypatch.setattr(cli, "stop", fail_stop)
    monkeypatch.setattr(cli, "generate", fail_generate)
    monkeypatch.setattr(cli, "start", fake_start)
    assert cli.restart(tmp_path / "models.json", 8766) == 0
    assert started == [8766]
    assert not any(cmd[:3] == ["systemctl", "--user", "restart"] for cmd in calls)


def test_run_foreground_syncs_then_serves(monkeypatch, tmp_path):
    calls: list[str] = []
    monkeypatch.setattr(cli, "sync_desktop", lambda *_args, **_kwargs: calls.append("sync") or 0)
    monkeypatch.setattr(cli, "serve_foreground", lambda *_args, **_kwargs: calls.append("serve") or 0)
    assert cli.run_foreground(tmp_path / "models.json", 8765) == 0
    assert calls == ["sync", "serve"]


def test_logrotate_conf_rotates_by_size_with_copytruncate():
    conf = cli._logrotate_conf_content(log_path=Path("/home/user/.codex-shim/shim.log"))
    assert "/home/user/.codex-shim/shim.log" in conf
    assert "size 30M" in conf
    assert "rotate 10" in conf
    assert "compress" in conf
    assert "copytruncate" in conf


def test_logrotate_timer_is_hourly():
    timer = cli._logrotate_timer_unit_content()
    assert "OnCalendar=hourly" in timer
    assert "WantedBy=timers.target" in timer


def test_install_logrotate_writes_units(monkeypatch, tmp_path):
    conf = tmp_path / "logrotate.d" / "codex-shim"
    state_dir = tmp_path / "state"
    service = tmp_path / "systemd" / "codex-shim-logrotate.service"
    timer = tmp_path / "systemd" / "codex-shim-logrotate.timer"
    log = tmp_path / "shim.log"
    log.write_text("x" * 10)

    monkeypatch.setattr(cli, "LOGROTATE_CONF_PATH", conf)
    monkeypatch.setattr(cli, "LOGROTATE_STATE_DIR", state_dir)
    monkeypatch.setattr(cli, "LOGROTATE_STATE_PATH", state_dir / "logrotate.status")
    monkeypatch.setattr(cli, "LOGROTATE_SERVICE_UNIT", service)
    monkeypatch.setattr(cli, "LOGROTATE_TIMER_UNIT", timer)
    monkeypatch.setattr(cli, "SERVICE_LOG_PATH", log)
    monkeypatch.setattr(cli.shutil, "which", lambda name: "/usr/bin/logrotate" if name == "logrotate" else None)

    calls: list[list[str]] = []

    def fake_run(cmd, check=False, capture_output=False, text=False):
        calls.append(list(cmd))
        return type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    assert cli.install_logrotate(force_rotate=False) == 0
    assert conf.is_file()
    assert service.is_file()
    assert timer.is_file()
    assert any(cmd[:3] == ["systemctl", "--user", "enable"] for cmd in calls)
    # Under-size log should not force-rotate.
    assert not any(cmd and cmd[0] == "logrotate" for cmd in calls)


def test_install_logrotate_force_rotates_oversized(monkeypatch, tmp_path):
    conf = tmp_path / "logrotate.d" / "codex-shim"
    state_dir = tmp_path / "state"
    service = tmp_path / "systemd" / "codex-shim-logrotate.service"
    timer = tmp_path / "systemd" / "codex-shim-logrotate.timer"
    log = tmp_path / "shim.log"
    log.write_bytes(b"x" * (30 * 1024 * 1024 + 1))

    monkeypatch.setattr(cli, "LOGROTATE_CONF_PATH", conf)
    monkeypatch.setattr(cli, "LOGROTATE_STATE_DIR", state_dir)
    monkeypatch.setattr(cli, "LOGROTATE_STATE_PATH", state_dir / "logrotate.status")
    monkeypatch.setattr(cli, "LOGROTATE_SERVICE_UNIT", service)
    monkeypatch.setattr(cli, "LOGROTATE_TIMER_UNIT", timer)
    monkeypatch.setattr(cli, "SERVICE_LOG_PATH", log)
    monkeypatch.setattr(cli.shutil, "which", lambda name: "/usr/bin/logrotate" if name == "logrotate" else None)

    calls: list[list[str]] = []

    def fake_run(cmd, check=False, capture_output=False, text=False):
        calls.append(list(cmd))
        return type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    assert cli.install_logrotate(force_rotate=False) == 0
    assert any(cmd and cmd[0] == "logrotate" and "-f" in cmd for cmd in calls)

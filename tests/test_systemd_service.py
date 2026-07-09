from __future__ import annotations

from pathlib import Path

from codex_shim import cli


def test_systemd_unit_runs_sync_desktop_then_foreground(monkeypatch, tmp_path):
    load_env = tmp_path / "load-env.sh"
    load_env.write_text("")
    monkeypatch.setattr(cli, "LOAD_ENV_SCRIPT", load_env)
    unit = cli._systemd_unit_content(
        codex_shim_bin="/usr/bin/codex-shim",
        settings_path=Path("/home/user/.codex-shim/models.json"),
        port=8765,
    )
    assert "sync-desktop" in unit
    assert f'source "{load_env}"' in unit
    assert "ExecStart=" in unit
    assert "WantedBy=graphical-session.target" in unit
    assert "network-online.target" not in unit
    assert "network-ready-user.service" not in unit
    assert "StandardOutput=append:" in unit
    assert "StandardError=append:" in unit
    assert str(cli.SERVICE_LOG_PATH) in unit


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

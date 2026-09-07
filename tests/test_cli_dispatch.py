from __future__ import annotations

import signal
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest

from codex_shim import cli


def test_main_dispatches_lifecycle_commands(monkeypatch, tmp_path):
    calls: list[tuple] = []
    monkeypatch.setattr(cli, "generate", lambda *args: calls.append(("generate", args)))
    monkeypatch.setattr(cli, "start", lambda *args: calls.append(("start", args)) or 0)
    monkeypatch.setattr(cli, "stop", lambda: calls.append(("stop",)) or 0)
    monkeypatch.setattr(cli, "restart", lambda *args: calls.append(("restart", args)) or 0)
    monkeypatch.setattr(cli, "restore_codex_config", lambda: calls.append(("restore",)))
    monkeypatch.setattr(cli, "install_codex_config", lambda *args: calls.append(("install", args)))
    monkeypatch.setattr(cli, "serve_foreground", lambda *args: calls.append(("serve", args)) or 0)
    monkeypatch.setattr(cli, "run_foreground", lambda *args: calls.append(("run", args)) or 0)
    monkeypatch.setattr(cli, "sync_desktop", lambda *args: calls.append(("sync", args)) or 0)
    monkeypatch.setattr(cli, "install_service", lambda *args: calls.append(("service", args)) or 0)
    monkeypatch.setattr(cli, "install_logrotate", lambda **kwargs: calls.append(("logrotate", kwargs)) or 0)
    monkeypatch.setattr(cli, "status", lambda port: calls.append(("status", port)) or 0)
    monkeypatch.setattr(cli, "discover_models", lambda *args, **kwargs: calls.append(("discover", kwargs)) or 0)
    monkeypatch.setattr(cli, "list_models", lambda path: calls.append(("list", path)) or 0)
    monkeypatch.setattr(cli, "patch_codex_app", lambda: calls.append(("patch",)) or 0)
    monkeypatch.setattr(cli, "restore_codex_app_bundle", lambda: calls.append(("restore-app",)) or 0)
    monkeypatch.setattr(cli, "migrate_threads_command", lambda **kwargs: calls.append(("migrate", kwargs)) or 0)
    monkeypatch.setattr(cli, "refresh_opencode_go", lambda *args: calls.append(("opencode", args)) or 0)
    monkeypatch.setattr(cli, "ensure_started", lambda *args: calls.append(("ensure", args)))
    monkeypatch.setattr(cli, "exec_codex", lambda *args: calls.append(("exec", args)))
    monkeypatch.setattr(cli, "exec_codex_app", lambda *args: calls.append(("app", args)))

    quota: dict[str, list[str]] = {}

    def fake_quota(argv):
        quota["argv"] = list(argv)
        return 0

    monkeypatch.setattr("codex_shim.quota_dashboard.main", fake_quota)

    settings = tmp_path / "models.json"
    port_args = ["--settings", str(settings), "--port", "8767"]

    assert cli.main([*port_args, "generate"]) == 0
    assert cli.main([*port_args, "discover", "--refresh"]) == 0
    assert cli.main([*port_args, "list"]) == 0
    assert cli.main([*port_args, "start"]) == 0
    assert cli.main([*port_args, "enable"]) == 0
    assert cli.main([*port_args, "stop"]) == 0
    assert cli.main([*port_args, "disable"]) == 0
    assert cli.main([*port_args, "restart"]) == 0
    assert cli.main([*port_args, "serve"]) == 0
    assert cli.main([*port_args, "run"]) == 0
    assert cli.main([*port_args, "sync-desktop"]) == 0
    assert cli.main([*port_args, "install-service"]) == 0
    assert cli.main([*port_args, "install-logrotate", "--force"]) == 0
    assert cli.main([*port_args, "status"]) == 0
    assert cli.main([*port_args, "patch-app"]) == 0
    assert cli.main([*port_args, "restore-app"]) == 0
    assert cli.main([*port_args, "migrate-threads", "--dry-run"]) == 0
    assert cli.main([*port_args, "quota-report", "--json", "--since", "2026-01-01", "--until", "2026-02-01", "--log-dir", str(tmp_path)]) == 0
    assert cli.main([*port_args, "opencode-go", "refresh"]) == 0
    assert cli.main([*port_args, "model", "list"]) == 0
    assert cli.main([*port_args, "model", "use", "local-llama"]) == 0
    assert cli.main([*port_args, "codex", "--", "exec", "hi"]) == 0
    assert cli.main([*port_args, "app", "-m", "local-llama", "."]) == 0

    assert ("generate", (settings, 8767)) in calls
    assert ("start", (settings, 8767)) in calls
    assert ("restore",) in calls
    assert ("install", (settings, 8767)) in calls
    assert ("logrotate", {"force_rotate": True}) in calls
    assert ("migrate", {"dry_run": True}) in calls
    assert quota["argv"] == ["--log-dir", str(tmp_path), "--since", "2026-01-01", "--until", "2026-02-01", "--json"]
    assert any(item[0] == "exec" for item in calls)
    assert any(item[0] == "app" for item in calls)


def test_start_skips_when_pid_already_running(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_read_pid", lambda: 4242)
    monkeypatch.setattr(cli, "_pid_running", lambda pid: pid == 4242)
    assert cli.start(Path("unused"), 8767) == 0
    assert "already running" in capsys.readouterr().out


def test_start_reports_health_then_exit_then_timeout(monkeypatch, tmp_path, capsys):
    runtime = tmp_path / "runtime"
    monkeypatch.setattr(cli, "RUNTIME_DIR", runtime)
    monkeypatch.setattr(cli, "PID_PATH", runtime / "shim.pid")
    monkeypatch.setattr(cli, "LOG_PATH", runtime / "shim.log")
    monkeypatch.setattr(cli, "_pid_running", lambda pid: False)
    monkeypatch.setattr(cli, "_STARTUP_HEALTH_WAIT_S", 0.05)
    monkeypatch.setattr(cli, "_STARTUP_POLL_INTERVAL_S", 0.0)

    class HealthyProc:
        pid = 99

        def poll(self):
            return None

    monkeypatch.setattr(cli, "_popen_daemon", lambda *args, **kwargs: HealthyProc())
    monkeypatch.setattr(cli, "_healthy", lambda port: True)
    assert cli.start(tmp_path / "models.json", 8767) == 0
    assert (runtime / "shim.pid").read_text() == "99"
    assert "Shim started" in capsys.readouterr().out

    class DeadProc:
        pid = 100

        def poll(self):
            return 1

    monkeypatch.setattr(cli, "_popen_daemon", lambda *args, **kwargs: DeadProc())
    monkeypatch.setattr(cli, "_healthy", lambda port: False)
    assert cli.start(tmp_path / "models.json", 8767) == 1
    assert "exited during startup" in capsys.readouterr().err

    class HangProc:
        pid = 101

        def poll(self):
            return None

    monkeypatch.setattr(cli, "_popen_daemon", lambda *args, **kwargs: HangProc())
    monkeypatch.setattr(cli, "_healthy", lambda port: False)
    monkeypatch.setattr(cli, "_STARTUP_HEALTH_WAIT_S", 0.0)
    assert cli.start(tmp_path / "models.json", 8767) == 1
    assert "health check timed out" in capsys.readouterr().err


def test_exec_codex_strips_double_dash_and_sets_no_proxy(monkeypatch, tmp_path):
    captured: dict[str, object] = {}
    monkeypatch.setattr(cli, "_override_args", lambda *args: ["-c", "foo=bar"])
    monkeypatch.setattr(cli.os, "name", "posix")

    def fake_execvpe(cmd, args, env):
        captured["cmd"] = cmd
        captured["args"] = list(args)
        captured["env"] = env

    monkeypatch.setattr(cli.os, "execvpe", fake_execvpe)
    cli.exec_codex(tmp_path / "models.json", 8767, ["--", "exec", "hi"])
    assert captured["cmd"] == "codex"
    assert captured["args"] == ["codex", "-c", "foo=bar", "exec", "hi"]
    assert "127.0.0.1" in captured["env"]["NO_PROXY"]
    assert "localhost" in captured["env"]["no_proxy"]


def test_exec_codex_app_uses_open_when_patched_bundle_exists(monkeypatch):
    launched: list[list[str]] = []
    monkeypatch.setattr(cli, "_quit_codex_app", lambda: None)
    monkeypatch.setattr(cli, "_foreground_codex_app", lambda: None)
    monkeypatch.setattr(cli, "patched_codex_app_bundle", lambda: Path("/tmp/Codex.app"))
    monkeypatch.setattr(
        cli.subprocess,
        "Popen",
        lambda args, env=None: launched.append(list(args)),
    )
    cli.exec_codex_app(Path("unused"), 8767, ".")
    assert launched == [["open", "-a", "/tmp/Codex.app"]]

    launched.clear()
    monkeypatch.setattr(cli, "patched_codex_app_bundle", lambda: None)
    cli.exec_codex_app(Path("unused"), 8767, "/tmp/project")
    assert launched == [["codex", "app", "/tmp/project"]]


def test_migrate_threads_command_prints_database_counts(capsys, monkeypatch):
    monkeypatch.setattr(cli, "migrate_thread_providers", lambda dry_run=False: {"updated": 0, "databases": {}})
    assert cli.migrate_threads_command() == 0
    assert "No legacy" in capsys.readouterr().out

    monkeypatch.setattr(
        cli,
        "migrate_thread_providers",
        lambda dry_run=False: {"updated": 3, "databases": {"/tmp/a.sqlite": 2, "/tmp/b.sqlite": 1}},
    )
    assert cli.migrate_threads_command(dry_run=True) == 0
    out = capsys.readouterr().out
    assert "Would migrate 2 thread(s) in /tmp/a.sqlite" in out
    assert "Would migrate 3 thread(s) total." in out


def test_load_settings_and_models_error_paths(tmp_path):
    missing = tmp_path / "nope.json"
    assert cli._load_settings_data(missing) is None
    (tmp_path / "bad.json").write_text("{")
    assert cli._load_settings_data(tmp_path / "bad.json") is None
    (tmp_path / "list.json").write_text("[]")
    assert cli._load_settings_data(tmp_path / "list.json") is None
    (tmp_path / "ok.json").write_text('{"models": []}')
    assert cli._load_settings_data(tmp_path / "ok.json") == {"models": []}

    with pytest.raises(SystemExit, match="Settings file not found"):
        cli._load_models(missing)
    with pytest.raises(SystemExit, match="not valid JSON"):
        cli._load_models(tmp_path / "bad.json")


def test_load_models_with_budget_timeout_and_error(monkeypatch, tmp_path):
    class AliveThread:
        def __init__(self, target, daemon=True):
            del target, daemon

        def start(self):
            return None

        def join(self, timeout):
            del timeout

        def is_alive(self):
            return True

    monkeypatch.setattr(cli.threading, "Thread", AliveThread)
    with pytest.raises(TimeoutError, match="model discovery exceeded"):
        cli._load_models_with_budget(tmp_path / "models.json", 0.01)

    class ErrorThread:
        def __init__(self, target, daemon=True):
            self._target = target

        def start(self):
            self._target()

        def join(self, timeout):
            del timeout

        def is_alive(self):
            return False

    monkeypatch.setattr(cli.threading, "Thread", ErrorThread)
    monkeypatch.setattr(cli, "_load_models", lambda path: (_ for _ in ()).throw(ValueError("boom")))
    with pytest.raises(ValueError, match="boom"):
        cli._load_models_with_budget(tmp_path / "models.json", 1.0)


def test_health_payload_and_model_count(monkeypatch):
    assert cli._health_model_count(12) == 12
    assert cli._health_model_count(["a", "b"]) == 2
    assert cli._health_model_count("nope") == 0

    class Ok:
        status = 200

        def read(self):
            return b'{"ok": true, "models": 4}'

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(cli, "urlopen", lambda *args, **kwargs: Ok())
    assert cli._health(8767)["models"] == 4
    assert cli._healthy(8767) is True

    class BadStatus:
        status = 500

        def read(self):
            return b"{}"

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(cli, "urlopen", lambda *args, **kwargs: BadStatus())
    assert cli._health(8767) is None

    class NotOk:
        status = 200

        def read(self):
            return b'{"ok": false}'

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(cli, "urlopen", lambda *args, **kwargs: NotOk())
    assert cli._health(8767) is None

    def boom(*args, **kwargs):
        raise OSError("down")

    monkeypatch.setattr(cli, "urlopen", boom)
    assert cli._health(8767) is None


def test_systemd_main_pid_and_terminate_fallback(monkeypatch, tmp_path):
    monkeypatch.setattr(cli.os, "name", "posix")
    monkeypatch.setattr(cli.shutil, "which", lambda name: "/usr/bin/systemctl" if name == "systemctl" else None)
    unit = tmp_path / "codex-shim.service"
    unit.write_text("[Service]\n")
    monkeypatch.setattr(cli, "SYSTEMD_USER_UNIT", unit)

    monkeypatch.setattr(
        cli.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout=""),
    )
    assert cli._systemd_main_pid() is None

    monkeypatch.setattr(
        cli.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="abc\n"),
    )
    assert cli._systemd_main_pid() is None

    monkeypatch.setattr(
        cli.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="0\n"),
    )
    assert cli._systemd_main_pid() is None

    monkeypatch.setattr(
        cli.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="4321\n"),
    )
    assert cli._systemd_main_pid() == 4321

    kills: list[tuple] = []

    def killpg(pid, sig):
        raise OSError("no process group")

    monkeypatch.setattr(cli.os, "killpg", killpg)
    monkeypatch.setattr(cli.os, "kill", lambda pid, sig: kills.append((pid, sig)))
    cli._terminate_pid(77)
    assert kills == [(77, signal.SIGTERM)]


def test_asar_helpers_and_lossy_text(tmp_path):
    payload = b'{"files":{}}'
    asar = tmp_path / "app.asar"
    asar.write_bytes(struct.pack("<4I", 4, 0, 0, len(payload)) + payload)
    assert len(cli._app_asar_hash(asar)) == 64
    assert cli._app_asar_header_hash(asar) == __import__("hashlib").sha256(payload).hexdigest()
    assert cli._path_is_writable(asar) is True
    assert cli._path_is_writable(tmp_path / "missing") is False
    assert cli._app_asar_is_patched(asar) is False
    asar.write_bytes(b"let x=!1; foo.forEach recentConversationSortKey,modelProviders:[],archived:!1,sourceKinds:k")
    # Applied regexes need both picker and sidebar markers; this is only a negative/positive probe.
    assert cli._has_command("python3") is True
    assert cli._has_command("definitely-not-a-binary-xyz") is False
    weird = tmp_path / "lossy.txt"
    weird.write_bytes(b"ok\xffmore")
    assert "ok" in cli._read_text_lossy(weird)
    assert cli._env_flag("UNSET_FLAG_FOR_TEST") is False
    assert cli._bool_text(0) == "false"
    assert cli._bool_text("yes") == "true"


def test_with_loopback_no_proxy_adds_all_hosts():
    env = cli._with_loopback_no_proxy({"NO_PROXY": "example.com"})
    for host in ("127.0.0.1", "localhost", "::1"):
        assert host in env["NO_PROXY"]
        assert host in env["no_proxy"]


def test_current_managed_model_reads_block(tmp_path, monkeypatch):
    config = tmp_path / "config.toml"
    monkeypatch.setattr(cli, "CODEX_CONFIG_PATH", config)
    assert cli._current_managed_model() is None
    config.write_text(
        "\n".join(
            [
                cli.MANAGED_BEGIN,
                'model = "local-llama"',
                cli.MANAGED_END,
            ]
        )
        + "\n"
    )
    assert cli._current_managed_model() == "local-llama"


def test_doctor_passthrough_env_flags(monkeypatch):
    monkeypatch.setenv("CODEX_SHIM_DISABLE_CHATGPT", "1")
    checks = cli._doctor_chatgpt()
    assert checks[0].status == "INFO"
    monkeypatch.setenv("CODEX_SHIM_DISABLE_CURSOR", "true")
    checks = cli._doctor_cursor()
    assert checks[0].status == "INFO"


def test_ensure_started_raises_when_start_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "_read_pid", lambda: None)
    monkeypatch.setattr(cli, "_pid_running", lambda pid: False)
    monkeypatch.setattr(cli, "_healthy", lambda port: False)
    monkeypatch.setattr(cli, "start", lambda *args: 1)
    with pytest.raises(SystemExit) as exc:
        cli.ensure_started(tmp_path / "models.json", 8767)
    assert exc.value.code == 1


def test_serve_and_run_foreground_call_server(monkeypatch, tmp_path):
    captured: dict[str, object] = {}
    monkeypatch.setattr("codex_shim.nous_auth.refresh_nous_oauth_on_startup", lambda: captured.setdefault("nous", True))
    monkeypatch.setattr("codex_shim.server.main", lambda argv: captured.setdefault("argv", argv))
    monkeypatch.setattr(cli, "sync_desktop", lambda *args, **kwargs: captured.setdefault("sync", args) or 0)
    assert cli.serve_foreground(tmp_path / "models.json", 8767) == 0
    assert captured["nous"] is True
    assert "--port" in captured["argv"]
    assert cli.run_foreground(tmp_path / "models.json", 8767) == 0
    assert captured["sync"][0] == tmp_path / "models.json"


def _completed(returncode=0, stdout="", stderr=""):
    return type("R", (), {"returncode": returncode, "stdout": stdout, "stderr": stderr})()


def test_install_service_writes_unit_and_warns_when_logrotate_fails(monkeypatch, tmp_path, capsys):
    unit = tmp_path / "systemd" / "codex-shim.service"
    dropin = tmp_path / "systemd" / "codex-shim.service.d" / "network-ready.conf"
    ready = tmp_path / "systemd" / "network-ready-user.service"
    ready.parent.mkdir(parents=True)
    ready.write_text("[Unit]\n")
    monkeypatch.setattr(cli, "SYSTEMD_USER_UNIT", unit)
    monkeypatch.setattr(cli, "SYSTEMD_NETWORK_READY_DROPIN", dropin)
    monkeypatch.setattr(cli, "NETWORK_READY_USER_UNIT", ready)
    monkeypatch.setattr(cli.os, "name", "posix")
    monkeypatch.setattr(cli.shutil, "which", lambda name: "/usr/bin/codex-shim" if name == "codex-shim" else None)
    monkeypatch.setattr(cli, "stop", lambda: 0)
    monkeypatch.setattr(cli, "install_logrotate", lambda **kwargs: 1)
    calls: list[list[str]] = []

    def fake_run(cmd, check=False, capture_output=False, text=False):
        del check, capture_output, text
        calls.append(list(cmd))
        if cmd[:1] == ["loginctl"]:
            return _completed(1, stderr="no linger")
        return _completed()

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    assert cli.install_service(tmp_path / "models.json", 8767) == 0
    assert unit.is_file()
    assert "ExecStart=" in unit.read_text()
    assert dropin.is_file()
    err = capsys.readouterr().err
    assert "enable-linger" in err
    assert "logrotate install failed" in err
    assert ["systemctl", "--user", "enable", "--now", "codex-shim.service"] in calls


def test_install_service_missing_binary_and_systemctl_failure(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli.os, "name", "posix")
    monkeypatch.setattr(cli.shutil, "which", lambda name: None)
    assert cli.install_service(tmp_path / "models.json", 8765) == 1
    assert "not found on PATH" in capsys.readouterr().err

    unit = tmp_path / "codex-shim.service"
    monkeypatch.setattr(cli, "SYSTEMD_USER_UNIT", unit)
    monkeypatch.setattr(cli.shutil, "which", lambda name: "/usr/bin/codex-shim")
    monkeypatch.setattr(cli, "stop", lambda: 0)
    monkeypatch.setattr(cli, "_ensure_network_ready_dropin", lambda: None)
    monkeypatch.setattr(
        cli.subprocess,
        "run",
        lambda cmd, check=False, capture_output=False, text=False: _completed(3),
    )
    assert cli.install_service(tmp_path / "models.json", 8765) == 3


def test_install_logrotate_missing_binary_and_force_rotate(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli.os, "name", "posix")
    monkeypatch.setattr(cli.shutil, "which", lambda name: None)
    assert cli.install_logrotate() == 1
    assert "logrotate not found" in capsys.readouterr().err

    conf = tmp_path / "logrotate.d" / "codex-shim"
    state_dir = tmp_path / "state"
    service = tmp_path / "systemd" / "codex-shim-logrotate.service"
    timer = tmp_path / "systemd" / "codex-shim-logrotate.timer"
    log = tmp_path / "shim.log"
    monkeypatch.setattr(cli, "LOGROTATE_CONF_PATH", conf)
    monkeypatch.setattr(cli, "LOGROTATE_STATE_DIR", state_dir)
    monkeypatch.setattr(cli, "LOGROTATE_STATE_PATH", state_dir / "status")
    monkeypatch.setattr(cli, "LOGROTATE_SERVICE_UNIT", service)
    monkeypatch.setattr(cli, "LOGROTATE_TIMER_UNIT", timer)
    monkeypatch.setattr(cli, "SERVICE_LOG_PATH", log)
    monkeypatch.setattr(cli.shutil, "which", lambda name: "/usr/bin/logrotate")
    calls: list[list[str]] = []

    def fake_run(cmd, check=False, capture_output=False, text=False):
        del check
        calls.append(list(cmd))
        if cmd[:1] == ["logrotate"]:
            return _completed(0, stdout="ok")
        return _completed()

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    log.write_text("tiny")
    assert cli.install_logrotate(force_rotate=True) == 0
    assert conf.is_file()
    assert timer.is_file()
    assert any(cmd[:1] == ["logrotate"] for cmd in calls)
    assert "Rotated" in capsys.readouterr().out


def test_restart_systemd_times_out_without_health(monkeypatch, capsys):
    monkeypatch.setattr(cli, "SYSTEMD_HEALTH_WAIT_S", 0.0)
    monkeypatch.setattr(cli, "_STARTUP_POLL_INTERVAL_S", 0.0)
    monkeypatch.setattr(cli, "_healthy", lambda port: False)
    monkeypatch.setattr(cli.subprocess, "run", lambda *args, **kwargs: _completed())
    assert cli._restart_systemd_service(8765) == 1
    assert "health check timed out" in capsys.readouterr().err


def test_restart_aborts_when_stop_fails(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli.os, "name", "posix")
    monkeypatch.setattr(cli, "SYSTEMD_USER_UNIT", tmp_path / "missing.service")
    monkeypatch.setattr(cli.shutil, "which", lambda name: None)
    monkeypatch.setattr(cli, "stop", lambda: 1)
    assert cli.restart(tmp_path / "models.json", cli.DEFAULT_PORT) == 1
    assert "could not stop" in capsys.readouterr().err


def test_stop_reports_undetermined_listener(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, "PID_PATH", tmp_path / "shim.pid")
    monkeypatch.setattr(cli, "_stop_systemd_unit_if_active", lambda: False)
    monkeypatch.setattr(cli, "_read_pid", lambda: None)
    monkeypatch.setattr(cli, "_pid_running", lambda pid: False)
    monkeypatch.setattr(cli, "_health", lambda port: {"ok": True})
    monkeypatch.setattr(cli, "_listener_pid", lambda port: None)
    assert cli.stop() == 1
    assert "could not be determined" in capsys.readouterr().err

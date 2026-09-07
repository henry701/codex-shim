from __future__ import annotations

import plistlib
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest

from codex_shim import cli


def test_load_explicit_returns_empty_when_file_missing(tmp_path):
    from codex_shim.settings import ModelSettings, ShimModel

    missing = tmp_path / "no-such-models.json"
    assert ModelSettings(missing).load_explicit() == []
    assert ShimModel(
        slug="a",
        model="a",
        display_name="A",
        provider="anthropic",
        base_url="http://x",
    ).is_anthropic
    assert ShimModel(
        slug="b",
        model="b",
        display_name="B",
        provider="generic-chat-completion-api",
        base_url="http://x",
    ).is_openai_chat
    assert ShimModel(
        slug="c",
        model="c",
        display_name="C",
        provider="openai-responses",
        base_url="http://x",
    ).is_openai_responses


def test_popen_daemon_posix_and_windows(monkeypatch):
    captured: list[dict] = []

    def fake_popen(cmd, **kwargs):
        del cmd
        captured.append(kwargs)
        return SimpleNamespace(pid=1)

    monkeypatch.setattr(cli.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(cli.os, "name", "posix")
    cli._popen_daemon(["python"], log=None, env={})
    assert captured[-1]["start_new_session"] is True

    monkeypatch.setattr(cli.os, "name", "nt")
    monkeypatch.setattr(cli.subprocess, "CREATE_NEW_PROCESS_GROUP", 0x200, raising=False)
    monkeypatch.setattr(cli.subprocess, "DETACHED_PROCESS", 0x8, raising=False)
    cli._popen_daemon(["python"], log=None, env={})
    assert captured[-1]["creationflags"] == 0x200 | 0x8


def test_pid_running_and_terminate_windows(monkeypatch):
    class Kernel:
        def __init__(self) -> None:
            self.closed = 0
            self.terminated = 0

        def OpenProcess(self, access, inherit, pid):
            del inherit
            if pid == 1:
                return None
            return 0xBEEF if access else None

        def TerminateProcess(self, handle, code):
            del handle, code
            self.terminated += 1
            return True

        def CloseHandle(self, handle):
            del handle
            self.closed += 1
            return True

        def GetExitCodeProcess(self, handle, exit_code):
            del handle
            exit_code._obj.value = cli.WINDOWS_STILL_ACTIVE
            return True

    kernel = Kernel()
    monkeypatch.setattr(cli.ctypes, "windll", SimpleNamespace(kernel32=kernel), raising=False)
    monkeypatch.setattr(cli.os, "name", "nt")
    assert cli._pid_running(None) is False
    assert cli._pid_running(1) is False
    assert cli._pid_running(42) is True
    cli._terminate_pid(1)
    cli._terminate_pid(42)
    assert kernel.terminated == 1
    assert kernel.closed >= 1


def test_pid_running_get_exit_code_failure(monkeypatch):
    class Kernel:
        def OpenProcess(self, access, inherit, pid):
            del access, inherit, pid
            return 0xBEEF

        def GetExitCodeProcess(self, handle, exit_code):
            del handle, exit_code
            return False

        def CloseHandle(self, handle):
            del handle
            return True

    monkeypatch.setattr(cli.ctypes, "windll", SimpleNamespace(kernel32=Kernel()), raising=False)
    monkeypatch.setattr(cli.os, "name", "nt")
    assert cli._pid_running(9) is False


def test_quit_and_foreground_osascript(monkeypatch):
    monkeypatch.setattr(cli.time, "sleep", lambda _s: None)
    monkeypatch.setattr(
        cli.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("no osascript")),
    )
    cli._quit_codex_app()
    cli._foreground_codex_app()

    runs: list[list[str]] = []

    def fake_run(cmd, check=False, stdout=None, stderr=None):
        del check, stdout, stderr
        runs.append(list(cmd))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    cli._quit_codex_app()
    cli._foreground_codex_app()
    assert all(cmd[0] == "osascript" for cmd in runs)
    assert len(runs) == 2


def test_resign_codex_app(monkeypatch, capsys):
    runs: list[list[str]] = []
    monkeypatch.setattr(
        cli.subprocess,
        "run",
        lambda cmd, check=True: runs.append(list(cmd)) or SimpleNamespace(returncode=0),
    )
    cli._resign_codex_app(Path("/tmp/Codex.app"))
    assert runs[0][:4] == ["codesign", "--force", "--deep", "--sign"]
    assert "Re-signed" in capsys.readouterr().out


def test_update_app_asar_integrity_success_and_missing_key(tmp_path):
    payload = b'{"files":{}}'
    asar = tmp_path / "app.asar"
    asar.write_bytes(struct.pack("<4I", 4, 0, 0, len(payload)) + payload)
    plist = tmp_path / "Info.plist"
    plist.write_bytes(plistlib.dumps({"ElectronAsarIntegrity": {"Resources/app.asar": {"hash": "old"}}}))
    cli._update_app_asar_integrity(asar, plist)
    data = plistlib.loads(plist.read_bytes())
    assert data["ElectronAsarIntegrity"]["Resources/app.asar"]["hash"] == cli._app_asar_header_hash(asar)

    bare = tmp_path / "bare.plist"
    bare.write_bytes(plistlib.dumps({"CFBundleName": "Codex"}))
    with pytest.raises(RuntimeError, match="Could not update ElectronAsarIntegrity"):
        cli._update_app_asar_integrity(asar, bare)


def test_ensure_user_codex_app_existing_and_missing(monkeypatch, tmp_path):
    user = tmp_path / "user" / "Codex.app"
    user_asar = user / "Contents/Resources/app.asar"
    user_asar.parent.mkdir(parents=True)
    user_asar.write_bytes(b"asar")
    monkeypatch.setattr(cli, "USER_CODEX_APP", user)
    assert cli._ensure_user_codex_app() == user

    missing_user = tmp_path / "missing-user" / "Codex.app"
    monkeypatch.setattr(cli, "USER_CODEX_APP", missing_user)
    monkeypatch.setattr(cli, "SYSTEM_CODEX_APP", tmp_path / "missing-system" / "Codex.app")
    with pytest.raises(SystemExit, match="Codex Desktop not found"):
        cli._ensure_user_codex_app()

    system = tmp_path / "system" / "Codex.app"
    system_asar = system / "Contents/Resources/app.asar"
    system_asar.parent.mkdir(parents=True)
    system_asar.write_bytes(b"sys")
    copied: list[list[str]] = []
    monkeypatch.setattr(cli, "SYSTEM_CODEX_APP", system)
    monkeypatch.setattr(
        cli.subprocess,
        "run",
        lambda cmd, check=True: copied.append(list(cmd)) or SimpleNamespace(returncode=0),
    )
    dest = cli._ensure_user_codex_app()
    assert dest == missing_user
    assert copied[0][0] == "ditto"


def test_codex_app_bundle_for_patch_prefers_writable_system(monkeypatch, tmp_path):
    system = tmp_path / "system" / "Codex.app"
    asar = system / "Contents/Resources/app.asar"
    asar.parent.mkdir(parents=True)
    asar.write_bytes(b"asar")
    monkeypatch.setattr(cli, "SYSTEM_CODEX_APP", system)
    monkeypatch.setattr(cli, "_path_is_writable", lambda path: path == asar)
    assert cli._codex_app_bundle_for_patch() == system

    monkeypatch.setattr(cli, "_path_is_writable", lambda path: False)
    user = tmp_path / "user" / "Codex.app"
    user_asar = user / "Contents/Resources/app.asar"
    user_asar.parent.mkdir(parents=True)
    user_asar.write_bytes(b"user")
    monkeypatch.setattr(cli, "USER_CODEX_APP", user)
    assert cli._codex_app_bundle_for_patch() == user


def test_patch_codex_app_darwin_error_branches(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli.sys, "platform", "darwin")
    monkeypatch.setattr(cli, "_quit_codex_app", lambda: None)
    app = tmp_path / "Codex.app"
    app.mkdir()
    monkeypatch.setattr(cli, "_codex_app_bundle_for_patch", lambda: app)
    assert cli.patch_codex_app() == 1
    assert "not found" in capsys.readouterr().err

    asar = app / "Contents/Resources/app.asar"
    asar.parent.mkdir(parents=True)
    asar.write_bytes(b"asar")
    assert cli.patch_codex_app() == 1
    assert "Info.plist not found" in capsys.readouterr().err

    plist = app / "Contents/Info.plist"
    plist.write_bytes(b"plist")
    monkeypatch.setattr(cli, "_has_command", lambda cmd: False)
    assert cli.patch_codex_app() == 1
    assert "npx is required" in capsys.readouterr().err


def test_patch_codex_app_darwin_packs_when_changed(monkeypatch, tmp_path):
    monkeypatch.setattr(cli.sys, "platform", "darwin")
    monkeypatch.setattr(cli, "_quit_codex_app", lambda: None)
    monkeypatch.setattr(cli, "_resign_codex_app", lambda app: None)
    monkeypatch.setattr(cli, "_has_command", lambda cmd: True)
    monkeypatch.setattr(cli, "_patch_codex_desktop_bundles", lambda workdir: True)
    monkeypatch.setattr(cli, "_update_app_asar_integrity", lambda *args: None)
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    monkeypatch.setattr(cli, "RUNTIME_DIR", runtime)
    app = tmp_path / "Codex.app"
    asar = app / "Contents/Resources/app.asar"
    plist = app / "Contents/Info.plist"
    asar.parent.mkdir(parents=True)
    asar.write_bytes(b"asar-bytes")
    plist.write_bytes(b"plist")
    monkeypatch.setattr(cli, "_codex_app_bundle_for_patch", lambda: app)
    monkeypatch.setattr(cli, "USER_CODEX_APP", app)
    runs: list[list[str]] = []

    def fake_run(cmd, check=False, **kwargs):
        del check, kwargs
        runs.append(list(cmd))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    assert cli.patch_codex_app() == 0
    assert any(cmd[:2] == ["npx", "--yes"] and "extract" in cmd for cmd in runs)
    assert any(cmd[:2] == ["npx", "--yes"] and "pack" in cmd for cmd in runs)
    assert (runtime / cli.APP_ASAR_BACKUP_NAME).is_file()


def test_patch_codex_app_darwin_skips_pack_when_unchanged(monkeypatch, tmp_path):
    monkeypatch.setattr(cli.sys, "platform", "darwin")
    monkeypatch.setattr(cli, "_quit_codex_app", lambda: None)
    monkeypatch.setattr(cli, "_has_command", lambda cmd: True)
    monkeypatch.setattr(cli, "_patch_codex_desktop_bundles", lambda workdir: False)
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    monkeypatch.setattr(cli, "RUNTIME_DIR", runtime)
    app = tmp_path / "other" / "Codex.app"
    asar = app / "Contents/Resources/app.asar"
    plist = app / "Contents/Info.plist"
    asar.parent.mkdir(parents=True)
    asar.write_bytes(b"asar")
    plist.write_bytes(b"plist")
    monkeypatch.setattr(cli, "_codex_app_bundle_for_patch", lambda: app)
    monkeypatch.setattr(cli, "USER_CODEX_APP", tmp_path / "user" / "Codex.app")
    monkeypatch.setattr(
        cli.subprocess,
        "run",
        lambda cmd, check=False, **kwargs: SimpleNamespace(returncode=0),
    )
    assert cli.patch_codex_app() == 0


def test_patch_codex_app_darwin_returns_when_patch_helper_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(cli.sys, "platform", "darwin")
    monkeypatch.setattr(cli, "_quit_codex_app", lambda: None)
    monkeypatch.setattr(cli, "_has_command", lambda cmd: True)
    monkeypatch.setattr(cli, "_patch_codex_desktop_bundles", lambda workdir: None)
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    monkeypatch.setattr(cli, "RUNTIME_DIR", runtime)
    app = tmp_path / "Codex.app"
    asar = app / "Contents/Resources/app.asar"
    plist = app / "Contents/Info.plist"
    asar.parent.mkdir(parents=True)
    asar.write_bytes(b"asar")
    plist.write_bytes(b"plist")
    monkeypatch.setattr(cli, "_codex_app_bundle_for_patch", lambda: app)
    monkeypatch.setattr(
        cli.subprocess,
        "run",
        lambda cmd, check=False, **kwargs: SimpleNamespace(returncode=0),
    )
    assert cli.patch_codex_app() == 1


def test_restore_codex_app_bundle_darwin(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli.sys, "platform", "darwin")
    monkeypatch.setattr(cli, "_quit_codex_app", lambda: None)
    monkeypatch.setattr(cli, "_resign_codex_app", lambda app: None)
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    monkeypatch.setattr(cli, "RUNTIME_DIR", runtime)
    app = tmp_path / "Codex.app"
    asar = app / "Contents/Resources/app.asar"
    plist = app / "Contents/Info.plist"
    asar.parent.mkdir(parents=True)
    asar.write_bytes(b"patched")
    plist.write_bytes(b"plist")
    monkeypatch.setattr(cli, "patched_codex_app_bundle", lambda: app)
    assert cli.restore_codex_app_bundle() == 0
    assert "No app.asar backup" in capsys.readouterr().out

    backup = runtime / cli.APP_ASAR_BACKUP_NAME
    backup.write_bytes(b"orig-asar")
    info_backup = runtime / cli.INFO_PLIST_BACKUP_NAME
    info_backup.write_bytes(b"orig-plist")
    assert cli.restore_codex_app_bundle() == 0
    assert asar.read_bytes() == b"orig-asar"
    assert plist.read_bytes() == b"orig-plist"


def test_restore_codex_app_bundle_updates_integrity_without_plist_backup(monkeypatch, tmp_path):
    monkeypatch.setattr(cli.sys, "platform", "darwin")
    monkeypatch.setattr(cli, "_quit_codex_app", lambda: None)
    monkeypatch.setattr(cli, "_resign_codex_app", lambda app: None)
    updated: list[tuple] = []
    monkeypatch.setattr(cli, "_update_app_asar_integrity", lambda *args: updated.append(args))
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    monkeypatch.setattr(cli, "RUNTIME_DIR", runtime)
    (runtime / cli.APP_ASAR_BACKUP_NAME).write_bytes(b"orig")
    app = tmp_path / "Codex.app"
    asar = app / "Contents/Resources/app.asar"
    plist = app / "Contents/Info.plist"
    asar.parent.mkdir(parents=True)
    asar.write_bytes(b"patched")
    plist.write_bytes(b"plist")
    monkeypatch.setattr(cli, "patched_codex_app_bundle", lambda: None)
    monkeypatch.setattr(cli, "_codex_app_bundle_for_patch", lambda: app)
    assert cli.restore_codex_app_bundle() == 0
    assert updated == [(asar, plist)]


def test_exec_codex_windows_uses_subprocess_call(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "_override_args", lambda *args: ["-c", "foo=bar"])
    monkeypatch.setattr(cli.os, "name", "nt")
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        cli.subprocess,
        "call",
        lambda args, env=None: captured.update({"args": list(args), "env": env}) or 7,
    )
    with pytest.raises(SystemExit) as exc:
        cli.exec_codex(tmp_path / "models.json", 8767, ["exec", "hi"])
    assert exc.value.code == 7
    assert captured["args"][:1] == ["codex"]


def test_entrypoint_broken_pipe(monkeypatch):
    monkeypatch.setattr(cli, "main", lambda: (_ for _ in ()).throw(BrokenPipeError()))
    flushes = {"n": 0}
    real_flush = cli.sys.stdout.flush

    def maybe_flush():
        flushes["n"] += 1
        if flushes["n"] == 1:
            raise BrokenPipeError()
        return real_flush()

    monkeypatch.setattr(cli.sys.stdout, "flush", maybe_flush)
    monkeypatch.setattr(cli.sys.stdout, "fileno", lambda: 1)
    monkeypatch.setattr(cli.os, "open", lambda *args, **kwargs: 3)
    monkeypatch.setattr(cli.os, "dup2", lambda *args: None)
    assert cli._entrypoint() == 0
    assert flushes["n"] >= 1

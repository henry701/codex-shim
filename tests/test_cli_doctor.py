from __future__ import annotations

import json

from codex_shim import cli


def _patch_cli_paths(monkeypatch, tmp_path):
    runtime_dir = tmp_path / "runtime"
    codex_home = tmp_path / "codex-home"
    monkeypatch.setattr(cli, "RUNTIME_DIR", runtime_dir)
    monkeypatch.setattr(cli, "CATALOG_PATH", runtime_dir / "custom_model_catalog.json")
    monkeypatch.setattr(cli, "CONFIG_PATH", runtime_dir / "config.toml")
    monkeypatch.setattr(cli, "LOG_PATH", runtime_dir / "shim.log")
    monkeypatch.setattr(cli, "PID_PATH", runtime_dir / "shim.pid")
    monkeypatch.setattr(cli, "CODEX_CONFIG_PATH", codex_home / "config.toml")
    monkeypatch.setattr(cli, "DESKTOP_CATALOG_PATH", codex_home / "custom_model_catalog.json")
    monkeypatch.setattr(cli, "CODEX_CONFIG_BACKUP_PATH", runtime_dir / "config.toml.before-codex-shim")
    monkeypatch.setattr(cli, "migrate_thread_providers", lambda **_kwargs: {"updated": 0, "databases": {}})
    return runtime_dir, codex_home


def test_doctor_accepts_fork_openai_provider_config(monkeypatch, tmp_path, capsys):
    runtime_dir, codex_home = _patch_cli_paths(monkeypatch, tmp_path)
    settings = tmp_path / "models.json"
    settings.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "model": "llama3.2",
                        "display_name": "Llama",
                        "provider": "generic-chat-completion-api",
                        "base_url": "http://127.0.0.1:11434/v1",
                        "api_key": "local",
                    }
                ]
            }
        )
    )
    runtime_dir.mkdir(parents=True)
    codex_home.mkdir(parents=True)
    cli.install_codex_config(settings, 8765)
    (runtime_dir / "shim.log").write_text("[req] /v1/responses model='llama3-2'\n")
    monkeypatch.setattr(cli, "_health", lambda port: {"models": 1})
    monkeypatch.setattr(cli, "_read_pid", lambda: 123)
    monkeypatch.setattr(cli, "_pid_running", lambda pid: True)

    code = cli.main(["--settings", str(settings), "doctor"])

    out = capsys.readouterr().out
    assert code == 0
    assert "OK" in out
    assert "model_provider = openai" in out
    assert "openai_base_url = http://127.0.0.1:8765/v1" in out
    assert "[model_providers.codex_shim]" not in out


def test_doctor_fails_when_enabled_config_uses_legacy_codex_shim_provider(monkeypatch, tmp_path, capsys):
    runtime_dir, codex_home = _patch_cli_paths(monkeypatch, tmp_path)
    settings = tmp_path / "models.json"
    settings.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "model": "llama3.2",
                        "display_name": "Llama",
                        "provider": "generic-chat-completion-api",
                        "base_url": "http://127.0.0.1:11434/v1",
                        "api_key": "local",
                    }
                ]
            }
        )
    )
    runtime_dir.mkdir(parents=True)
    codex_home.mkdir(parents=True)
    cli.DESKTOP_CATALOG_PATH.write_text(json.dumps({"models": [{"slug": "llama3-2"}]}))
    cli.CODEX_CONFIG_PATH.write_text(
        'model = "llama3-2"\n'
        'model_provider = "codex_shim"\n'
        '[model_providers.codex_shim]\n'
        'base_url = "http://127.0.0.1:8765/v1"\n'
    )
    monkeypatch.setattr(cli, "_health", lambda port: {"models": 1})
    monkeypatch.setattr(cli, "_read_pid", lambda: 123)
    monkeypatch.setattr(cli, "_pid_running", lambda pid: True)

    code = cli.main(["--settings", str(settings), "doctor"])

    out = capsys.readouterr().out
    assert code == 1
    assert "FAIL" in out
    assert "model_provider is codex_shim" in out


def test_doctor_warns_when_health_ok_but_pid_file_is_stale(monkeypatch, tmp_path, capsys):
    runtime_dir, _codex_home = _patch_cli_paths(monkeypatch, tmp_path)
    settings = tmp_path / "models.json"
    settings.write_text(json.dumps({"models": []}))
    runtime_dir.mkdir(parents=True)
    monkeypatch.setattr(cli, "_health", lambda port: {"models": 3, "ok": True})
    monkeypatch.setattr(cli, "_read_pid", lambda: 111)
    monkeypatch.setattr(cli, "_pid_running", lambda pid: False)
    monkeypatch.setattr(cli, "_listener_pid", lambda port: 4242)
    monkeypatch.setattr(cli, "chatgpt_passthrough_available", lambda: False)
    monkeypatch.setattr(cli, "cursor_passthrough_available", lambda: False)

    code = cli.main(["--settings", str(settings), "doctor"])

    out = capsys.readouterr().out
    assert code == 0
    assert "WARN" in out
    assert "stale pid file" in out
    assert "4242" in out

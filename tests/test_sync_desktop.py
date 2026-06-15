from __future__ import annotations

import json
from pathlib import Path

from codex_shim import cli


def test_sync_desktop_writes_catalog_only(monkeypatch, tmp_path):
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
    codex_home = tmp_path / "codex-home"
    desktop_catalog = codex_home / "custom_model_catalog.json"
    config_path = codex_home / "config.toml"
    runtime_dir = tmp_path / "runtime"

    monkeypatch.setattr(cli, "RUNTIME_DIR", runtime_dir)
    monkeypatch.setattr(cli, "CATALOG_PATH", runtime_dir / "custom_model_catalog.json")
    monkeypatch.setattr(cli, "CONFIG_PATH", runtime_dir / "config.toml")
    monkeypatch.setattr(cli, "CODEX_CONFIG_PATH", config_path)
    monkeypatch.setattr(cli, "DESKTOP_CATALOG_PATH", desktop_catalog)
    monkeypatch.setattr(cli, "CODEX_CONFIG_BACKUP_PATH", runtime_dir / "config.toml.before-codex-shim")

    assert cli.sync_desktop(settings, 8765) == 0

    payload = json.loads(desktop_catalog.read_text())
    assert any(entry.get("slug") == "llama3-2" for entry in payload["models"])
    assert not config_path.exists() or 'model_provider = "codex_shim"' not in config_path.read_text()


def test_sync_desktop_can_install_config(monkeypatch, tmp_path):
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
    codex_home = tmp_path / "codex-home"
    desktop_catalog = codex_home / "custom_model_catalog.json"
    config_path = codex_home / "config.toml"
    runtime_dir = tmp_path / "runtime"

    monkeypatch.setattr(cli, "RUNTIME_DIR", runtime_dir)
    monkeypatch.setattr(cli, "CATALOG_PATH", runtime_dir / "custom_model_catalog.json")
    monkeypatch.setattr(cli, "CONFIG_PATH", runtime_dir / "config.toml")
    monkeypatch.setattr(cli, "CODEX_CONFIG_PATH", config_path)
    monkeypatch.setattr(cli, "DESKTOP_CATALOG_PATH", desktop_catalog)
    monkeypatch.setattr(cli, "CODEX_CONFIG_BACKUP_PATH", runtime_dir / "config.toml.before-codex-shim")

    assert cli.sync_desktop(settings, 8765, install_config=True) == 0

    config_text = config_path.read_text()
    assert str(desktop_catalog) in config_text
    assert 'model_provider = "openai"' in config_text
    assert 'openai_base_url = "http://127.0.0.1:8765/v1"' in config_text
    assert "[model_providers.codex_shim]" not in config_text

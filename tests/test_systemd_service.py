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

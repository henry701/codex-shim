from __future__ import annotations

from pathlib import Path

from codex_shim import cli


def test_systemd_unit_runs_generate_then_foreground():
    unit = cli._systemd_unit_content(
        codex_shim_bin="/usr/bin/codex-shim",
        settings_path=Path("/home/user/.codex-shim/models.json"),
        port=8765,
    )
    assert "ExecStartPre=/usr/bin/codex-shim --settings /home/user/.codex-shim/models.json --port 8765 generate" in unit
    assert "ExecStart=/usr/bin/codex-shim --settings /home/user/.codex-shim/models.json --port 8765 run" in unit
    assert "WantedBy=default.target" in unit

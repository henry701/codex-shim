from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SMOKE_SCRIPT = ROOT / "scripts" / "smoke_chatgpt_passthrough.sh"


def test_smoke_chatgpt_passthrough_script_is_valid_bash():
    proc = subprocess.run(["bash", "-n", str(SMOKE_SCRIPT)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    text = SMOKE_SCRIPT.read_text()
    assert " serve " in text or " serve>>" in text or 'serve >>' in text
    assert "SMOKE_RESTART=1 on port 8765" in text


@pytest.mark.integration
def test_live_chatgpt_passthrough_smoke_optional():
    if os.environ.get("CODEX_SHIM_LIVE_SMOKE") != "1":
        pytest.skip("set CODEX_SHIM_LIVE_SMOKE=1 to run e2e ChatGPT smoke on 8766")
    env = os.environ.copy()
    env["SMOKE_PORT"] = env.get("SMOKE_PORT", "8766")
    env["SMOKE_RESTART"] = "0"
    env["SMOKE_PROMPT"] = env.get(
        "SMOKE_PROMPT",
        "Reply with the single word pong. Do not use tools.",
    )
    proc = subprocess.run(
        ["bash", str(SMOKE_SCRIPT)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr

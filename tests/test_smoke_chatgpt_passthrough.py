from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CHATGPT_SMOKE = ROOT / "scripts" / "smoke_chatgpt_passthrough.sh"
OPENCODE_SMOKE = ROOT / "scripts" / "smoke_opencode_free.sh"
SURFACES_SMOKE = ROOT / "scripts" / "smoke_net_surfaces.sh"


def _assert_valid_bash(path: Path) -> str:
    proc = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
    assert proc.returncode == 0, f"{path}: {proc.stderr}"
    return path.read_text()


def test_smoke_chatgpt_passthrough_script_is_valid_bash():
    text = _assert_valid_bash(CHATGPT_SMOKE)
    assert " serve " in text or " serve>>" in text or "serve >>" in text or '"${PORT}" serve' in text
    assert "SMOKE_RESTART=1 on port 8765" in text
    assert "codex-gpt-5-6-luna" in text
    assert "load-env.sh" in text
    assert "--settings" in text
    assert "SMOKE_TIMEOUT" in text
    assert "timeout --kill-after" in text
    assert "8767" in text
    assert "not a shim" in text or "occupied" in text.lower()
    assert "turn.completed" in text
    assert "mcp_servers={}" in text


def test_smoke_opencode_free_script_is_valid_bash():
    text = _assert_valid_bash(OPENCODE_SMOKE)
    assert "oc-free-" in text
    assert "smoke_chatgpt_passthrough.sh" in text
    assert "8765" in text


def test_smoke_net_surfaces_script_is_valid_bash():
    text = _assert_valid_bash(SURFACES_SMOKE)
    assert "smoke_chatgpt_passthrough.sh" in text
    assert "smoke_opencode_free.sh" in text
    assert "SMOKE_RESTART=1 on port 8765" in text
    assert "nvidia-nvidia-nemotron-3-5-lightning-30b-a3b" in text
    assert "or-nvidia-nemotron-3-5-lightning-free" in text
    assert "oc-free-muse-spark-1-3-contributor-free" in text
    assert "OPENCODE_API_KEY" in text
    assert "nvidia|openrouter|muse|zai" in text


def test_codex_exec_smoke_helper_is_valid_bash():
    path = ROOT / "scripts" / "codex_exec_smoke.sh"
    text = _assert_valid_bash(path)
    assert "mcp_servers={}" in text
    assert "timeout --kill-after" in text
    assert "8767" in text
    assert "Refusing" in text and "8765" in text
    assert "/dev/null" in text
    env = os.environ.copy()
    env["SMOKE_PORT"] = "8765"
    proc = subprocess.run(
        ["bash", str(path), "oc-free-ling-3-0-flash-fin-free", "/tmp/codex-exec-smoke-refuse.jsonl", "1"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert proc.returncode == 1
    assert "8765" in proc.stdout + proc.stderr


def test_smoke_zai_surface_fails_without_opencode_key():
    env = os.environ.copy()
    env.pop("OPENCODE_API_KEY", None)
    env["OPENCODE_API_KEY"] = ""
    env["SMOKE_SURFACES"] = "zai"
    env["SMOKE_RESTART"] = "0"
    proc = subprocess.run(
        ["bash", str(SURFACES_SMOKE)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert proc.returncode == 1
    assert "OPENCODE_API_KEY is unset" in proc.stdout + proc.stderr


def _live_smoke_env() -> dict[str, str]:
    env = os.environ.copy()
    env["SMOKE_PORT"] = env.get("SMOKE_PORT", "8766")
    env["SMOKE_RESTART"] = "0"
    env["SMOKE_PROMPT"] = env.get(
        "SMOKE_PROMPT",
        "Reply with the single word pong. Do not use tools.",
    )
    env.setdefault("SMOKE_REASONING_EFFORT", "low")
    return env


@pytest.mark.integration
def test_live_chatgpt_passthrough_smoke_optional():
    if os.environ.get("CODEX_SHIM_LIVE_SMOKE") != "1":
        pytest.skip("set CODEX_SHIM_LIVE_SMOKE=1 to run e2e ChatGPT smoke on 8766")
    proc = subprocess.run(
        ["bash", str(CHATGPT_SMOKE)],
        cwd=ROOT,
        env=_live_smoke_env(),
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr


@pytest.mark.integration
def test_live_opencode_free_smoke_optional():
    if os.environ.get("CODEX_SHIM_LIVE_SMOKE") != "1":
        pytest.skip("set CODEX_SHIM_LIVE_SMOKE=1 to run e2e OpenCode Free smoke on 8766")
    proc = subprocess.run(
        ["bash", str(OPENCODE_SMOKE)],
        cwd=ROOT,
        env=_live_smoke_env(),
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr

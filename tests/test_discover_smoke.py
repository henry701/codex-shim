from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from codex_shim import cli
from codex_shim.discover import (
    discover_byok_models,
    discover_chatgpt_models_from_cursor,
    fetch_nvidia_integrate_model_ids,
    fetch_openrouter_free_model_ids,
    fetch_zen_public_model_ids,
    list_opencode_cli_models,
)
from codex_shim.settings import ModelSettings

pytestmark = pytest.mark.enable_model_discovery


@pytest.fixture
def auth_missing(monkeypatch, tmp_path):
    monkeypatch.setattr("codex_shim.settings.DEFAULT_CODEX_AUTH", tmp_path / "missing-auth.json")


@pytest.fixture
def user_models_path() -> Path | None:
    path = Path.home() / ".codex-shim" / "models.json"
    return path if path.exists() else None


def test_opencode_cli_listing_non_empty_when_installed():
    if shutil.which("opencode") is None:
        pytest.skip("opencode CLI not installed")
    lines = list_opencode_cli_models()
    assert lines
    assert any(line.startswith("opencode/") for line in lines)
    assert any(line.startswith("openrouter/") for line in lines)


def test_live_zen_public_ids_when_network_available():
    if shutil.which("opencode") is None:
        pytest.skip("opencode CLI not installed")
    ids = fetch_zen_public_model_ids()
    assert ids
    assert any("free" in model_id or model_id == "big-pickle" for model_id in ids)


def test_live_openrouter_free_ids_include_router_when_opencode_installed():
    if shutil.which("opencode") is None:
        pytest.skip("opencode CLI not installed")
    ids = fetch_openrouter_free_model_ids()
    assert "openrouter/free" in ids
    assert all(model_id == "openrouter/free" or model_id.endswith(":free") for model_id in ids)


def test_live_nvidia_integrate_ids_exclude_flux_when_opencode_installed():
    if shutil.which("opencode") is None:
        pytest.skip("opencode CLI not installed")
    ids = fetch_nvidia_integrate_model_ids()
    assert ids
    assert not any("flux" in model_id.lower() for model_id in ids)


def test_live_cursor_chatgpt_listing_when_installed():
    if shutil.which("cursor-agent") is None:
        pytest.skip("cursor-agent not installed")
    rows = discover_chatgpt_models_from_cursor()
    assert rows
    assert all(row[0].startswith(("gpt-", "codex-", "o")) for row in rows)


def test_cli_discover_exits_zero_with_user_models(user_models_path: Path | None):
    if user_models_path is None:
        pytest.skip("~/.codex-shim/models.json not present")
    code = cli.discover_models(user_models_path)
    assert code == 0


def test_cli_generate_writes_catalog_with_discovered_slugs(
    tmp_path, monkeypatch, user_models_path: Path | None, auth_missing
):
    if user_models_path is None:
        pytest.skip("~/.codex-shim/models.json not present")
    runtime = tmp_path / ".codex-shim"
    monkeypatch.setattr(cli, "RUNTIME_DIR", runtime)
    monkeypatch.setattr(cli, "CATALOG_PATH", runtime / "custom_model_catalog.json")
    monkeypatch.setattr(cli, "CONFIG_PATH", runtime / "config.toml")
    monkeypatch.setattr(cli, "DESKTOP_CATALOG_PATH", tmp_path / "desktop-catalog.json")
    if shutil.which("opencode") is None:
        monkeypatch.setattr(
            "codex_shim.discover.fetch_zen_public_model_ids",
            lambda: ["minimax-m3-free"],
        )
        monkeypatch.setattr(
            "codex_shim.discover.fetch_openrouter_free_model_ids",
            lambda: ["openrouter/free"],
        )
        monkeypatch.setattr(
            "codex_shim.discover.fetch_nvidia_integrate_model_ids",
            lambda: [],
        )
    explicit = ModelSettings(user_models_path).load_explicit()
    cli.generate(user_models_path, 8765)
    catalog = json.loads((runtime / "custom_model_catalog.json").read_text())
    slugs = {entry["slug"] for entry in catalog["models"]}
    loaded = ModelSettings(user_models_path).load()
    assert len(loaded) > len(explicit)
    assert (runtime / "config.toml").exists()
    assert "local-llama" in slugs
    for model in loaded:
        if not model.api_key.strip():
            continue
        if model.raw.get("discovered") or model.slug in {entry.slug for entry in explicit}:
            assert model.slug in slugs


def test_module_entrypoint_help():
    result = subprocess.run(
        [sys.executable, "-m", "codex_shim.cli", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "discover" in result.stdout

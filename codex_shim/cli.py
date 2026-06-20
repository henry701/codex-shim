from __future__ import annotations

import argparse
import os
from pathlib import Path
import ctypes
import shutil
import signal
import socket
import subprocess
import sys
import time
import hashlib
import json
import plistlib
import re
import struct
from urllib.request import urlopen

from . import router as router_module
from .catalog import _toml_escape, codex_config_overrides, write_catalog, write_config
from .cursor_passthrough import (
    cursor_catalog_models,
    cursor_passthrough_available,
    cursor_passthrough_display_names,
    cursor_upstream_model,
    is_cursor_passthrough_slug,
)
from .discover import discover_byok_models, discover_summary
from .catalog_slugs import CHATGPT_CATALOG_SLUG
from .settings import (
    DEFAULT_SETTINGS,
    DEFAULT_HOST,
    DEFAULT_PORT,
    OPENAI_PROVIDER_ID,
    PROVIDER_NAME,
    ModelSettings,
    available_model_slugs,
    chatgpt_catalog_slug,
    chatgpt_passthrough_available,
    chatgpt_passthrough_display_names,
    chatgpt_passthrough_slugs,
    default_model_slug,
    is_chatgpt_passthrough_slug,
    usable_byok_models,
    byok_model_has_credentials,
)
from .opencode_go import (
    OPENCODE_GO_API_KEY_ENV,
    OPENCODE_GO_BASE_URL,
    refresh_opencode_go_settings,
)
from .thread_migrate import migrate_thread_providers


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_DIR = PROJECT_ROOT / ".codex-shim"
CATALOG_PATH = RUNTIME_DIR / "custom_model_catalog.json"
CONFIG_PATH = RUNTIME_DIR / "config.toml"
PID_PATH = RUNTIME_DIR / "shim.pid"
LOG_PATH = RUNTIME_DIR / "shim.log"
CODEX_CONFIG_PATH = Path.home() / ".codex" / "config.toml"
DESKTOP_CATALOG_PATH = CODEX_CONFIG_PATH.parent / "custom_model_catalog.json"
LOAD_ENV_SCRIPT = Path.home() / ".codex-shim" / "load-env.sh"
CODEX_CONFIG_BACKUP_PATH = RUNTIME_DIR / "config.toml.before-codex-shim"
MANAGED_BEGIN = "# >>> codex-shim managed >>>"
MANAGED_END = "# <<< codex-shim managed <<<"
WINDOWS_PROCESS_TERMINATE = 0x0001
WINDOWS_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
WINDOWS_STILL_ACTIVE = 259
PREVIOUS_TOP_LEVEL_PREFIX = "# codex-shim previous-top-level = "
PREVIOUS_FEATURES_PREFIX = "# codex-shim previous-features = "
MANAGED_TOP_LEVEL_KEYS = {"model", "model_provider", "model_catalog_json", "openai_base_url"}
MANAGED_TOP_LEVEL_RESTORE_ORDER = (
    "model",
    "model_provider",
    "openai_base_url",
    "model_catalog_json",
)
MANAGED_FEATURES_SECTION = "features"
MANAGED_TOOL_SEARCH_FEATURE_KEY = "tool_search_always_defer_mcp_tools"
MANAGED_REQUEST_COMPRESSION_FEATURE_KEY = "enable_request_compression"
MANAGED_FEATURE_KEYS = {MANAGED_TOOL_SEARCH_FEATURE_KEY, MANAGED_REQUEST_COMPRESSION_FEATURE_KEY}
APP_ASAR_BACKUP_NAME = "app.asar.before-codex-shim-model-picker-patch"
INFO_PLIST_BACKUP_NAME = "Info.plist.before-codex-shim-model-picker-patch"
SYSTEM_CODEX_APP = Path("/Applications/Codex.app")
USER_CODEX_APP = Path.home() / "Applications" / "Codex.app"
MODEL_PICKER_NEEDLE = "let u=c.useHiddenModels&&o!==`amazonBedrock`,d;"
MODEL_PICKER_REPLACEMENT = "let u=!1,d;"
SIDEBAR_RECENT_THREADS_NEEDLE = (
    "listRecentThreads({cursor:e,limit:t}){return this.params.requestClient.sendRequest(`thread/list`,"
    "{limit:t,cursor:e,sortKey:this.recentConversationSortKey,modelProviders:null,archived:!1,sourceKinds:ke})}"
)
SIDEBAR_RECENT_THREADS_REPLACEMENT = (
    "listRecentThreads({cursor:e,limit:t}){return this.params.requestClient.sendRequest(`thread/list`,"
    "{limit:t,cursor:e,sortKey:this.recentConversationSortKey,modelProviders:[],archived:!1,sourceKinds:ke})}"
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="codex-shim")
    parser.add_argument("--settings", type=Path, default=DEFAULT_SETTINGS)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("generate")
    discover_parser = sub.add_parser(
        "discover",
        help="List models that would be auto-discovered from provider CLIs/APIs.",
    )
    discover_parser.add_argument(
        "--refresh",
        action="store_true",
        help="Bypass cached Cursor model listings.",
    )
    sub.add_parser("list")
    sub.add_parser("start")
    sub.add_parser("enable")
    sub.add_parser("stop")
    sub.add_parser("disable")
    sub.add_parser("restart")
    sub.add_parser("run", help="Run the shim in the foreground (for systemd and debugging).")
    sub.add_parser(
        "sync-desktop",
        help="Regenerate catalogs under ~/.codex (does not modify ~/.codex/config.toml).",
    )
    sub.add_parser(
        "install-service",
        help="Install and enable a systemd user service so the shim starts at login/boot.",
    )
    sub.add_parser("status")
    sub.add_parser("patch-app", help="Patch Codex Desktop picker/sidebar handling for custom shim models.")
    sub.add_parser("restore-app", help="Restore Codex Desktop app.asar from the pre-patch backup.")
    migrate_threads_parser = sub.add_parser(
        "migrate-threads",
        help="Rewrite legacy codex_shim thread rows to model_provider=openai in ~/.codex/state_*.sqlite.",
    )
    migrate_threads_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report how many threads would be updated without writing.",
    )

    opencode_parser = sub.add_parser("opencode-go", help="Discover and configure OpenCode Go models.")
    opencode_sub = opencode_parser.add_subparsers(dest="opencode_go_command", required=True)
    refresh_parser = opencode_sub.add_parser("refresh", help="Refresh OpenCode Go models into the settings file.")
    refresh_parser.add_argument("--api-key-env", default=OPENCODE_GO_API_KEY_ENV)
    refresh_parser.add_argument("--base-url", default=OPENCODE_GO_BASE_URL)
    refresh_parser.add_argument("--prefer", choices=["chat", "messages"], default="chat")
    refresh_parser.add_argument("--timeout", type=float, default=30.0)

    model_parser = sub.add_parser("model", help="List or set the active shim model in Codex config.")
    model_sub = model_parser.add_subparsers(dest="model_command", required=True)
    model_sub.add_parser("list")
    use_parser = model_sub.add_parser("use")
    use_parser.add_argument("model_slug")

    codex_parser = sub.add_parser("codex", help="Run Codex CLI with opt-in shim config overrides.")
    codex_parser.add_argument("args", nargs=argparse.REMAINDER)

    app_parser = sub.add_parser("app", help="Launch Codex Desktop with opt-in shim config overrides.")
    app_parser.add_argument("-m", "--model", dest="model_slug")
    app_parser.add_argument("path", nargs="?", default=".")

    args = parser.parse_args(argv)
    if args.command == "generate":
        generate(args.settings, args.port)
        return 0
    if args.command == "discover":
        return discover_models(args.settings, refresh=getattr(args, "refresh", False))
    if args.command == "list":
        return list_models(args.settings)
    if args.command in {"start", "enable"}:
        generate(args.settings, args.port)
        code = start(args.settings, args.port)
        if code == 0 and args.command == "enable":
            install_codex_config(args.settings, args.port)
        return code
    if args.command in {"stop", "disable"}:
        if args.command == "disable":
            restore_codex_config()
        return stop()
    if args.command == "restart":
        if stop() != 0:
            print("Restart aborted: could not stop the running shim.", file=sys.stderr)
            return 1
        generate(args.settings, args.port)
        return start(args.settings, args.port)
    if args.command == "run":
        return run_foreground(args.settings, args.port)
    if args.command == "sync-desktop":
        return sync_desktop(args.settings, args.port)
    if args.command == "install-service":
        return install_service(args.settings, args.port)
    if args.command == "status":
        return status(args.port)
    if args.command == "patch-app":
        return patch_codex_app()
    if args.command == "restore-app":
        return restore_codex_app_bundle()
    if args.command == "migrate-threads":
        return migrate_threads_command(dry_run=args.dry_run)
    if args.command == "opencode-go":
        if args.opencode_go_command == "refresh":
            return refresh_opencode_go(args.settings, args.api_key_env, args.base_url, args.prefer, args.timeout)
    if args.command == "model":
        if args.model_command == "list":
            return list_models(args.settings)
        if args.model_command == "use":
            generate(args.settings, args.port)
            ensure_started(args.settings, args.port)
            install_codex_config(args.settings, args.port, args.model_slug)
            print(f"Active Codex shim model: {args.model_slug}")
            return 0
    if args.command == "codex":
        generate(args.settings, args.port)
        ensure_started(args.settings, args.port)
        exec_codex(args.settings, args.port, args.args)
        return 0
    if args.command == "app":
        generate(args.settings, args.port)
        ensure_started(args.settings, args.port)
        install_codex_config(args.settings, args.port, args.model_slug)
        exec_codex_app(args.settings, args.port, args.path)
        return 0
    return 2


def _load_settings_data(settings_path: Path) -> dict | None:
    expanded = Path(settings_path).expanduser()
    if not expanded.exists():
        return None
    try:
        data = json.loads(expanded.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _load_models(settings_path: Path):
    expanded = Path(settings_path).expanduser()
    try:
        return ModelSettings(expanded).load()
    except FileNotFoundError as exc:
        raise SystemExit(
            f"Settings file not found: {expanded}\n"
            "Create ~/.codex-shim/models.json, or pass --settings /path/to/models.json."
        ) from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Settings file is not valid JSON: {expanded}: {exc}") from exc


def _active_router(models, settings_path: Path):
    """RouterConfig when the Auto Router is enabled and has a usable candidate."""
    config = router_module.load_router_config(Path(settings_path).expanduser())
    if config and router_module.router_is_active(config, available_model_slugs(models)):
        return config
    return None


def _publish_catalog(models, router_config) -> Path:
    write_catalog(models, CATALOG_PATH, router_config=router_config)
    DESKTOP_CATALOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(CATALOG_PATH, DESKTOP_CATALOG_PATH)
    return DESKTOP_CATALOG_PATH


def generate(settings_path: Path, port: int) -> None:
    models = _load_models(settings_path)
    try:
        default_model_slug(models)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    router_config = router_module.load_router_config(Path(settings_path).expanduser())
    _publish_catalog(models, router_config)
    write_config(models, CONFIG_PATH, CATALOG_PATH, port)
    discovered_count = sum(1 for model in models if model.raw.get("discovered"))
    print(f"Generated {len(models)} model entries ({discovered_count} auto-discovered):")
    if _active_router(models, settings_path) is not None:
        print(f"  auto router: {router_config.slug} ({router_config.display_name})")
    print(f"  catalog: {CATALOG_PATH}")
    print(f"  desktop catalog: {DESKTOP_CATALOG_PATH}")
    print(f"  config:  {CONFIG_PATH}")
    print("No files under ~/.codex were modified.")


def sync_desktop(settings_path: Path, port: int, model_slug: str | None = None, *, install_config: bool = False) -> int:
    models = _load_models(settings_path)
    try:
        default_model_slug(models)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    router_config = router_module.load_router_config(Path(settings_path).expanduser())
    desktop_catalog = _publish_catalog(models, router_config)
    write_config(models, CONFIG_PATH, CATALOG_PATH, port)
    if install_config:
        install_codex_config(settings_path, port, model_slug)
    discovered_count = sum(1 for model in models if model.raw.get("discovered"))
    catalog_models = len(json.loads(desktop_catalog.read_text()).get("models", []))
    print(f"Synced {catalog_models} Desktop catalog entries ({discovered_count} auto-discovered routes):")
    print(f"  desktop catalog: {desktop_catalog}")
    if install_config:
        print(f"  codex config:    {CODEX_CONFIG_PATH}")
    else:
        print(f"  codex config:    (unchanged; run `codex-shim enable` to wire the shim provider)")
    return 0


def discover_models(settings_path: Path, *, refresh: bool = False) -> int:
    expanded = Path(settings_path).expanduser()
    settings_data = _load_settings_data(expanded)
    try:
        explicit = ModelSettings(expanded).load_explicit() if expanded.exists() else []
    except FileNotFoundError:
        explicit = []
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Settings file is not valid JSON: {expanded}: {exc}") from exc

    if refresh and cursor_passthrough_available():
        cursor_catalog_models(force_refresh=True)

    rows = discover_summary(explicit, settings_data=settings_data)
    if cursor_passthrough_available():
        for model in cursor_catalog_models(force_refresh=refresh):
            rows.append(
                {
                    "slug": model.catalog_slug,
                    "model": model.upstream_id,
                    "display_name": model.display_name,
                    "description": f"{model.display_name} via Cursor subscription.",
                    "kind": "cursor",
                }
            )
    if chatgpt_passthrough_available():
        for slug, display_name in chatgpt_passthrough_display_names().items():
            rows.append(
                {
                    "slug": slug,
                    "model": slug,
                    "display_name": display_name,
                    "description": f"{display_name} via ChatGPT passthrough.",
                    "kind": "chatgpt",
                }
            )

    if not rows:
        print("No discoverable models. Add provider templates to ~/.codex-shim/models.json.")
        return 1

    width = max(len(row["slug"]) for row in rows)
    for row in sorted(rows, key=lambda item: (item["kind"], item["slug"])):
        print(
            f"{row['slug']:<{width}}  [{row['kind']}]  {row['display_name']}  ->  {row['model']}",
            flush=True,
        )
    print(f"\n{len(rows)} discoverable models.")
    return 0


def refresh_opencode_go(settings_path: Path, api_key_env: str, base_url: str, prefer: str, timeout: float) -> int:
    print(f"Refreshing OpenCode Go models from {base_url}...")
    try:
        result = refresh_opencode_go_settings(
            settings_path,
            api_key_env=api_key_env,
            base_url=base_url,
            prefer=prefer,
            timeout=timeout,
        )
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"Refreshed {len(result.models)} OpenCode Go models into {result.settings_path}.")
    if result.skipped:
        print(f"Skipped {len(result.skipped)} models with no working probed endpoint:")
        for model_id, chat_status, messages_status in result.skipped:
            print(f"  {model_id}: chat={chat_status}, messages={messages_status}")
    for row in result.models:
        print(f"  {row['slug']}  ->  {row['model']} ({row['provider']}, {row['opencode_go_endpoint']})")
    return 0


def install_codex_config(settings_path: Path, port: int, model_slug: str | None = None) -> None:
    models = _load_models(settings_path)
    router_config = _active_router(models, settings_path)
    default_slug = _resolve_model_slug(models, model_slug, router_config)
    CODEX_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    original = CODEX_CONFIG_PATH.read_text() if CODEX_CONFIG_PATH.exists() else ""
    original = _normalize_features_section_header(original)
    cleaned = _remove_managed_config(original)
    current_top_level = _extract_top_level_key_lines(cleaned, MANAGED_TOP_LEVEL_KEYS)
    if current_top_level:
        previous_top_level = current_top_level
    else:
        previous_top_level = _managed_previous_top_level(original)
    if not previous_top_level and CODEX_CONFIG_BACKUP_PATH.exists():
        previous_top_level = _extract_top_level_key_lines(CODEX_CONFIG_BACKUP_PATH.read_text(), MANAGED_TOP_LEVEL_KEYS)
    cleaned = _remove_top_level_keys(cleaned, MANAGED_TOP_LEVEL_KEYS)
    cleaned = _remove_section(cleaned, f"model_providers.{PROVIDER_NAME}")
    previous_features = _extract_section_key_lines(cleaned, MANAGED_FEATURES_SECTION, MANAGED_FEATURE_KEYS)
    if not previous_features:
        previous_features = _managed_previous_features(original)
    if not previous_features and CODEX_CONFIG_BACKUP_PATH.exists():
        previous_features = _extract_section_key_lines(
            CODEX_CONFIG_BACKUP_PATH.read_text(),
            MANAGED_FEATURES_SECTION,
            MANAGED_FEATURE_KEYS,
        )
    cleaned = _remove_keys_from_section(cleaned, MANAGED_FEATURES_SECTION, MANAGED_FEATURE_KEYS)
    managed_block = _managed_config_block(
        default_slug,
        port,
        previous_top_level,
        previous_features=previous_features,
    )
    cleaned = _apply_managed_features(cleaned)
    CODEX_CONFIG_PATH.write_text(managed_block + "\n" + cleaned.lstrip())
    print(f"Installed shim config into {CODEX_CONFIG_PATH}.")


def list_models(settings_path: Path) -> int:
    models = _load_models(settings_path)
    rows: list[tuple[str, str, str, str]] = []
    router_config = _active_router(models, settings_path)
    if router_config is not None:
        rows.append((router_config.slug, router_config.display_name, "per-task pick", "auto"))
    if chatgpt_passthrough_available():
        for slug, display_name in chatgpt_passthrough_display_names().items():
            rows.append((slug, display_name, slug, "chatgpt"))
    if cursor_passthrough_available():
        for model in cursor_catalog_models():
            rows.append((model.catalog_slug, model.display_name, model.upstream_id, "cursor-subscription"))
    rows.extend((model.slug, model.display_name, model.model, model.provider) for model in usable_byok_models(models))
    for model in models:
        if model not in usable_byok_models(models):
            rows.append((model.slug, f"{model.display_name} (missing API key)", model.model, model.provider))
    if not rows:
        print(
            "No models available. Create ~/.codex-shim/models.json, pass --settings /path/to/models.json, "
            "run `codex login` for GPT passthrough, or run `cursor-agent login` for Composer passthrough.",
            file=sys.stderr,
        )
        return 1
    width = max(len(row[0]) for row in rows)
    for slug, display_name, model, provider in rows:
        print(f"{slug:<{width}}  {display_name}  ->  {model} ({provider})", flush=True)
    return 0


def run_foreground(settings_path: Path, port: int) -> int:
    sync_desktop(settings_path, port, install_config=False)
    from .server import main as server_main

    server_main(
        [
            "--settings",
            str(Path(settings_path).expanduser()),
            "--host",
            DEFAULT_HOST,
            "--port",
            str(port),
        ]
    )
    return 0


SYSTEMD_USER_UNIT = Path.home() / ".config/systemd/user/codex-shim.service"


def _systemd_shell_command(codex_shim_bin: str, common_args: str, subcommand: str) -> str:
    command = f"{codex_shim_bin} {common_args} {subcommand}"
    if LOAD_ENV_SCRIPT.is_file():
        return f'/bin/bash -lc \'source "{LOAD_ENV_SCRIPT}" && exec {command}\''
    return command


def _systemd_unit_content(
    *,
    codex_shim_bin: str,
    settings_path: Path,
    port: int,
) -> str:
    settings = Path(settings_path).expanduser()
    common_args = f"--settings {settings} --port {port}"
    sync_cmd = _systemd_shell_command(codex_shim_bin, common_args, "sync-desktop")
    run_cmd = _systemd_shell_command(codex_shim_bin, common_args, "run")
    return f"""[Unit]
Description=Codex BYOK shim

[Service]
Type=simple
ExecStartPre={sync_cmd}
ExecStart={run_cmd}
Restart=on-failure
RestartSec=3

[Install]
WantedBy=graphical-session.target
"""


SYSTEMD_NETWORK_READY_DROPIN = (
    SYSTEMD_USER_UNIT.parent / "codex-shim.service.d" / "10-network-ready.conf"
)
NETWORK_READY_USER_UNIT = SYSTEMD_USER_UNIT.parent / "network-ready-user.service"


def _ensure_network_ready_dropin() -> None:
    if not NETWORK_READY_USER_UNIT.is_file():
        return
    SYSTEMD_NETWORK_READY_DROPIN.parent.mkdir(parents=True, exist_ok=True)
    if SYSTEMD_NETWORK_READY_DROPIN.is_file():
        return
    SYSTEMD_NETWORK_READY_DROPIN.write_text(
        """[Unit]
After=network-ready-user.service
Wants=network-ready-user.service
"""
    )


def install_service(settings_path: Path, port: int) -> int:
    if os.name == "nt":
        print("systemd user services are not supported on Windows.", file=sys.stderr)
        return 1
    codex_shim_bin = shutil.which("codex-shim")
    if not codex_shim_bin:
        print("codex-shim not found on PATH; install it before enabling the service.", file=sys.stderr)
        return 1
    stop()
    SYSTEMD_USER_UNIT.parent.mkdir(parents=True, exist_ok=True)
    SYSTEMD_USER_UNIT.write_text(
        _systemd_unit_content(
            codex_shim_bin=codex_shim_bin,
            settings_path=settings_path,
            port=port,
        )
    )
    _ensure_network_ready_dropin()
    for cmd in (
        ["systemctl", "--user", "daemon-reload"],
        ["systemctl", "--user", "enable", "--now", "codex-shim.service"],
    ):
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            print(f"Command failed: {' '.join(cmd)}", file=sys.stderr)
            return result.returncode
    linger = subprocess.run(
        ["loginctl", "enable-linger", os.environ.get("USER", "")],
        check=False,
        capture_output=True,
        text=True,
    )
    if linger.returncode != 0:
        print(
            "Service enabled for this login session. "
            "Run `loginctl enable-linger $USER` if you need it at boot without an active session.",
            file=sys.stderr,
        )
    print(f"Installed {SYSTEMD_USER_UNIT}")
    print("codex-shim.service is enabled and running.")
    return 0


def start(settings_path: Path, port: int) -> int:
    if _pid_running(_read_pid()):
        print(f"Shim already running with pid {_read_pid()}.")
        return 0
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    log = LOG_PATH.open("ab")
    cmd = [
        sys.executable,
        "-m",
        "codex_shim.server",
        "--settings",
        str(settings_path),
        "--host",
        DEFAULT_HOST,
        "--port",
        str(port),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    process = _popen_daemon(cmd, log, env)
    PID_PATH.write_text(str(process.pid))
    deadline = time.monotonic() + _STARTUP_HEALTH_WAIT_S
    while time.monotonic() < deadline:
        if _healthy(port):
            print(f"Shim started on http://{DEFAULT_HOST}:{port} with pid {process.pid}.")
            print(f"Log: {LOG_PATH}")
            return 0
        if process.poll() is not None:
            print(f"Shim exited during startup. See {LOG_PATH}.", file=sys.stderr)
            return 1
        time.sleep(_STARTUP_POLL_INTERVAL_S)
    print(f"Shim process started but health check timed out. See {LOG_PATH}.", file=sys.stderr)
    return 1


_STARTUP_HEALTH_WAIT_S = 30.0
_STARTUP_POLL_INTERVAL_S = 0.1
_STARTUP_HEALTH_TIMEOUT_S = 1.0
_SHUTDOWN_TERM_WAIT_S = 10.0
_SHUTDOWN_KILL_WAIT_S = 5.0
_SHUTDOWN_POLL_INTERVAL_S = 0.1
_PORT_FREE_WAIT_S = 10.0


def _wait_for_pid_exit(pid: int, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not _pid_running(pid):
            return True
        time.sleep(_SHUTDOWN_POLL_INTERVAL_S)
    return not _pid_running(pid)


def stop() -> int:
    pid = _read_pid()
    if not _pid_running(pid):
        print("Shim is not running.")
        PID_PATH.unlink(missing_ok=True)
        return 0
    port = DEFAULT_PORT
    _terminate_pid(pid)
    if _wait_for_pid_exit(pid, _SHUTDOWN_TERM_WAIT_S):
        _wait_for_port_free(port, _PORT_FREE_WAIT_S)
        PID_PATH.unlink(missing_ok=True)
        print("Shim stopped.")
        return 0
    if os.name != "nt":
        print(
            f"Shim pid {pid} did not exit after SIGTERM; sending SIGKILL.",
            file=sys.stderr,
        )
        try:
            os.killpg(pid, signal.SIGKILL)
        except OSError:
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
    else:
        _terminate_pid(pid)
    if _wait_for_pid_exit(pid, _SHUTDOWN_KILL_WAIT_S):
        _wait_for_port_free(port, _PORT_FREE_WAIT_S)
        PID_PATH.unlink(missing_ok=True)
        print("Shim stopped.")
        return 0
    print(f"Shim pid {pid} did not exit after SIGKILL.", file=sys.stderr)
    return 1


def restore_codex_config() -> None:
    if CODEX_CONFIG_PATH.exists():
        current = CODEX_CONFIG_PATH.read_text()
        current = _normalize_features_section_header(current)
        previous_top_level = _managed_previous_top_level(current)
        previous_features = _managed_previous_features(current)
        if not previous_top_level and CODEX_CONFIG_BACKUP_PATH.exists():
            previous_top_level = _extract_top_level_key_lines(CODEX_CONFIG_BACKUP_PATH.read_text(), MANAGED_TOP_LEVEL_KEYS)
        if not previous_features and CODEX_CONFIG_BACKUP_PATH.exists():
            previous_features = _extract_section_key_lines(
                CODEX_CONFIG_BACKUP_PATH.read_text(),
                MANAGED_FEATURES_SECTION,
                MANAGED_FEATURE_KEYS,
            )
        restored = _remove_managed_config(current)
        restored = _remove_section(restored, f"model_providers.{PROVIDER_NAME}")
        restored = _restore_missing_top_level_keys(restored.lstrip(), previous_top_level)
        restored = _restore_features_keys(restored, previous_features)
        if not _features_section_has_assignments(restored):
            restored = _remove_section(restored, MANAGED_FEATURES_SECTION)
        CODEX_CONFIG_PATH.write_text(restored)
        print(f"Removed shim config from {CODEX_CONFIG_PATH}.")
    if CODEX_CONFIG_BACKUP_PATH.exists():
        CODEX_CONFIG_BACKUP_PATH.unlink()
        print(f"Removed stale shim backup {CODEX_CONFIG_BACKUP_PATH}.")


def status(port: int) -> int:
    pid = _read_pid()
    if _pid_running(pid):
        health = _health(port)
        if health is not None:
            model_count = health.get("models", "unknown")
            print(f"Shim is running on http://{DEFAULT_HOST}:{port} with pid {pid} ({model_count} models).")
            return 0
    if _pid_running(pid):
        print(f"Shim process {pid} exists but health check failed.")
        return 1
    print("Shim is stopped.")
    return 1


def ensure_started(settings_path: Path, port: int) -> None:
    if not (_pid_running(_read_pid()) and _healthy(port)):
        code = start(settings_path, port)
        if code:
            raise SystemExit(code)


def exec_codex(settings_path: Path, port: int, codex_args: list[str]) -> None:
    overrides = _override_args(settings_path, port)
    codex_args = list(codex_args or [])
    if codex_args[:1] == ["--"]:
        codex_args = codex_args[1:]
    args = ["codex", *overrides, *codex_args]
    env = _with_loopback_no_proxy(os.environ.copy())
    if os.name == "nt":
        raise SystemExit(subprocess.call(args, env=env))
    os.execvpe("codex", args, env)


def exec_codex_app(settings_path: Path, port: int, path: str) -> None:
    _quit_codex_app()
    codex_app = patched_codex_app_bundle()
    if codex_app is not None:
        subprocess.Popen(["open", "-a", str(codex_app)], env=_with_loopback_no_proxy(os.environ.copy()))
    else:
        args = ["codex", "app", path]
        subprocess.Popen(args, env=_with_loopback_no_proxy(os.environ.copy()))
    _foreground_codex_app()


def _with_loopback_no_proxy(env: dict[str, str]) -> dict[str, str]:
    loopback = ["127.0.0.1", "localhost", "::1"]
    for key in ("NO_PROXY", "no_proxy"):
        values = [part.strip() for part in env.get(key, "").split(",") if part.strip()]
        lower_values = {value.lower() for value in values}
        for host in loopback:
            if host.lower() not in lower_values:
                values.append(host)
        env[key] = ",".join(values)
    return env


def _quit_codex_app() -> None:
    script = 'tell application "Codex" to if it is running then quit'
    try:
        subprocess.run(["osascript", "-e", script], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(1.0)
    except OSError:
        pass


def patch_codex_app() -> int:
    if sys.platform != "darwin":
        print("patch-app is macOS-only; Windows MSIX Codex Desktop cannot be patched with this ASAR helper.", file=sys.stderr)
        return 1
    codex_app = _codex_app_bundle_for_patch()
    app_asar = codex_app / "Contents/Resources/app.asar"
    info_plist = codex_app / "Contents/Info.plist"
    backup = RUNTIME_DIR / APP_ASAR_BACKUP_NAME
    info_backup = RUNTIME_DIR / INFO_PLIST_BACKUP_NAME

    if not app_asar.exists():
        print(f"Codex app bundle not found at {codex_app}.", file=sys.stderr)
        return 1
    if codex_app == USER_CODEX_APP:
        print(f"Patching user Codex copy at {codex_app}.")
    if not info_plist.exists():
        print(f"Codex Info.plist not found at {info_plist}.", file=sys.stderr)
        return 1
    if not _has_command("npx"):
        print("npx is required to patch the Electron asar bundle.", file=sys.stderr)
        return 1

    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    if not backup.exists():
        backup.write_bytes(app_asar.read_bytes())
        print(f"Backed up original app.asar to {backup}.")
    versioned_backup = RUNTIME_DIR / f"app.asar.before-codex-shim-model-picker-patch.{_app_asar_hash(app_asar)[:12]}"
    if not versioned_backup.exists():
        versioned_backup.write_bytes(app_asar.read_bytes())
        print(f"Backed up current app.asar to {versioned_backup}.")
    if not info_backup.exists():
        info_backup.write_bytes(info_plist.read_bytes())
        print(f"Backed up original Info.plist to {info_backup}.")

    _quit_codex_app()
    workdir = RUNTIME_DIR / "app-asar-work-user"
    if workdir.exists():
        import shutil

        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)

    subprocess.run(["npx", "--yes", "asar", "extract", str(app_asar), str(workdir)], check=True)
    changed = _patch_codex_desktop_bundles(workdir)
    if changed is None:
        return 1
    if changed:
        subprocess.run(["npx", "--yes", "asar", "pack", str(workdir), str(app_asar)], check=True)
        _update_app_asar_integrity(app_asar, info_plist)
        _resign_codex_app(codex_app)
    return 0


def restore_codex_app_bundle() -> int:
    if sys.platform != "darwin":
        print("restore-app is macOS-only; Windows MSIX Codex Desktop cannot be restored with this ASAR helper.", file=sys.stderr)
        return 1
    codex_app = patched_codex_app_bundle() or _codex_app_bundle_for_patch()
    app_asar = codex_app / "Contents/Resources/app.asar"
    info_plist = codex_app / "Contents/Info.plist"
    backup = RUNTIME_DIR / APP_ASAR_BACKUP_NAME
    info_backup = RUNTIME_DIR / INFO_PLIST_BACKUP_NAME
    if not backup.exists():
        print(f"No app.asar backup found at {backup}.")
        return 0
    _quit_codex_app()
    app_asar.write_bytes(backup.read_bytes())
    if info_backup.exists():
        info_plist.write_bytes(info_backup.read_bytes())
        print(f"Restored {info_plist} from {info_backup}.")
    elif info_plist.exists():
        _update_app_asar_integrity(app_asar, info_plist)
    _resign_codex_app(codex_app)
    print(f"Restored {app_asar} from {backup}.")
    return 0


def _has_command(command: str) -> bool:
    from shutil import which

    return which(command) is not None


def _app_asar_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _app_asar_header_hash(path: Path) -> str:
    with path.open("rb") as f:
        _, _, _, json_size = struct.unpack("<4I", f.read(16))
        header_json = f.read(json_size)
    return hashlib.sha256(header_json).hexdigest()


def _update_app_asar_integrity(app_asar: Path, info_plist: Path) -> None:
    header_hash = _app_asar_header_hash(app_asar)
    data = plistlib.loads(info_plist.read_bytes())
    try:
        data["ElectronAsarIntegrity"]["Resources/app.asar"]["hash"] = header_hash
    except KeyError as exc:
        raise RuntimeError(f"Could not update ElectronAsarIntegrity in {info_plist}") from exc
    info_plist.write_bytes(plistlib.dumps(data))
    print("Updated ElectronAsarIntegrity for app.asar.")


def _patch_codex_desktop_bundles(workdir: Path) -> bool | None:
    patches = [
        (
            "model picker allowlist filter",
            ["model-queries-*.js", "*.js"],
            MODEL_PICKER_NEEDLE,
            MODEL_PICKER_REPLACEMENT,
        ),
        (
            "shim-mode sidebar provider filter",
            ["app-server-manager-signals-*.js", "*.js"],
            SIDEBAR_RECENT_THREADS_NEEDLE,
            SIDEBAR_RECENT_THREADS_REPLACEMENT,
        ),
    ]
    changed = False
    for label, globs, needle, replacement in patches:
        bundle_file = _find_js_bundle(workdir, globs, needle, replacement)
        if bundle_file is None:
            print(f"Could not find the expected {label} in Codex Desktop.", file=sys.stderr)
            return None
        result = _replace_once(bundle_file, needle, replacement)
        if result is None:
            print(f"Could not patch the expected {label} in Codex Desktop.", file=sys.stderr)
            return None
        if result:
            changed = True
            print(f"Patched Codex Desktop {label}.")
        else:
            print(f"Codex Desktop {label} patch is already applied.")
    return changed


def _find_js_bundle(workdir: Path, globs: list[str], needle: str, replacement: str) -> Path | None:
    assets_dir = workdir / "webview" / "assets"
    if not assets_dir.exists():
        return None
    candidates: list[Path] = []
    for pattern in globs:
        candidates.extend(p for p in sorted(assets_dir.glob(pattern)) if p not in candidates)
    for path in candidates:
        text = _read_text_lossy(path)
        if needle in text or replacement in text:
            return path
    return None


def _replace_once(path: Path, needle: str, replacement: str) -> bool | None:
    text = _read_text_lossy(path)
    if replacement in text:
        return False
    count = text.count(needle)
    if count != 1:
        return None
    path.write_text(text.replace(needle, replacement, 1))
    return True


def _read_text_lossy(path: Path) -> str:
    try:
        return path.read_text()
    except UnicodeDecodeError:
        return path.read_text(errors="ignore")


def patched_codex_app_bundle() -> Path | None:
    for codex_app in (USER_CODEX_APP, SYSTEM_CODEX_APP):
        app_asar = codex_app / "Contents/Resources/app.asar"
        if app_asar.exists() and _app_asar_is_patched(app_asar):
            return codex_app
    return None


def _codex_app_bundle_for_patch() -> Path:
    system_asar = SYSTEM_CODEX_APP / "Contents/Resources/app.asar"
    if system_asar.exists() and _path_is_writable(system_asar):
        return SYSTEM_CODEX_APP
    return _ensure_user_codex_app()


def _ensure_user_codex_app() -> Path:
    user_asar = USER_CODEX_APP / "Contents/Resources/app.asar"
    if user_asar.exists():
        return USER_CODEX_APP
    system_asar = SYSTEM_CODEX_APP / "Contents/Resources/app.asar"
    if not system_asar.exists():
        raise SystemExit(f"Codex Desktop not found at {SYSTEM_CODEX_APP}.")
    USER_CODEX_APP.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ditto", str(SYSTEM_CODEX_APP), str(USER_CODEX_APP)], check=True)
    print(f"Copied Codex Desktop to {USER_CODEX_APP} for patching.")
    return USER_CODEX_APP


def _path_is_writable(path: Path) -> bool:
    try:
        with path.open("r+b"):
            return True
    except OSError:
        return False


def _app_asar_is_patched(app_asar: Path) -> bool:
    try:
        text = app_asar.read_bytes().decode("utf-8", errors="ignore")
    except OSError:
        return False
    return MODEL_PICKER_REPLACEMENT in text and SIDEBAR_RECENT_THREADS_REPLACEMENT in text


def _resign_codex_app(codex_app: Path = SYSTEM_CODEX_APP) -> None:
    # Electron validates app.asar through the bundle signature metadata at
    # startup. Re-sign after patching so the modified archive does not trip the
    # asar integrity check.
    subprocess.run(
        ["codesign", "--force", "--deep", "--sign", "-", str(codex_app)],
        check=True,
    )
    print(f"Re-signed {codex_app} after patch.")


def _foreground_codex_app() -> None:
    script = '''
tell application "Codex" to activate
delay 0.5
tell application "System Events"
  if exists process "Codex" then
    tell process "Codex"
      set frontmost to true
      if (count of windows) is 0 then
        keystroke "n" using command down
        delay 0.3
      end if
      if (count of windows) > 0 then
        set position of window 1 to {80, 60}
        set size of window 1 to {1400, 980}
      end if
    end tell
  end if
end tell
'''
    try:
        subprocess.run(["osascript", "-e", script], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError:
        pass


def _provider_display_name(models, slug: str, router_config=None) -> str:
    if router_config is not None and slug == router_config.slug:
        return router_config.display_name
    if chatgpt_passthrough_available():
        display_name = chatgpt_passthrough_display_names().get(slug)
        if display_name:
            return display_name
    if cursor_passthrough_available():
        display_name = cursor_passthrough_display_names().get(slug)
        if display_name:
            return display_name
    for model in models:
        if model.slug == slug:
            return model.display_name
    return "Codex Shim"


def migrate_threads_command(*, dry_run: bool = False) -> int:
    result = migrate_thread_providers(dry_run=dry_run)
    updated = int(result["updated"])
    databases = result["databases"]
    if not databases:
        print("No legacy codex_shim threads found.")
        return 0
    action = "Would migrate" if dry_run else "Migrated"
    for path, count in databases.items():
        print(f"{action} {count} thread(s) in {path}")
    print(f"{action} {updated} thread(s) total.")
    return 0


def _managed_config_block(
    default_slug: str,
    port: int,
    previous_top_level: dict[str, str] | None = None,
    previous_features: dict[str, str] | None = None,
) -> str:
    metadata = ""
    if previous_top_level:
        metadata += PREVIOUS_TOP_LEVEL_PREFIX + json.dumps(previous_top_level, sort_keys=True) + "\n"
    if previous_features:
        metadata += PREVIOUS_FEATURES_PREFIX + json.dumps(previous_features, sort_keys=True) + "\n"
    return f'''{MANAGED_BEGIN}
{metadata}model = "{_toml_escape(default_slug)}"
model_provider = "{PROVIDER_NAME}"
model_catalog_json = "{_toml_escape(str(DESKTOP_CATALOG_PATH))}"

[model_providers.{PROVIDER_NAME}]
name = "Codex Shim"
base_url = "http://127.0.0.1:{port}/v1"
wire_api = "responses"
requires_openai_auth = false
supports_websockets = false
{MANAGED_END}
'''


def _managed_config_blocks(
    default_slug: str,
    port: int,
    previous_top_level: dict[str, str] | None = None,
    provider_name: str = "Codex Shim",
    previous_features: dict[str, str] | None = None,
) -> tuple[str, str]:
    block = _managed_config_block(default_slug, port, previous_top_level, previous_features)
    return block, ""


def _managed_feature_block_lines() -> list[str]:
    return [
        MANAGED_BEGIN,
        f"{MANAGED_TOOL_SEARCH_FEATURE_KEY} = true",
        f"{MANAGED_REQUEST_COMPRESSION_FEATURE_KEY} = false",
        MANAGED_END,
    ]


def _apply_managed_features(text: str) -> str:
    """Insert shim-managed feature keys into the single user [features] section."""
    block = _managed_feature_block_lines()
    header = _section_header(MANAGED_FEATURES_SECTION)
    lines = text.splitlines()
    output: list[str] = []
    in_features = False
    inserted = False
    section_exists = header in {ln.strip() for ln in lines if ln.strip().startswith("[") and ln.strip().endswith("]")}
    for raw_line in lines:
        stripped = raw_line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            if in_features and not inserted:
                output.extend(block)
                inserted = True
            in_features = stripped == header
            output.append(raw_line)
            continue
        output.append(raw_line)
    if section_exists:
        if in_features and not inserted:
            output.extend(block)
        return "\n".join(output) + ("\n" if text.endswith("\n") else "")
    return text.rstrip() + "\n\n" + header + "\n" + "\n".join(block) + "\n"


def _normalize_features_section_header(text: str) -> str:
    header = _section_header(MANAGED_FEATURES_SECTION)
    return re.sub(rf"(\S){re.escape(header)}", rf"\1\n\n{header}", text)


def _features_section_has_assignments(text: str) -> bool:
    in_section = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_section = stripped == _section_header(MANAGED_FEATURES_SECTION)
            continue
        if not in_section:
            continue
        if stripped in {MANAGED_BEGIN, MANAGED_END}:
            continue
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        return True
    return False


def _remove_managed_config(text: str) -> str:
    while MANAGED_BEGIN in text:
        before, rest = text.split(MANAGED_BEGIN, 1)
        if MANAGED_END not in rest:
            return before
        _, after = rest.split(MANAGED_END, 1)
        text = before + after
    return text


def _stored_config_rhs(key: str, stored: str) -> str:
    """Return the TOML RHS for a stored value (value-only or legacy full assignment)."""
    stripped = stored.strip()
    if "=" in stripped:
        lhs, rhs = stripped.split("=", 1)
        if lhs.strip() == key:
            return rhs.strip()
    return stripped


def _format_top_level_assignment(key: str, stored: str) -> str:
    return f"{key} = {_stored_config_rhs(key, stored)}"


def _format_section_assignment(key: str, stored: str) -> str:
    return _format_top_level_assignment(key, stored)


def _remove_top_level_keys(text: str, keys: set[str]) -> str:
    lines = text.splitlines()
    output: list[str] = []
    in_top_level = True
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("["):
            in_top_level = False
        key = stripped.split("=", 1)[0].strip() if "=" in stripped else ""
        if in_top_level and key in keys:
            continue
        output.append(line)
    return "\n".join(output) + ("\n" if text.endswith("\n") else "")


def _extract_top_level_key_lines(text: str, keys: set[str]) -> dict[str, str]:
    found: dict[str, str] = {}
    in_top_level = True
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("["):
            in_top_level = False
        if not in_top_level or not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key = stripped.split("=", 1)[0].strip()
        if key in keys:
            found[key] = stripped.split("=", 1)[1].strip()
    return found


def _managed_previous_top_level(text: str) -> dict[str, str]:
    in_managed = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == MANAGED_BEGIN:
            in_managed = True
            continue
        if stripped == MANAGED_END:
            in_managed = False
            continue
        if in_managed and stripped.startswith(PREVIOUS_TOP_LEVEL_PREFIX):
            encoded = stripped[len(PREVIOUS_TOP_LEVEL_PREFIX) :]
            try:
                payload = json.loads(encoded)
            except json.JSONDecodeError:
                return {}
            if isinstance(payload, dict):
                return {str(k): str(v) for k, v in payload.items() if k in MANAGED_TOP_LEVEL_KEYS}
    return {}


def _managed_previous_features(text: str) -> dict[str, str]:
    in_managed = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == MANAGED_BEGIN:
            in_managed = True
            continue
        if stripped == MANAGED_END:
            in_managed = False
            continue
        if in_managed and stripped.startswith(PREVIOUS_FEATURES_PREFIX):
            encoded = stripped[len(PREVIOUS_FEATURES_PREFIX) :]
            try:
                payload = json.loads(encoded)
            except json.JSONDecodeError:
                return {}
            if isinstance(payload, dict):
                return {
                    str(k): str(v)
                    for k, v in payload.items()
                    if k in MANAGED_FEATURE_KEYS and str(v).strip()
                }
    return {}


def _restore_missing_top_level_keys(text: str, previous_top_level: dict[str, str]) -> str:
    if not previous_top_level:
        return text
    current = _extract_top_level_key_lines(text, MANAGED_TOP_LEVEL_KEYS)
    lines = [
        _format_top_level_assignment(key, previous_top_level[key])
        for key in MANAGED_TOP_LEVEL_RESTORE_ORDER
        if key in previous_top_level and key not in current
    ]
    if not lines:
        return text
    prefix = "\n".join(lines) + "\n"
    if text and not text.startswith("\n"):
        return prefix + text
    return prefix + text.lstrip()


def _remove_section(text: str, section: str) -> str:
    lines = text.splitlines()
    output: list[str] = []
    skipping = False
    header = f"[{section}]"
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            skipping = stripped == header
            if skipping:
                continue
        if not skipping:
            output.append(line)
    return "\n".join(output) + ("\n" if text.endswith("\n") else "")


def _section_header(section: str) -> str:
    return f"[{section}]"


def _extract_section_key_lines(text: str, section: str, keys: set[str]) -> dict[str, str]:
    found: dict[str, str] = {}
    in_section = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_section = stripped == _section_header(section)
            continue
        if not in_section or not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key = stripped.split("=", 1)[0].strip()
        if key in keys:
            found[key] = stripped.split("=", 1)[1].strip()
    return found


def _remove_keys_from_section(text: str, section: str, keys: set[str]) -> str:
    lines = text.splitlines()
    output: list[str] = []
    in_section = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_section = stripped == _section_header(section)
            output.append(line)
            continue
        if in_section and stripped and not stripped.startswith("#") and "=" in stripped:
            key = stripped.split("=", 1)[0].strip()
            if key in keys:
                continue
        output.append(line)
    return "\n".join(output) + ("\n" if text.endswith("\n") else "")


def _restore_features_keys(text: str, previous_features: dict[str, str]) -> str:
    if not previous_features:
        return text
    restored = text
    for key in MANAGED_FEATURE_KEYS:
        stored = previous_features.get(key)
        if not stored:
            continue
        restored = _upsert_key_in_section(
            restored,
            MANAGED_FEATURES_SECTION,
            key,
            _format_section_assignment(key, stored),
        )
    return restored


def _strip_trailing_blank_lines(lines: list[str]) -> None:
    while lines and not lines[-1].strip():
        lines.pop()


def _upsert_key_in_section(text: str, section: str, key: str, line: str) -> str:
    header = _section_header(section)
    lines = text.splitlines()
    output: list[str] = []
    in_section = False
    inserted = False
    section_exists = header in {ln.strip() for ln in lines if ln.strip().startswith("[") and ln.strip().endswith("]")}
    for raw_line in lines:
        stripped = raw_line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            if in_section and not inserted:
                _strip_trailing_blank_lines(output)
                output.append(line)
                inserted = True
            in_section = stripped == header
            output.append(raw_line)
            continue
        if in_section and stripped and not stripped.startswith("#") and "=" in stripped:
            current_key = stripped.split("=", 1)[0].strip()
            if current_key == key:
                output.append(line)
                inserted = True
                continue
        output.append(raw_line)
    if section_exists:
        if in_section and not inserted:
            _strip_trailing_blank_lines(output)
            output.append(line)
        return "\n".join(output) + ("\n" if text.endswith("\n") else "")
    prefix = "\n" if text and not text.endswith("\n") else ""
    return text.rstrip() + f"{prefix}{header}\n{line}\n"


def _popen_daemon(cmd: list[str], log, env: dict[str, str]) -> subprocess.Popen:
    kwargs = {"cwd": str(PROJECT_ROOT), "env": env, "stdout": log, "stderr": log}
    if os.name == "nt":
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(subprocess, "DETACHED_PROCESS", 0)
        return subprocess.Popen(cmd, creationflags=creationflags, **kwargs)
    return subprocess.Popen(cmd, start_new_session=True, **kwargs)


def _terminate_pid(pid: int) -> None:
    if os.name == "nt":
        handle = ctypes.windll.kernel32.OpenProcess(WINDOWS_PROCESS_TERMINATE, False, pid)
        if handle:
            try:
                ctypes.windll.kernel32.TerminateProcess(handle, 0)
            finally:
                ctypes.windll.kernel32.CloseHandle(handle)
        return
    try:
        os.killpg(pid, signal.SIGTERM)
    except OSError:
        os.kill(pid, signal.SIGTERM)


def _override_args(settings_path: Path, port: int) -> list[str]:
    models = _load_models(settings_path)
    try:
        default_slug = default_model_slug(models)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    pairs = codex_config_overrides(DESKTOP_CATALOG_PATH, default_slug, port)
    args: list[str] = []
    for pair in pairs:
        args.extend(["-c", pair])
    return args


def _resolve_model_slug(models, requested: str | None, router_config=None) -> str:
    if requested is None:
        current = _current_managed_model()
        if current in _valid_model_slugs(models, router_config):
            return current
        try:
            return default_model_slug(models)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    if router_config is not None and requested == router_config.slug:
        return requested
    if is_chatgpt_passthrough_slug(requested):
        if not chatgpt_passthrough_available():
            raise SystemExit(
                "ChatGPT passthrough requires a Codex login. "
                "Run `codex login` so ~/.codex/auth.json contains tokens.access_token."
            )
        return chatgpt_catalog_slug(requested)
    if is_cursor_passthrough_slug(requested):
        if not cursor_passthrough_available():
            raise SystemExit(
                "Composer passthrough requires Cursor CLI login. "
                "Run `cursor-agent login`, then `cursor-agent status`."
            )
        if requested in cursor_passthrough_display_names():
            return requested
        for model in cursor_catalog_models():
            if requested in {model.catalog_slug, model.upstream_id}:
                return model.catalog_slug
        return cursor_upstream_model(requested)
    by_slug = {model.slug: model.slug for model in models}
    by_model: dict[str, list[str]] = {}
    for model in models:
        by_model.setdefault(model.model, []).append(model.slug)
    if requested in by_slug:
        return requested
    configured = {model.slug: model for model in models}
    if requested in configured and not byok_model_has_credentials(configured[requested]):
        if is_cursor_passthrough_slug(requested):
            raise SystemExit(
                f"Model {requested!r} is configured for BYOK but has no API key. "
                "Remove it from ~/.codex-shim/models.json to use Cursor subscription passthrough, "
                "or set CURSOR_API_KEY / ~/.codex-shim/cursor-api-key."
            )
        raise SystemExit(
            f"Model {requested!r} is configured but has no API key. "
            "Set the provider API key in ~/.codex-shim/models.json or the matching env var."
        )
    if requested in by_model and len(by_model[requested]) == 1:
        return by_model[requested][0]
    matches = [model.slug for model in models if requested.lower() in model.display_name.lower()]
    if len(matches) == 1:
        return matches[0]
    if matches:
        raise SystemExit(f"Ambiguous model {requested!r}. Matches: {', '.join(matches)}")
    raise SystemExit(f"Unknown shim model {requested!r}. Run: codex-shim model list")


def _current_managed_model() -> str | None:
    if not CODEX_CONFIG_PATH.exists():
        return None
    in_managed = False
    for line in CODEX_CONFIG_PATH.read_text().splitlines():
        stripped = line.strip()
        if stripped == MANAGED_BEGIN:
            in_managed = True
            continue
        if stripped == MANAGED_END:
            in_managed = False
            continue
        if in_managed and stripped.startswith("model = "):
            return stripped.split("=", 1)[1].strip().strip('"')
    return None


def _valid_model_slugs(models, router_config=None) -> set[str]:
    slugs = {model.slug for model in usable_byok_models(models)}
    if router_config is not None:
        slugs.add(router_config.slug)
    if chatgpt_passthrough_available():
        slugs.update(chatgpt_passthrough_slugs())
    if cursor_passthrough_available():
        slugs.update(cursor_passthrough_display_names())
    return slugs


def _healthy(port: int) -> bool:
    return _health(port) is not None


def _port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex((DEFAULT_HOST, port)) != 0


def _wait_for_port_free(port: int, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if _port_is_free(port):
            return True
        time.sleep(_SHUTDOWN_POLL_INTERVAL_S)
    return _port_is_free(port)


def _health(port: int) -> dict | None:
    try:
        with urlopen(
            f"http://{DEFAULT_HOST}:{port}/health",
            timeout=_STARTUP_HEALTH_TIMEOUT_S,
        ) as response:
            if response.status != 200:
                return None
            payload = json.loads(response.read().decode("utf-8"))
            if not payload.get("ok"):
                return None
            return payload
    except Exception:
        return None


def _read_pid() -> int | None:
    try:
        return int(PID_PATH.read_text().strip())
    except Exception:
        return None


def _pid_running(pid: int | None) -> bool:
    if not pid:
        return False
    if os.name == "nt":
        handle = ctypes.windll.kernel32.OpenProcess(WINDOWS_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False
        try:
            exit_code = ctypes.c_ulong()
            if not ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return False
            return exit_code.value == WINDOWS_STILL_ACTIVE
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _entrypoint() -> int:
    try:
        return main()
    except BrokenPipeError:
        # Downstream pipe (e.g. `codex-shim list | head`) closed early. Mute the
        # interpreter's atexit flush so we exit cleanly instead of dumping a
        # traceback to stderr.
        try:
            sys.stdout.flush()
        except BrokenPipeError:
            pass
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        return 0


if __name__ == "__main__":
    raise SystemExit(_entrypoint())

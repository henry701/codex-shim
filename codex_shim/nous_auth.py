from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import fcntl
import json
import logging
import os
from pathlib import Path
from typing import Any, Iterator, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

_TOKEN_KEYS = ("agent_key", "access_token", "api_key")
DEFAULT_NOUS_PORTAL_URL = "https://portal.nousresearch.com"
DEFAULT_NOUS_INFERENCE_URL = "https://inference-api.nousresearch.com/v1"
DEFAULT_NOUS_CLIENT_ID = "hermes-cli"
_ALLOWED_PORTAL_HOSTS = frozenset({"portal.nousresearch.com"})

_startup_refresh_done = False


def hermes_home(*, environ: Mapping[str, str] | None = None) -> Path:
    env = os.environ if environ is None else environ
    override = str(env.get("HERMES_HOME") or "").strip()
    if override:
        return Path(override).expanduser()
    return Path.home() / ".hermes"


def resolve_nous_api_key(
    *,
    environ: Mapping[str, str] | None = None,
    hermes_dir: Path | None = None,
) -> str:
    """Nous Portal bearer: ``NOUS_API_KEY``, else Hermes OAuth JWT.

    Reads ``~/.hermes/auth.json`` and ``~/.hermes/shared/nous_auth.json``.
    Never returns ``refresh_token``. Startup refresh rotates that token on disk.
    """
    env = os.environ if environ is None else environ
    from_env = str(env.get("NOUS_API_KEY") or "").strip()
    if from_env:
        return from_env
    home = hermes_dir if hermes_dir is not None else hermes_home(environ=env)
    for path in (home / "auth.json", home / "shared" / "nous_auth.json"):
        token = _token_from_auth_file(path)
        if token:
            return token
    return ""


def reset_nous_oauth_startup_state() -> None:
    global _startup_refresh_done
    _startup_refresh_done = False


def refresh_nous_oauth_on_startup(
    *,
    hermes_dir: Path | None = None,
    environ: Mapping[str, str] | None = None,
    timeout: float = 20.0,
) -> bool:
    """Force-refresh Nous OAuth once per process. Safe to call from serve and sync."""
    global _startup_refresh_done
    if _startup_refresh_done:
        return True
    ok = refresh_nous_oauth(hermes_dir=hermes_dir, environ=environ, timeout=timeout)
    _startup_refresh_done = True
    return ok


def refresh_nous_oauth(
    *,
    hermes_dir: Path | None = None,
    environ: Mapping[str, str] | None = None,
    timeout: float = 20.0,
) -> bool:
    """Rotate Nous Portal tokens the way Hermes CLI does, then persist immediately.

    POST ``{portal}/api/oauth/token`` with ``grant_type=refresh_token`` and
    ``x-nous-refresh-token``. Refresh tokens are single-use: the new pair is
    written to ``auth.json`` and ``shared/nous_auth.json`` before return.
    """
    env = os.environ if environ is None else environ
    home = hermes_dir if hermes_dir is not None else hermes_home(environ=env)
    with _auth_lock(home):
        auth_path = home / "auth.json"
        shared_path = home / "shared" / "nous_auth.json"
        store = _read_json_object(auth_path)
        nous = _nous_state_from_store(store)
        if not str(nous.get("refresh_token") or "").strip():
            shared = _read_json_object(shared_path)
            if str(shared.get("refresh_token") or "").strip():
                nous = {**nous, **shared}
            else:
                logger.info("Nous OAuth refresh skipped: no refresh_token in %s", home)
                return False
        refresh_token = str(nous.get("refresh_token") or "").strip()
        client_id = str(nous.get("client_id") or DEFAULT_NOUS_CLIENT_ID).strip() or DEFAULT_NOUS_CLIENT_ID
        portal = _portal_base_url(nous, env)
        try:
            payload = _exchange_refresh_token(
                portal_base_url=portal,
                client_id=client_id,
                refresh_token=refresh_token,
                timeout=timeout,
            )
        except (OSError, URLError, HTTPError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
            logger.warning("Nous OAuth refresh failed: %s", exc)
            return False
        access_token = str(payload.get("access_token") or "").strip()
        if not access_token:
            logger.warning("Nous OAuth refresh returned no access_token")
            return False
        next_refresh = str(payload.get("refresh_token") or refresh_token).strip()
        ttl = _ttl_seconds(payload.get("expires_in"))
        now = datetime.now(timezone.utc)
        expires_at = (now + timedelta(seconds=ttl)).isoformat() if ttl else str(nous.get("expires_at") or "")
        updated = dict(nous)
        updated.update(
            {
                "access_token": access_token,
                "agent_key": access_token,
                "refresh_token": next_refresh,
                "token_type": str(payload.get("token_type") or updated.get("token_type") or "Bearer"),
                "scope": str(payload.get("scope") or updated.get("scope") or "inference:invoke"),
                "client_id": client_id,
                "portal_base_url": portal,
                "inference_base_url": str(
                    updated.get("inference_base_url") or DEFAULT_NOUS_INFERENCE_URL
                ).rstrip("/"),
                "obtained_at": now.isoformat(),
                "expires_in": ttl,
                "expires_at": expires_at,
                "agent_key_expires_at": expires_at,
                "agent_key_expires_in": ttl,
            }
        )
        if not isinstance(store, dict):
            store = {}
        providers = store.get("providers")
        if not isinstance(providers, dict):
            providers = {}
            store["providers"] = providers
        providers["nous"] = updated
        store["active_provider"] = "nous"
        _atomic_write_json(auth_path, store)
        _atomic_write_json(
            shared_path,
            {
                "_schema": 1,
                "access_token": access_token,
                "refresh_token": next_refresh,
                "token_type": updated["token_type"],
                "scope": updated["scope"],
                "client_id": client_id,
                "portal_base_url": portal,
                "inference_base_url": updated["inference_base_url"],
                "obtained_at": updated["obtained_at"],
                "expires_at": expires_at,
                "updated_at": now.isoformat(),
            },
        )
        logger.info("Refreshed Nous Portal OAuth tokens in %s", auth_path)
        return True


def _token_from_auth_file(path: Path) -> str:
    data = _read_json_object(path)
    return _token_from_mapping(data) if data else ""


def _token_from_mapping(data: dict[str, Any]) -> str:
    providers = data.get("providers")
    if isinstance(providers, dict):
        nous = providers.get("nous")
        if isinstance(nous, dict):
            token = _first_token(nous)
            if token:
                return token
    return _first_token(data)


def _first_token(row: dict[str, Any]) -> str:
    for key in _TOKEN_KEYS:
        token = str(row.get(key) or "").strip()
        if token:
            return token
    return ""


def _nous_state_from_store(store: dict[str, Any]) -> dict[str, Any]:
    providers = store.get("providers")
    if isinstance(providers, dict):
        nous = providers.get("nous")
        if isinstance(nous, dict):
            return dict(nous)
    if str(store.get("refresh_token") or "").strip():
        return dict(store)
    return {}


def _portal_base_url(nous: Mapping[str, Any], env: Mapping[str, str]) -> str:
    for key in ("HERMES_PORTAL_BASE_URL", "NOUS_PORTAL_BASE_URL"):
        override = str(env.get(key) or "").strip().rstrip("/")
        if override:
            return override
    stored = str(nous.get("portal_base_url") or "").strip().rstrip("/")
    host = (urlparse(stored).hostname or "").lower()
    if stored.startswith("https://") and host in _ALLOWED_PORTAL_HOSTS:
        return stored
    return DEFAULT_NOUS_PORTAL_URL


def _ttl_seconds(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _exchange_refresh_token(
    *,
    portal_base_url: str,
    client_id: str,
    refresh_token: str,
    timeout: float,
) -> dict[str, Any]:
    body = urlencode({"grant_type": "refresh_token", "client_id": client_id}).encode("utf-8")
    request = Request(
        f"{portal_base_url.rstrip('/')}/api/oauth/token",
        data=body,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
            "User-Agent": "HermesAgent/1.0",
            "x-nous-refresh-token": refresh_token,
        },
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Nous refresh response was not an object")
    return payload


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


@contextmanager
def _auth_lock(home: Path) -> Iterator[None]:
    home.mkdir(parents=True, exist_ok=True)
    lock_path = home / "auth.json.lock"
    fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        os.chmod(path, 0o600)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass

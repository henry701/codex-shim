from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import fcntl
import json
import logging
import os
from pathlib import Path
import time
from typing import Any, Iterator, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import urlopen

from .net.retry import RetryPolicy, request_urllib

logger = logging.getLogger(__name__)

_TOKEN_KEYS = ("agent_key", "access_token", "api_key")
DEFAULT_NOUS_PORTAL_URL = "https://portal.nousresearch.com"
DEFAULT_NOUS_INFERENCE_URL = "https://inference-api.nousresearch.com/v1"
DEFAULT_NOUS_CLIENT_ID = "hermes-cli"
_ALLOWED_PORTAL_HOSTS = frozenset({"portal.nousresearch.com"})
AUTH_LOCK_TIMEOUT_SECONDS = 15.0
_PERSIST_ATTEMPTS = 3


class AuthStoreUnreadable(Exception):
    def __init__(self, path: Path, reason: str):
        super().__init__(f"unreadable {path}: {reason}")
        self.path = path


@dataclass(frozen=True)
class _PendingPersist:
    auth_path: Path
    store: dict[str, Any]
    shared_path: Path
    shared: dict[str, Any]


_startup_refresh_done = False
_pending_persist: _PendingPersist | None = None


def hermes_home(*, environ: Mapping[str, str] | None = None) -> Path:
    env = os.environ if environ is None else environ
    override = str(env.get("HERMES_HOME") or "").strip()
    if override:
        return Path(override).expanduser()
    return Path.home() / ".hermes"


def shared_nous_auth_path(
    *,
    environ: Mapping[str, str] | None = None,
    hermes_dir: Path | None = None,
) -> Path:
    env = os.environ if environ is None else environ
    override = str(env.get("HERMES_SHARED_AUTH_DIR") or "").strip()
    if override:
        return Path(override).expanduser() / "nous_auth.json"
    home = hermes_dir if hermes_dir is not None else hermes_home(environ=env)
    return home / "shared" / "nous_auth.json"


def resolve_nous_api_key(
    *,
    environ: Mapping[str, str] | None = None,
    hermes_dir: Path | None = None,
) -> str:
    """Nous Portal bearer: ``NOUS_API_KEY``, else Hermes OAuth JWT.

    Reads ``~/.hermes/auth.json`` and the shared Nous store. Never returns
    ``refresh_token``. Startup refresh rotates that token on disk.
    """
    env = os.environ if environ is None else environ
    from_env = str(env.get("NOUS_API_KEY") or "").strip()
    if from_env:
        return from_env
    if _pending_persist is not None:
        _flush_pending_persist()
    pending = _pending_persist
    if pending is not None:
        token = _token_from_mapping(pending.store) or _first_token(pending.shared)
        if token:
            return token
    home = hermes_dir if hermes_dir is not None else hermes_home(environ=env)
    for path in (home / "auth.json", shared_nous_auth_path(environ=env, hermes_dir=home)):
        token = _token_from_auth_file(path)
        if token:
            return token
    return ""


def reset_nous_oauth_startup_state() -> None:
    global _startup_refresh_done, _pending_persist
    _startup_refresh_done = False
    _pending_persist = None


def refresh_nous_oauth_on_startup(
    *,
    hermes_dir: Path | None = None,
    environ: Mapping[str, str] | None = None,
    timeout: float = 20.0,
) -> bool:
    """Force-refresh Nous OAuth once per successful persist. Safe to call from serve and sync."""
    global _startup_refresh_done
    if _startup_refresh_done:
        return True
    ok = refresh_nous_oauth(hermes_dir=hermes_dir, environ=environ, timeout=timeout)
    if ok:
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
    written to ``auth.json`` and the shared Nous store before return. If HTTP
    succeeds and disk writes fail, the new pair is held in memory and retried
    without posting the old refresh token again.
    """
    global _pending_persist
    env = os.environ if environ is None else environ
    home = hermes_dir if hermes_dir is not None else hermes_home(environ=env)
    shared_path = shared_nous_auth_path(environ=env, hermes_dir=home)
    _refuse_live_hermes_paths_during_pytest(home, shared_path)
    if _pending_persist is not None:
        return _flush_pending_persist()
    try:
        with _auth_locks(home, shared_path):
            return _refresh_nous_oauth_locked(
                home=home,
                shared_path=shared_path,
                env=env,
                timeout=timeout,
            )
    except TimeoutError as exc:
        logger.warning("Nous OAuth refresh failed: %s", exc)
        return False


def _refresh_nous_oauth_locked(
    *,
    home: Path,
    shared_path: Path,
    env: Mapping[str, str],
    timeout: float,
) -> bool:
    auth_path = home / "auth.json"
    try:
        loaded = _load_json_object(auth_path)
    except AuthStoreUnreadable as exc:
        logger.error("Nous OAuth refresh refused: %s", exc)
        return False
    store = {} if loaded is None else dict(loaded)
    nous = _nous_state_from_store(store)
    if not str(nous.get("refresh_token") or "").strip():
        try:
            shared = _load_json_object(shared_path) or {}
        except AuthStoreUnreadable as exc:
            logger.error("Nous OAuth refresh refused: %s", exc)
            return False
        if str(shared.get("refresh_token") or "").strip():
            nous = {**nous, **shared}
        else:
            logger.info("Nous OAuth refresh skipped: no refresh_token in %s", home)
            return True
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
    returned_refresh = str(payload.get("refresh_token") or "").strip()
    if not access_token and not returned_refresh:
        logger.warning("Nous OAuth refresh returned no access_token")
        return False
    next_refresh = returned_refresh or refresh_token
    ttl = _ttl_seconds(payload.get("expires_in"))
    now = datetime.now(timezone.utc)
    expires_at = (now + timedelta(seconds=ttl)).isoformat() if ttl else str(nous.get("expires_at") or "")
    updated = dict(nous)
    if access_token:
        updated["access_token"] = access_token
        updated["agent_key"] = access_token
    updated.update(
        {
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
    providers = store.get("providers")
    if not isinstance(providers, dict):
        providers = {}
        store["providers"] = providers
    providers["nous"] = updated
    shared = {
        "_schema": 1,
        "access_token": str(updated.get("access_token") or ""),
        "refresh_token": next_refresh,
        "token_type": updated["token_type"],
        "scope": updated["scope"],
        "client_id": client_id,
        "portal_base_url": portal,
        "inference_base_url": updated["inference_base_url"],
        "obtained_at": updated["obtained_at"],
        "expires_at": expires_at,
        "updated_at": now.isoformat(),
    }
    if not _persist_rotated(auth_path, store, shared_path, shared):
        return False
    if not access_token:
        logger.warning("Nous OAuth refresh persisted a new refresh_token without access_token")
        return False
    logger.info("Refreshed Nous Portal OAuth tokens in %s", auth_path)
    return True


def _persist_rotated(
    auth_path: Path,
    store: dict[str, Any],
    shared_path: Path,
    shared: dict[str, Any],
) -> bool:
    global _pending_persist
    last_exc: OSError | None = None
    for _attempt in range(_PERSIST_ATTEMPTS):
        try:
            _atomic_write_json_pair(auth_path, store, shared_path, shared)
            _pending_persist = None
            return True
        except OSError as exc:
            last_exc = exc
    _pending_persist = _PendingPersist(
        auth_path=auth_path,
        store=store,
        shared_path=shared_path,
        shared=shared,
    )
    logger.error(
        "Nous OAuth refresh HTTP succeeded but persist failed; will retry disk write without reusing the old refresh_token: %s",
        last_exc,
    )
    return False


def _flush_pending_persist() -> bool:
    global _pending_persist
    pending = _pending_persist
    if pending is None:
        return True
    try:
        with _auth_locks(pending.auth_path.parent, pending.shared_path):
            if _persist_rotated(pending.auth_path, pending.store, pending.shared_path, pending.shared):
                logger.info("Persisted previously rotated Nous Portal OAuth tokens to %s", pending.auth_path)
                return True
    except TimeoutError as exc:
        logger.warning("Nous OAuth persist retry failed: %s", exc)
        return False
    return False


def _token_from_auth_file(path: Path) -> str:
    try:
        data = _load_json_object(path)
    except AuthStoreUnreadable:
        return ""
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
    result = request_urllib(
        f"{portal_base_url.rstrip('/')}/api/oauth/token",
        method="POST",
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
            "User-Agent": "HermesAgent/1.0",
            "x-nous-refresh-token": refresh_token,
        },
        data=body,
        timeout=timeout,
        policy=RetryPolicy(attempts=1, retry_json_decode=False),
        urlopen_fn=urlopen,
        sleep_fn=time.sleep,
        label="nous-oauth-refresh",
    )
    payload = json.loads(result.body.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Nous refresh response was not an object")
    return payload


def _load_json_object(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AuthStoreUnreadable(path, str(exc)) from exc
    if not isinstance(data, dict):
        raise AuthStoreUnreadable(path, "not a JSON object")
    return data


def _refuse_live_hermes_paths_during_pytest(home: Path, shared_path: Path) -> None:
    if not os.environ.get("PYTEST_CURRENT_TEST"):
        return
    live_home = (Path.home() / ".hermes").resolve()
    try:
        resolved_home = home.expanduser().resolve()
    except OSError:
        resolved_home = home
    live_shared = live_home / "shared" / "nous_auth.json"
    try:
        resolved_shared = shared_path.expanduser().resolve()
    except OSError:
        resolved_shared = shared_path
    if resolved_home == live_home or resolved_shared == live_shared:
        raise RuntimeError(
            f"refusing to touch live Hermes home during pytest: {home}. "
            "Set HERMES_HOME / HERMES_SHARED_AUTH_DIR to a tmp_path."
        )


@contextmanager
def _exclusive_file_lock(lock_path: Path, timeout: float = AUTH_LOCK_TIMEOUT_SECONDS) -> Iterator[None]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o600)
    deadline = time.monotonic() + timeout
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError as exc:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"timed out waiting for {lock_path}") from exc
                time.sleep(0.05)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass
        os.close(fd)


@contextmanager
def _auth_locks(home: Path, shared_path: Path) -> Iterator[None]:
    with _exclusive_file_lock((home / "auth.json").with_suffix(".lock")):
        with _exclusive_file_lock(shared_path.with_suffix(".lock")):
            yield


def _stage_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}.{time.monotonic_ns()}")
    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return tmp


def _commit_staged(tmp: Path, dest: Path) -> None:
    os.replace(tmp, dest)
    os.chmod(dest, 0o600)


def _atomic_write_json_pair(
    auth_path: Path,
    store: dict[str, Any],
    shared_path: Path,
    shared: dict[str, Any],
) -> None:
    auth_tmp = _stage_json(auth_path, store)
    try:
        shared_tmp = _stage_json(shared_path, shared)
    except Exception:
        if auth_tmp.exists():
            try:
                auth_tmp.unlink()
            except OSError:
                pass
        raise
    try:
        _commit_staged(auth_tmp, auth_path)
        last_exc: OSError | None = None
        for _attempt in range(_PERSIST_ATTEMPTS):
            try:
                _commit_staged(shared_tmp, shared_path)
                return
            except OSError as exc:
                last_exc = exc
        raise last_exc or OSError(f"failed to persist {shared_path}")
    finally:
        for tmp in (auth_tmp, shared_tmp):
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass


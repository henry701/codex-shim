from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import parse_qs

import pytest

from codex_shim import nous_auth
from codex_shim.discover import (
    NOUS_PORTAL_TEMPLATE,
    discover_byok_models,
    fetch_nous_portal_model_ids,
)
from codex_shim.nous_auth import (
    hermes_home,
    refresh_nous_oauth,
    refresh_nous_oauth_on_startup,
    reset_nous_oauth_startup_state,
    resolve_nous_api_key,
)
from codex_shim.settings import byok_model_has_credentials


pytestmark = [pytest.mark.nous_portal]


def _write_auth(path: Path, *, access: str = "", agent: str = "", api_key: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "active_provider": "nous",
                "providers": {
                    "nous": {
                        "access_token": access,
                        "agent_key": agent,
                        "api_key": api_key,
                        "refresh_token": "rt_must_not_be_used_as_bearer",
                        "inference_base_url": "https://inference-api.nousresearch.com/v1",
                    }
                },
            }
        )
    )


def test_hermes_home_prefers_env(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile"))
    assert hermes_home() == tmp_path / "profile"


def test_resolve_nous_api_key_prefers_env_over_hermes(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("NOUS_API_KEY", "sk-nous-from-env")
    _write_auth(tmp_path / "auth.json", access="eyJ-from-file")
    assert resolve_nous_api_key() == "sk-nous-from-env"


def test_resolve_nous_api_key_reads_hermes_auth_json(tmp_path, monkeypatch):
    monkeypatch.delenv("NOUS_API_KEY", raising=False)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _write_auth(tmp_path / "auth.json", access="eyJ-access", agent="eyJ-agent")
    assert resolve_nous_api_key() == "eyJ-agent"


def test_resolve_nous_api_key_falls_back_to_access_token(tmp_path, monkeypatch):
    monkeypatch.delenv("NOUS_API_KEY", raising=False)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _write_auth(tmp_path / "auth.json", access="eyJ-access-only")
    assert resolve_nous_api_key() == "eyJ-access-only"


def test_resolve_nous_api_key_reads_shared_nous_auth_json(tmp_path, monkeypatch):
    monkeypatch.delenv("NOUS_API_KEY", raising=False)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    shared = tmp_path / "shared" / "nous_auth.json"
    shared.parent.mkdir(parents=True)
    shared.write_text(json.dumps({"access_token": "eyJ-shared", "refresh_token": "rt_nope"}))
    assert resolve_nous_api_key() == "eyJ-shared"


def test_resolve_nous_api_key_never_returns_refresh_token(tmp_path, monkeypatch):
    monkeypatch.delenv("NOUS_API_KEY", raising=False)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _write_auth(tmp_path / "auth.json")
    assert resolve_nous_api_key() == ""


def test_fetch_nous_portal_model_ids_falls_back_to_ox_alpha(monkeypatch):
    monkeypatch.setattr(
        "codex_shim.discover.fetch_http_json",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("offline")),
    )
    assert fetch_nous_portal_model_ids(api_key="sk-nous-test") == ["stealth/ox-alpha"]


def test_fetch_nous_portal_model_ids_empty_without_credentials(monkeypatch):
    monkeypatch.setattr("codex_shim.discover.resolve_nous_api_key", lambda **_kwargs: "")
    monkeypatch.setattr(
        "codex_shim.discover.fetch_http_json",
        lambda *_args, **_kwargs: {"data": [{"id": "should-not-fetch"}]},
    )
    assert fetch_nous_portal_model_ids(api_key="") == []


def test_discover_nous_portal_ox_alpha_from_hermes_auth(tmp_path, monkeypatch):
    monkeypatch.delenv("NOUS_API_KEY", raising=False)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _write_auth(tmp_path / "auth.json", agent="eyJ-live")
    monkeypatch.setattr(
        "codex_shim.discover.fetch_http_json",
        lambda *_args, **_kwargs: {"data": [{"id": "stealth/ox-alpha"}]},
    )
    models = discover_byok_models(
        [],
        settings_data={
            "discover": {
                "zen_public": False,
                "zen": False,
                "openrouter_free": False,
                "nvidia_integrate": False,
                "nous": True,
            }
        },
    )
    route = next(model for model in models if model.model == "stealth/ox-alpha")
    assert route.slug == "nous-stealth-ox-alpha"
    assert route.base_url.rstrip("/") == "https://inference-api.nousresearch.com/v1"
    assert route.api_key == "eyJ-live"
    assert byok_model_has_credentials(route)
    assert route.extra_headers["HTTP-Referer"] == "https://hermes-agent.nousresearch.com"
    assert route.extra_headers["X-Title"] == "Hermes Agent"
    assert "User-Agent" not in route.extra_headers


def test_nous_portal_template_has_hermes_attribution_without_user_agent():
    assert NOUS_PORTAL_TEMPLATE.kind == "nous"
    assert NOUS_PORTAL_TEMPLATE.slug_prefix == "nous"
    assert NOUS_PORTAL_TEMPLATE.extra_headers["HTTP-Referer"] == "https://hermes-agent.nousresearch.com"
    assert NOUS_PORTAL_TEMPLATE.extra_headers["X-Title"] == "Hermes Agent"
    assert "User-Agent" not in NOUS_PORTAL_TEMPLATE.extra_headers
    assert "Authorization" not in NOUS_PORTAL_TEMPLATE.extra_headers


class _TokenResponse:
    def __init__(self, payload: dict, status: int = 200):
        self._payload = payload
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return json.dumps(self._payload).encode()


def test_refresh_nous_oauth_rotates_and_persists_tokens(tmp_path, monkeypatch):
    monkeypatch.delenv("NOUS_API_KEY", raising=False)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _write_auth(tmp_path / "auth.json", access="eyJ-old", agent="eyJ-old")
    captured: dict = {}

    def fake_urlopen(request, timeout=None):
        captured["url"] = request.full_url
        captured["headers"] = {key.lower(): value for key, value in request.header_items()}
        captured["body"] = request.data.decode() if request.data else ""
        return _TokenResponse(
            {
                "access_token": "eyJ-new",
                "refresh_token": "rt_new",
                "expires_in": 3600,
                "token_type": "Bearer",
                "scope": "inference:invoke",
            }
        )

    monkeypatch.setattr("codex_shim.nous_auth.urlopen", fake_urlopen)
    assert refresh_nous_oauth(hermes_dir=tmp_path) is True

    form = parse_qs(captured["body"])
    assert captured["url"] == "https://portal.nousresearch.com/api/oauth/token"
    assert captured["headers"]["x-nous-refresh-token"] == "rt_must_not_be_used_as_bearer"
    assert "authorization" not in captured["headers"]
    assert form["grant_type"] == ["refresh_token"]
    assert form["client_id"] == ["hermes-cli"]
    assert "refresh_token" not in form

    nous = json.loads((tmp_path / "auth.json").read_text())["providers"]["nous"]
    assert nous["access_token"] == "eyJ-new"
    assert nous["agent_key"] == "eyJ-new"
    assert nous["refresh_token"] == "rt_new"
    assert nous["expires_at"]
    shared = json.loads((tmp_path / "shared" / "nous_auth.json").read_text())
    assert shared["access_token"] == "eyJ-new"
    assert shared["refresh_token"] == "rt_new"
    assert (tmp_path / "auth.json").stat().st_mode & 0o777 == 0o600
    assert (tmp_path / "auth.lock").exists()
    assert (tmp_path / "shared" / "nous_auth.lock").exists()
    assert not (tmp_path / "auth.json.lock").exists()
    assert not (tmp_path / "shared" / "nous_auth.json.lock").exists()
    assert resolve_nous_api_key() == "eyJ-new"


def test_refresh_nous_oauth_skips_without_refresh_token(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    calls = {"n": 0}

    def boom(*_args, **_kwargs):
        calls["n"] += 1
        raise AssertionError("should not hit the token endpoint")

    (tmp_path / "auth.json").write_text(json.dumps({"providers": {"nous": {"access_token": "eyJ-only"}}}))
    monkeypatch.setattr("codex_shim.nous_auth.urlopen", boom)
    assert refresh_nous_oauth(hermes_dir=tmp_path) is True
    assert calls["n"] == 0


def test_refresh_nous_oauth_leaves_tokens_on_http_error(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _write_auth(tmp_path / "auth.json", access="eyJ-old")

    def fail(*_args, **_kwargs):
        raise OSError("portal down")

    monkeypatch.setattr("codex_shim.nous_auth.urlopen", fail)
    assert refresh_nous_oauth(hermes_dir=tmp_path) is False
    nous = json.loads((tmp_path / "auth.json").read_text())["providers"]["nous"]
    assert nous["access_token"] == "eyJ-old"
    assert nous["refresh_token"] == "rt_must_not_be_used_as_bearer"


def test_refresh_nous_oauth_on_startup_is_once_per_process(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _write_auth(tmp_path / "auth.json", access="eyJ-old")
    reset_nous_oauth_startup_state()
    calls = {"n": 0}

    def fake_urlopen(request, timeout=None):
        calls["n"] += 1
        return _TokenResponse({"access_token": f"eyJ-{calls['n']}", "refresh_token": f"rt_{calls['n']}", "expires_in": 60})

    monkeypatch.setattr("codex_shim.nous_auth.urlopen", fake_urlopen)
    assert refresh_nous_oauth_on_startup(hermes_dir=tmp_path) is True
    assert refresh_nous_oauth_on_startup(hermes_dir=tmp_path) is True
    assert calls["n"] == 1
    nous = json.loads((tmp_path / "auth.json").read_text())["providers"]["nous"]
    assert nous["refresh_token"] == "rt_1"


def test_refresh_nous_oauth_does_not_steal_active_provider(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "auth.json").write_text(
        json.dumps(
            {
                "active_provider": "openai",
                "providers": {
                    "openai": {"api_key": "sk-keep"},
                    "nous": {
                        "access_token": "eyJ-old",
                        "refresh_token": "rt_must_not_be_used_as_bearer",
                        "client_id": "hermes-cli",
                    },
                },
            }
        )
    )

    def fake_urlopen(request, timeout=None):
        return _TokenResponse({"access_token": "eyJ-new", "refresh_token": "rt_new", "expires_in": 60})

    monkeypatch.setattr("codex_shim.nous_auth.urlopen", fake_urlopen)
    assert refresh_nous_oauth(hermes_dir=tmp_path) is True
    store = json.loads((tmp_path / "auth.json").read_text())
    assert store["active_provider"] == "openai"
    assert store["providers"]["openai"] == {"api_key": "sk-keep"}
    assert store["providers"]["nous"]["refresh_token"] == "rt_new"


def test_refresh_nous_oauth_refuses_corrupt_auth_json(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "auth.json").write_text("{not-json")
    shared = tmp_path / "shared" / "nous_auth.json"
    shared.parent.mkdir(parents=True)
    shared.write_text(json.dumps({"refresh_token": "rt_shared", "access_token": "eyJ-shared"}))
    calls = {"n": 0}

    def boom(*_args, **_kwargs):
        calls["n"] += 1
        raise AssertionError("must not consume refresh_token when auth.json is unreadable")

    monkeypatch.setattr("codex_shim.nous_auth.urlopen", boom)
    assert refresh_nous_oauth(hermes_dir=tmp_path) is False
    assert calls["n"] == 0
    assert (tmp_path / "auth.json").read_text() == "{not-json"


def test_refresh_nous_oauth_on_startup_retries_after_http_error(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _write_auth(tmp_path / "auth.json", access="eyJ-old")
    reset_nous_oauth_startup_state()
    calls = {"n": 0}

    def flaky_urlopen(request, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("portal down")
        return _TokenResponse({"access_token": "eyJ-new", "refresh_token": "rt_new", "expires_in": 60})

    monkeypatch.setattr("codex_shim.nous_auth.urlopen", flaky_urlopen)
    assert refresh_nous_oauth_on_startup(hermes_dir=tmp_path) is False
    assert refresh_nous_oauth_on_startup(hermes_dir=tmp_path) is True
    assert calls["n"] == 2
    nous = json.loads((tmp_path / "auth.json").read_text())["providers"]["nous"]
    assert nous["refresh_token"] == "rt_new"


def test_refresh_retries_persist_after_http_success(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _write_auth(tmp_path / "auth.json", access="eyJ-old")
    calls = {"n": 0}
    writes = {"n": 0}
    real_write = nous_auth._atomic_write_json_pair

    def fake_urlopen(request, timeout=None):
        calls["n"] += 1
        return _TokenResponse({"access_token": "eyJ-new", "refresh_token": "rt_new", "expires_in": 60})

    def flaky_write(auth_path, store, shared_path, shared):
        writes["n"] += 1
        if writes["n"] == 1:
            raise OSError("ENOSPC")
        return real_write(auth_path, store, shared_path, shared)

    monkeypatch.setattr("codex_shim.nous_auth.urlopen", fake_urlopen)
    monkeypatch.setattr("codex_shim.nous_auth._atomic_write_json_pair", flaky_write)
    assert refresh_nous_oauth(hermes_dir=tmp_path) is True
    assert calls["n"] == 1
    nous = json.loads((tmp_path / "auth.json").read_text())["providers"]["nous"]
    assert nous["refresh_token"] == "rt_new"


def test_refresh_does_not_reexchange_when_persist_still_pending(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _write_auth(tmp_path / "auth.json", access="eyJ-old")
    reset_nous_oauth_startup_state()
    calls = {"n": 0}
    fail_writes = {"on": True}
    real_write = nous_auth._atomic_write_json_pair

    def fake_urlopen(request, timeout=None):
        calls["n"] += 1
        return _TokenResponse({"access_token": "eyJ-new", "refresh_token": "rt_new", "expires_in": 60})

    def maybe_write(auth_path, store, shared_path, shared):
        if fail_writes["on"]:
            raise OSError("ENOSPC")
        return real_write(auth_path, store, shared_path, shared)

    monkeypatch.setattr("codex_shim.nous_auth.urlopen", fake_urlopen)
    monkeypatch.setattr("codex_shim.nous_auth._atomic_write_json_pair", maybe_write)
    assert refresh_nous_oauth(hermes_dir=tmp_path) is False
    assert calls["n"] == 1
    assert json.loads((tmp_path / "auth.json").read_text())["providers"]["nous"]["refresh_token"] == (
        "rt_must_not_be_used_as_bearer"
    )
    fail_writes["on"] = False
    assert refresh_nous_oauth(hermes_dir=tmp_path) is True
    assert calls["n"] == 1
    nous = json.loads((tmp_path / "auth.json").read_text())["providers"]["nous"]
    assert nous["refresh_token"] == "rt_new"


def test_refresh_persists_rotated_refresh_token_without_access_token(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _write_auth(tmp_path / "auth.json", access="eyJ-old")

    def fake_urlopen(request, timeout=None):
        return _TokenResponse({"access_token": "", "refresh_token": "rt_rotated"})

    monkeypatch.setattr("codex_shim.nous_auth.urlopen", fake_urlopen)
    assert refresh_nous_oauth(hermes_dir=tmp_path) is False
    nous = json.loads((tmp_path / "auth.json").read_text())["providers"]["nous"]
    assert nous["refresh_token"] == "rt_rotated"


def test_resolve_and_refresh_honor_hermes_shared_auth_dir(tmp_path, monkeypatch):
    monkeypatch.delenv("NOUS_API_KEY", raising=False)
    home = tmp_path / "profile"
    shared_dir = tmp_path / "elsewhere"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_SHARED_AUTH_DIR", str(shared_dir))
    shared_dir.mkdir()
    (shared_dir / "nous_auth.json").write_text(json.dumps({"access_token": "eyJ-shared-override"}))
    assert resolve_nous_api_key() == "eyJ-shared-override"

    _write_auth(home / "auth.json", access="eyJ-old")

    def fake_urlopen(request, timeout=None):
        return _TokenResponse({"access_token": "eyJ-new", "refresh_token": "rt_new", "expires_in": 60})

    monkeypatch.setattr("codex_shim.nous_auth.urlopen", fake_urlopen)
    assert refresh_nous_oauth(hermes_dir=home) is True
    assert json.loads((shared_dir / "nous_auth.json").read_text())["refresh_token"] == "rt_new"
    assert not (home / "shared" / "nous_auth.json").exists()


def test_refresh_refuses_live_hermes_home_during_pytest(monkeypatch):
    live = Path.home() / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(live))
    calls = {"n": 0}

    def boom(*_args, **_kwargs):
        calls["n"] += 1
        raise AssertionError("must not refresh the live Hermes grant in pytest")

    monkeypatch.setattr("codex_shim.nous_auth.urlopen", boom)
    with pytest.raises(RuntimeError, match="live Hermes"):
        refresh_nous_oauth(hermes_dir=live)
    assert calls["n"] == 0


def test_refresh_retries_shared_commit_after_auth_json_lands(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _write_auth(tmp_path / "auth.json", access="eyJ-old")
    calls = {"n": 0}
    shared_fails = {"n": 0}
    real_commit = nous_auth._commit_staged

    def fake_urlopen(request, timeout=None):
        calls["n"] += 1
        return _TokenResponse({"access_token": "eyJ-new", "refresh_token": "rt_new", "expires_in": 60})

    def flaky_commit(tmp, dest):
        dest = Path(dest)
        if dest.name == "nous_auth.json" and shared_fails["n"] == 0:
            shared_fails["n"] += 1
            raise OSError("ENOSPC")
        return real_commit(tmp, dest)

    monkeypatch.setattr("codex_shim.nous_auth.urlopen", fake_urlopen)
    monkeypatch.setattr("codex_shim.nous_auth._commit_staged", flaky_commit)
    assert refresh_nous_oauth(hermes_dir=tmp_path) is True
    assert calls["n"] == 1
    assert shared_fails["n"] == 1
    assert json.loads((tmp_path / "auth.json").read_text())["providers"]["nous"]["refresh_token"] == "rt_new"
    assert json.loads((tmp_path / "shared" / "nous_auth.json").read_text())["refresh_token"] == "rt_new"


def test_resolve_uses_pending_access_token_when_persist_fails(tmp_path, monkeypatch):
    monkeypatch.delenv("NOUS_API_KEY", raising=False)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _write_auth(tmp_path / "auth.json", access="eyJ-old")
    fail_writes = {"on": True}
    real_write = nous_auth._atomic_write_json_pair

    def fake_urlopen(request, timeout=None):
        return _TokenResponse({"access_token": "eyJ-new", "refresh_token": "rt_new", "expires_in": 60})

    def maybe_write(auth_path, store, shared_path, shared):
        if fail_writes["on"]:
            raise OSError("ENOSPC")
        return real_write(auth_path, store, shared_path, shared)

    monkeypatch.setattr("codex_shim.nous_auth.urlopen", fake_urlopen)
    monkeypatch.setattr("codex_shim.nous_auth._atomic_write_json_pair", maybe_write)
    assert refresh_nous_oauth(hermes_dir=tmp_path) is False
    assert json.loads((tmp_path / "auth.json").read_text())["providers"]["nous"]["access_token"] == "eyJ-old"
    assert resolve_nous_api_key() == "eyJ-new"
    fail_writes["on"] = False
    assert resolve_nous_api_key() == "eyJ-new"
    assert json.loads((tmp_path / "auth.json").read_text())["providers"]["nous"]["refresh_token"] == "rt_new"

from __future__ import annotations

import json

from codex_shim.settings import ShimModel
from codex_shim.upstream_compat import (
    apply_openai_chat_compat,
    is_parallel_tool_calls_unsupported_error,
    learn_parallel_tool_calls_compat_if_needed,
    prepare_openai_chat_body,
    remember_parallel_tool_calls_unsupported,
    should_omit_parallel_tool_calls,
    supports_parallel_tool_calls_in_catalog,
)


def _route(**overrides) -> ShimModel:
    raw = overrides.pop("raw", {})
    base = {
        "slug": "oc-free-north-mini-code",
        "model": "north-mini-code",
        "display_name": "North Mini Code",
        "provider": "generic-chat-completion-api",
        "base_url": "https://opencode.ai/zen/v1",
    }
    base.update(overrides)
    return ShimModel(**base, raw=raw)


def test_is_parallel_tool_calls_unsupported_error_matches_opencode_message():
    assert is_parallel_tool_calls_unsupported_error(
        422,
        "unprocessable entity: parallel_tool_calls is not supported",
    )
    assert not is_parallel_tool_calls_unsupported_error(500, "internal error")


def test_apply_openai_chat_compat_strips_parallel_tool_calls_when_requested():
    body = {"model": "x", "messages": [], "parallel_tool_calls": True}
    out = apply_openai_chat_compat(body, omit_parallel_tool_calls=True)
    assert "parallel_tool_calls" not in out
    assert apply_openai_chat_compat(body, omit_parallel_tool_calls=False) is body


def test_remember_and_prepare_openai_chat_body(tmp_path):
    compat_path = tmp_path / "upstream-compat.json"
    route = _route()
    assert not should_omit_parallel_tool_calls(route, compat_path=compat_path)

    learned = remember_parallel_tool_calls_unsupported(
        route,
        reason="parallel_tool_calls is not supported",
        compat_path=compat_path,
    )
    assert learned is True
    assert should_omit_parallel_tool_calls(route, compat_path=compat_path)

    body = prepare_openai_chat_body(
        route,
        {"model": "north-mini-code", "messages": [], "parallel_tool_calls": True},
        compat_path=compat_path,
    )
    assert "parallel_tool_calls" not in body

    payload = json.loads(compat_path.read_text())
    assert payload["by_slug"]["oc-free-north-mini-code"]["omit_parallel_tool_calls"] is True
    assert payload["by_upstream_model"]["north-mini-code"]["omit_parallel_tool_calls"] is True


def test_learn_parallel_tool_calls_compat_if_needed_is_idempotent(tmp_path):
    compat_path = tmp_path / "upstream-compat.json"
    route = _route()
    message = "unprocessable entity: parallel_tool_calls is not supported"
    assert learn_parallel_tool_calls_compat_if_needed(route, 422, message, compat_path=compat_path)
    assert not learn_parallel_tool_calls_compat_if_needed(route, 422, message, compat_path=compat_path)


def test_explicit_model_raw_config_disables_parallel_tool_calls():
    route = _route(raw={"omit_parallel_tool_calls": True})
    body = prepare_openai_chat_body(route, {"parallel_tool_calls": False})
    assert "parallel_tool_calls" not in body
    assert supports_parallel_tool_calls_in_catalog(route) is False


def test_catalog_entry_disables_parallel_flag_when_learned(tmp_path, monkeypatch):
    from codex_shim import upstream_compat
    from codex_shim.catalog import catalog_entry

    compat_path = tmp_path / "upstream-compat.json"
    monkeypatch.setattr(upstream_compat, "DEFAULT_UPSTREAM_COMPAT_PATH", compat_path)
    route = _route()
    remember_parallel_tool_calls_unsupported(
        route,
        reason="parallel_tool_calls is not supported",
        compat_path=compat_path,
    )
    entry = catalog_entry(route)
    assert entry["supports_parallel_tool_calls"] is False

from __future__ import annotations

import json

from codex_shim.settings import ShimModel
from codex_shim.upstream_compat import (
    CONSOLE_CONTINUE_USER,
    apply_console_chat_compat,
    apply_openai_chat_compat,
    is_console_invalid_parameter_error,
    is_parallel_tool_calls_unsupported_error,
    learn_console_chat_compat_if_needed,
    learn_parallel_tool_calls_compat_if_needed,
    prepare_openai_chat_body,
    remember_parallel_tool_calls_unsupported,
    should_apply_console_chat_compat,
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


_CONSOLE_1210 = (
    "Error from provider (Console): Upstream request failed: "
    "[1210] Invalid API parameter, please check the documentation."
)
_CONSOLE_1214 = (
    "Error from provider (Console): Upstream request failed: "
    "[1214] The messages parameter is illegal. Please check the documentation."
)


def test_is_console_invalid_parameter_error_matches_zen_console_codes():
    assert is_console_invalid_parameter_error(400, _CONSOLE_1210)
    assert is_console_invalid_parameter_error(400, _CONSOLE_1214)
    assert is_console_invalid_parameter_error(422, "The messages parameter is illegal.")
    assert not is_console_invalid_parameter_error(400, "bad request")
    assert not is_console_invalid_parameter_error(500, _CONSOLE_1210)


def test_zen_public_routes_apply_console_chat_compat_proactively(tmp_path):
    zen_free = _route(raw={"discovered": True, "discover_kind": "zen_public"})
    zen_paid = _route(slug="zen-glm-5", model="glm-5", raw={"discover_kind": "zen"})
    local = _route(slug="local-llama", model="llama", raw={"discovered_local": True})
    missing = tmp_path / "missing-compat.json"
    assert should_apply_console_chat_compat(zen_free, compat_path=missing)
    assert should_apply_console_chat_compat(zen_paid, compat_path=missing)
    assert should_apply_console_chat_compat(_route(), compat_path=missing)
    assert not should_apply_console_chat_compat(local, compat_path=missing)


def test_apply_console_chat_compat_strips_glm_illegal_fields():
    body = {
        "model": "x-preview-f-free",
        "stream": True,
        "parallel_tool_calls": True,
        "reasoning_effort": "high",
        "messages": [
            {
                "role": "assistant",
                "content": None,
                "reasoning_content": "hidden think",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "index": 0,
                        "function": {"name": "exec_command", "arguments": "{\"cmd\":\"pwd\"}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "/home"},
        ],
    }
    out = apply_console_chat_compat(body, enabled=True)
    assert "parallel_tool_calls" not in out
    assert "reasoning_effort" not in out
    assert out["messages"][0]["content"] == ""
    assert "reasoning_content" not in out["messages"][0]
    assert "index" not in out["messages"][0]["tool_calls"][0]
    assert out["messages"][1]["name"] == "exec_command"
    assert out["messages"][-1] == {"role": "user", "content": CONSOLE_CONTINUE_USER}
    assert apply_console_chat_compat(body, enabled=False) is body


def test_apply_console_chat_compat_flattens_images_and_fills_empty_user():
    body = {
        "messages": [
            {"role": "system", "content": "rules"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "see"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
                ],
            },
        ]
    }
    out = apply_console_chat_compat(body, enabled=True)
    assert out["messages"][1]["content"] == "see\n[image omitted]"

    empty = apply_console_chat_compat({"messages": [{"role": "user", "content": ""}]}, enabled=True)
    assert empty["messages"] == [{"role": "user", "content": CONSOLE_CONTINUE_USER}]


def test_learn_console_chat_compat_retries_prepare(tmp_path):
    compat_path = tmp_path / "upstream-compat.json"
    route = _route(slug="byok-glm", model="glm-5", raw={})
    body = {
        "model": "glm-5",
        "parallel_tool_calls": True,
        "reasoning_effort": "high",
        "messages": [
            {"role": "assistant", "content": None, "reasoning_content": "think"},
            {"role": "user", "content": "go"},
        ],
    }
    prepared = prepare_openai_chat_body(route, body, compat_path=compat_path)
    assert prepared["messages"][0].get("reasoning_content") == "think"
    assert prepared["parallel_tool_calls"] is True

    assert learn_console_chat_compat_if_needed(route, 400, _CONSOLE_1210, compat_path=compat_path)
    assert not learn_console_chat_compat_if_needed(route, 400, _CONSOLE_1210, compat_path=compat_path)
    assert should_apply_console_chat_compat(route, compat_path=compat_path)

    cleaned = prepare_openai_chat_body(route, body, compat_path=compat_path)
    assert "reasoning_content" not in cleaned["messages"][0]
    assert cleaned["messages"][0]["content"] == ""
    assert "parallel_tool_calls" not in cleaned
    assert "reasoning_effort" not in cleaned

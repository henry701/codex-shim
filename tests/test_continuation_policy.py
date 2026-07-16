from __future__ import annotations

from codex_shim.continuation_policy import (
    ContinuationRoute,
    ContinuationSurface,
    is_previous_response_id_upstream_error,
    is_previous_response_id_upstream_event,
    should_expand_continuation,
)


def test_should_expand_always_for_http_byok():
    body = {"previous_response_id": "resp_1", "input": []}
    assert should_expand_continuation(
        surface=ContinuationSurface.HTTP,
        route=ContinuationRoute.BYOK,
        body=body,
    )


def test_should_expand_always_for_ws_byok():
    body = {"previous_response_id": "resp_1", "input": []}
    assert should_expand_continuation(
        surface=ContinuationSurface.WS,
        route=ContinuationRoute.BYOK,
        upstream_connection_reused=True,
        body=body,
    )


def test_should_expand_always_for_cursor():
    body = {"previous_response_id": "resp_1", "input": []}
    assert should_expand_continuation(
        surface=ContinuationSurface.HTTP,
        route=ContinuationRoute.CURSOR,
        body=body,
    )


def test_codex_ws_native_on_reused_connection():
    body = {"previous_response_id": "resp_1", "input": []}
    assert not should_expand_continuation(
        surface=ContinuationSurface.WS,
        route=ContinuationRoute.CHATGPT_CODEX,
        upstream_connection_reused=True,
        last_upstream_chained_response_id="resp_1",
        body=body,
    )


def test_codex_ws_expand_when_chain_broken_on_reused_connection():
    body = {"previous_response_id": "resp_2", "input": []}
    assert should_expand_continuation(
        surface=ContinuationSurface.WS,
        route=ContinuationRoute.CHATGPT_CODEX,
        upstream_connection_reused=True,
        last_upstream_chained_response_id="resp_1",
        body=body,
    )


def test_codex_ws_expand_on_new_connection():
    body = {"previous_response_id": "resp_1", "input": []}
    assert should_expand_continuation(
        surface=ContinuationSurface.WS,
        route=ContinuationRoute.CHATGPT_CODEX,
        upstream_connection_reused=False,
        body=body,
    )


def test_no_expand_without_previous_response_id():
    assert not should_expand_continuation(
        surface=ContinuationSurface.WS,
        route=ContinuationRoute.CHATGPT_CODEX,
        upstream_connection_reused=False,
        body={"input": []},
    )


def test_chatgpt_ws_force_expand_env(monkeypatch):
    from codex_shim.continuation_policy import (
        ContinuationRoute,
        ContinuationSurface,
        should_expand_continuation,
    )

    monkeypatch.setenv("CODEX_SHIM_CHATGPT_WS_FORCE_EXPAND", "1")
    body = {"previous_response_id": "resp_1", "input": []}
    assert should_expand_continuation(
        surface=ContinuationSurface.WS,
        route=ContinuationRoute.CHATGPT_CODEX,
        upstream_connection_reused=True,
        body=body,
    )


def test_is_previous_response_id_upstream_error():
    assert is_previous_response_id_upstream_error(
        "Unsupported parameter: previous_response_id"
    )
    assert is_previous_response_id_upstream_error(
        "Previous response with id 'resp_x' not found.",
        code="previous_response_not_found",
        param="previous_response_id",
    )
    assert not is_previous_response_id_upstream_error("rate limit exceeded")


def test_is_previous_response_id_upstream_event():
    event = {
        "type": "error",
        "error": {
            "type": "invalid_request_error",
            "code": "previous_response_not_found",
            "message": "Previous response with id 'resp_x' not found.",
            "param": "previous_response_id",
        },
        "status": 400,
    }
    assert is_previous_response_id_upstream_event(event)

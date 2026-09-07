from __future__ import annotations

from codex_shim.net.errors import (
    THROTTLE_NONE,
    THROTTLE_QUOTA,
    THROTTLE_RATE_LIMIT,
    THROTTLE_TRANSPORT,
    classify_throttle,
    classify_ws_event_throttle,
    is_quota_limit,
    parse_resets_in_seconds,
)


def test_classify_throttle_429_is_rate_limit_even_without_typed_body():
    assert classify_throttle(status=429, body="slow down") == THROTTLE_RATE_LIMIT


def test_classify_throttle_free_usage_limit_without_http_status():
    body = '{"error":{"type":"FreeUsageLimitError","message":"Rate limit exceeded"}}'
    assert classify_throttle(body=body) == THROTTLE_RATE_LIMIT


def test_classify_throttle_quota_beats_429_status():
    body = (
        '{"error":{"type":"usage_limit_reached","plan_type":"plus",'
        '"resets_in_seconds":90,"message":"You\'ve hit your usage limit"}}'
    )
    assert classify_throttle(status=429, body=body) == THROTTLE_QUOTA
    assert is_quota_limit(429, body) is True
    assert parse_resets_in_seconds(body) == 90.0


def test_classify_throttle_503_is_transport():
    assert classify_throttle(status=503, body="nope") == THROTTLE_TRANSPORT


def test_classify_throttle_400_is_none():
    body = '{"error":{"type":"invalid_request_error","message":"arguments must be valid JSON"}}'
    assert classify_throttle(status=400, body=body) == THROTTLE_NONE


def test_classify_ws_event_throttle_free_usage_limit_error():
    event = {
        "type": "error",
        "error": {"type": "FreeUsageLimitError", "message": "Rate limit exceeded"},
    }
    assert classify_ws_event_throttle(event) == THROTTLE_RATE_LIMIT


def test_classify_ws_event_throttle_nested_quota_on_response_failed():
    event = {
        "type": "response.failed",
        "response": {
            "error": {
                "type": "usage_limit_reached",
                "plan_type": "plus",
                "resets_in_seconds": 600000,
                "message": "You've hit your usage limit",
            }
        },
    }
    assert classify_ws_event_throttle(event) == THROTTLE_QUOTA


def test_classify_ws_event_throttle_ignores_invalid_request():
    event = {
        "type": "response.failed",
        "response": {
            "error": {
                "type": "invalid_request_error",
                "message": "arguments must be valid JSON",
            }
        },
    }
    assert classify_ws_event_throttle(event) == THROTTLE_NONE

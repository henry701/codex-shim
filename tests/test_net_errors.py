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


def test_classify_throttle_paid_model_credits_404_is_not_quota():
    body = (
        '{"status":404,"message":"Model requires available credits",'
        '"code":"insufficient_credits_for_paid_model"}'
    )
    assert is_quota_limit(404, body) is False
    assert classify_throttle(status=404, body=body) == THROTTLE_NONE


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


def test_classify_ws_event_throttle_ignores_rate_limits_updated():
    """ChatGPT Codex WS emits this as quota telemetry, not HTTP 429."""
    event = {
        "type": "rate_limits.updated",
        "rate_limits": [
            {"name": "requests", "limit": 10000, "remaining": 9999, "reset_seconds": 60},
        ],
    }
    assert classify_ws_event_throttle(event) == THROTTLE_NONE
    assert classify_throttle(body='{"type":"rate_limits.updated"}') == THROTTLE_NONE


def test_classify_throttle_rejects_type_that_only_contains_rate_limit():
    body = '{"error":{"type":"moderator_rate_limited_content","message":"blocked"}}'
    assert classify_throttle(body=body) == THROTTLE_NONE


def test_classify_throttle_rejects_rate_limiter_word_in_message():
    body = '{"error":{"message":"Corporate rate limiter policy applied"}}'
    assert classify_throttle(body=body) == THROTTLE_NONE


def test_classify_throttle_exact_error_type_rate_limit_error():
    body = '{"error":{"type":"rate_limit_error","code":"rate_limit_exceeded"}}'
    assert classify_throttle(body=body) == THROTTLE_RATE_LIMIT


def test_throttle_match_cause_names_exact_field():
    from codex_shim.net.errors import throttle_match_cause

    assert throttle_match_cause(status=429, body="slow down") == "http_status=429"
    assert (
        throttle_match_cause(
            body='{"error":{"type":"FreeUsageLimitError","message":"Rate limit exceeded"}}'
        )
        == "error.type=FreeUsageLimitError"
    )
    assert throttle_match_cause(body='{"type":"rate_limits.updated"}') is None
    assert throttle_match_cause(body='{"error":{"type":"moderator_rate_limited_content"}}') is None

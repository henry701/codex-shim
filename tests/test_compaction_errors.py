from __future__ import annotations

from types import SimpleNamespace

from codex_shim.compaction.errors import (
    describe_upstream_error,
    format_compaction_failure_detail,
    upstream_error_message,
)


def _opencode_error_response() -> SimpleNamespace:
    return SimpleNamespace(
        status=400,
        text=(
            '{"error":{"message":"Error from provider (Console): Upstream request failed",'
            '"type":"invalid_request_error","param":null,"code":"invalid_request_error"}}'
        ),
    )


def test_upstream_error_message_extracts_nested_openai_style_error():
    assert (
        upstream_error_message(_opencode_error_response())
        == "Error from provider (Console): Upstream request failed"
    )


def test_describe_upstream_error_includes_status_context_code_and_body():
    detail = describe_upstream_error(
        _opencode_error_response(),
        context="oc-free-deepseek-v4-flash-free → deepseek-v4-flash-free @ https://opencode.ai/zen/v1",
    )
    assert "[oc-free-deepseek-v4-flash-free → deepseek-v4-flash-free @ https://opencode.ai/zen/v1]" in detail
    assert "HTTP 400" in detail
    assert "code=invalid_request_error" in detail
    assert "Error from provider (Console): Upstream request failed" in detail
    assert "upstream_body=" in detail


def test_format_compaction_failure_detail_chains_native_and_summarization():
    native = describe_upstream_error(
        _opencode_error_response(),
        context="oc-free-deepseek-v4-flash-free → deepseek-v4-flash-free @ https://opencode.ai/zen/v1",
    )
    detail = format_compaction_failure_detail(
        slug="oc-free-deepseek-v4-flash-free",
        provider="byok",
        native_message=native,
        summarization_message=native,
        summarization_attempted=True,
        tertiary_slug=None,
    )
    assert "oc-free-deepseek-v4-flash-free" in detail
    assert "Native compact:" in detail
    assert "HTTP 400" in detail
    assert "upstream_body=" in detail
    assert "Summarization fallback:" in detail
    assert "Tertiary fallback: not configured" in detail
    assert "compaction.tertiary_fallback_slug" in detail


def test_format_compaction_failure_detail_tertiary_skip_reasons():
    no_creds = format_compaction_failure_detail(
        slug="codex-gpt-5-4-mini",
        provider="chatgpt",
        native_message="native failed",
        summarization_attempted=True,
        summarization_message="summarization failed",
        tertiary_skip_reason="no_credentials",
        tertiary_configured_slug="or-free-router",
    )
    assert "or-free-router" in no_creds
    assert "no API key" in no_creds

    route_err = format_compaction_failure_detail(
        slug="codex-gpt-5-4-mini",
        provider="chatgpt",
        native_message="native failed",
        summarization_attempted=True,
        summarization_message="summarization failed",
        tertiary_skip_reason="route_error",
        tertiary_configured_slug="missing-slug",
    )
    assert "route resolution failed" in route_err


def test_format_compaction_failure_detail_includes_tertiary_attempt():
    detail = format_compaction_failure_detail(
        slug="oc-free-deepseek-v4-flash-free",
        provider="byok",
        native_message="native failed",
        summarization_message="summarization failed",
        summarization_attempted=True,
        tertiary_slug="or-free-router",
        tertiary_message="tertiary failed",
        tertiary_attempted=True,
    )
    assert "Tertiary fallback (or-free-router): tertiary failed" in detail

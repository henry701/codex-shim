from __future__ import annotations

import sys
from typing import get_type_hints
from unittest.mock import MagicMock

from codex_shim.catalog_slugs import upstream_from_codex_catalog_slug
from codex_shim.compaction.adapters import compaction_request_from_v2
from codex_shim.compaction.config import CompactionSettings
from codex_shim.compaction.pipeline import PreparedInput
from codex_shim.compaction.strategies.bodies import (
    build_byok_compact_body,
    build_native_compact_body,
    build_summarization_compact_body,
)


def _prepared() -> PreparedInput:
    return PreparedInput(
        native_input=[{"type": "message", "role": "user", "content": "hi"}],
        summarization_input=[{"type": "message", "role": "assistant", "content": "ok"}],
        previous_summary="prior",
        excluded_user_turns=1,
        extra_context=["extra"],
    )


def test_upstream_from_codex_catalog_slug_passthrough_without_prefix():
    assert upstream_from_codex_catalog_slug("gpt-5.6-luna") == "gpt-5.6-luna"
    assert upstream_from_codex_catalog_slug("") == ""


def test_compaction_request_from_v2_uses_empty_models_when_catalog_load_fails(tmp_path):
    settings = tmp_path / "settings.json"
    settings.write_text("{}")
    server = MagicMock()
    server.settings.path = settings
    server.models_cached_or_load.side_effect = RuntimeError("catalog down")
    server._session_key.return_value = "sess"
    server._passthrough_fallback_slug.return_value = "fallback"
    request = compaction_request_from_v2(
        server,
        MagicMock(),
        {"model": "codex-gpt-5-6-luna"},
        [],
        provider="openai-responses",
        requested_slug="codex-gpt-5-6-luna",
    )
    assert request.requested_slug == "codex-gpt-5-6-luna"
    assert request.session_key == "sess"


def test_compaction_request_from_v2_transport_annotation_resolves():
    module = sys.modules["codex_shim.compaction.adapters"]
    namespace = dict(vars(module))
    namespace["ShimServer"] = object
    hints = get_type_hints(compaction_request_from_v2, globalns=namespace)
    assert "v2" in str(hints["transport"])
    assert "legacy_compact" in str(hints["transport"])


def test_build_native_compact_body_copies_optional_fields_and_client_instructions():
    settings = CompactionSettings(use_client_instructions_for_native=True)
    body = {
        "instructions": "stay brief",
        "tools": [{"type": "function", "name": "exec_command"}],
        "parallel_tool_calls": False,
        "reasoning": {"effort": "low"},
        "service_tier": "default",
        "text": {"format": {"type": "text"}},
    }
    compact = build_native_compact_body(
        _prepared(),
        body=body,
        upstream_model="gpt-5.4-mini",
        requested_slug="codex-gpt-5-6-luna",
        settings=settings,
        session_key="sess",
    )
    assert compact["model"] == "codex-gpt-5-6-luna"
    assert "stay brief" in compact["instructions"]
    assert compact["tools"] == body["tools"]
    assert compact["parallel_tool_calls"] is False
    assert compact["reasoning"] == {"effort": "low"}
    assert compact["service_tier"] == "default"
    assert compact["text"] == body["text"]
    assert compact["prompt_cache_key"]


def test_build_native_compact_body_ignores_non_string_client_instructions():
    settings = CompactionSettings(use_client_instructions_for_native=True)
    compact = build_native_compact_body(
        _prepared(),
        body={"instructions": ["not", "a", "string"]},
        upstream_model="upstream",
        requested_slug="slug",
        settings=settings,
    )
    assert compact["model"] == "slug"
    assert "instructions" in compact


def test_build_summarization_compact_body_appends_user_prompt_and_optional_fields():
    settings = CompactionSettings(summary_max_output_tokens=128)
    body = {
        "tools": [{"name": "exec_command"}],
        "parallel_tool_calls": True,
        "reasoning": {"effort": "medium"},
    }
    compact = build_summarization_compact_body(
        _prepared(),
        body=body,
        upstream_model="upstream",
        requested_slug="slug",
        settings=settings,
        stream=True,
    )
    assert compact["stream"] is True
    assert compact["max_output_tokens"] == 128
    assert compact["model"] == "slug"
    assert compact["tools"] == body["tools"]
    assert compact["parallel_tool_calls"] is True
    assert compact["input"][-1]["role"] == "user"


def test_build_byok_compact_body_summarization_path_and_native_path():
    settings = CompactionSettings(summary_max_output_tokens=64)
    prepared = _prepared()
    summarized = build_byok_compact_body(
        prepared,
        body={"model": "oc-free-ling-3-0-flash-fin-free", "tools": [1]},
        upstream_model="ling",
        for_summarization=True,
        settings=settings,
        session_key="k",
    )
    assert summarized["stream"] is False
    assert summarized["model"] == "oc-free-ling-3-0-flash-fin-free"
    native = build_byok_compact_body(
        prepared,
        body={"instructions": "x", "parallel_tool_calls": False, "tools": [1]},
        upstream_model="ling",
        for_summarization=False,
        settings=settings,
    )
    assert native["stream"] is False
    assert native["model"] == "ling"
    assert native["max_output_tokens"] == 64
    assert native["prompt_cache_key"]
    bare = build_byok_compact_body(
        prepared,
        body={},
        upstream_model="ling",
    )
    assert bare["max_output_tokens"] == 4096
    assert "prompt_cache_key" not in bare

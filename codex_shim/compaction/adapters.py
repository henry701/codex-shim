from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..settings import byok_model_has_credentials
from .config import load_compaction_settings
from .context import (
    compaction_budget_slug,
    context_window_tokens_for_slug,
    DEFAULT_COMPACTION_CATALOG_PATH,
)
from .model_resolver import CompactionModelResolver
from .orchestrator import (
    CompactionOrchestrator,
    CompactionOrchestratorError,
    enrich_orphan_upstream_warning,
    native_item_from_payload,
)
from .strategies.bodies import (
    build_byok_compact_body,
    build_native_compact_body,
    build_summarization_compact_body,
)
from .types import CompactionAdapters, CompactionRequest, NativeAttemptResult, SummarizationAttemptResult

if TYPE_CHECKING:
    from ..server import ShimServer


def build_server_compaction_adapters(server: ShimServer) -> CompactionAdapters:
    return CompactionAdapters(
        native_chatgpt=server._compaction_native_chatgpt,
        native_cursor=server._compaction_native_cursor,
        native_byok=server._compaction_native_byok,
        summarization_chatgpt=server._compaction_summarization_chatgpt,
        summarization_cursor=server._compaction_summarization_cursor,
        summarization_byok=server._compaction_summarization_byok,
        tertiary_byok=server._compaction_tertiary_byok,
        acquire_chatgpt_lock=server._compaction_acquire_chatgpt_lock,
    )


def compaction_orchestrator_for(server: ShimServer) -> CompactionOrchestrator:
    return CompactionOrchestrator(build_server_compaction_adapters(server))


def compaction_request_from_v2(
    server: ShimServer,
    request: Any,
    body: dict[str, Any],
    stripped_input: list[Any],
    *,
    provider: str,
    requested_slug: str,
    upstream_model: str | None = None,
    route: Any = None,
    tool_types: dict[str, str] | None = None,
    tool_resolve: dict[str, tuple[str | None, str]] | None = None,
    transport: Literal["v2", "legacy_compact"] = "v2",
    skip_native: bool = False,
    preset_native_message: str = "",
) -> CompactionRequest:
    settings = load_compaction_settings(server.settings.path)
    budget_slug = compaction_budget_slug(settings, requested_slug)
    try:
        models = server.settings.load()
    except Exception:
        models = []
    compaction_model_context_window = context_window_tokens_for_slug(
        budget_slug,
        byok_models=models,
        catalog_path=DEFAULT_COMPACTION_CATALOG_PATH,
    )
    return CompactionRequest(
        http_request=request,
        body=body,
        stripped_input=stripped_input,
        requested_slug=requested_slug,
        provider=provider,  # type: ignore[arg-type]
        session_key=server._session_key(request),
        transport=transport,  # type: ignore[arg-type]
        upstream_model=upstream_model,
        route=route,
        tool_types=tool_types,
        tool_resolve=tool_resolve,
        settings=settings,
        skip_native=skip_native,
        preset_native_message=preset_native_message,
        passthrough_fallback_slug=server._passthrough_fallback_slug(requested_slug),
        compaction_model_context_window=compaction_model_context_window,
        route_fn=server._route,
        has_credentials_fn=byok_model_has_credentials,
    )


__all__ = [
    "CompactionOrchestrator",
    "CompactionOrchestratorError",
    "build_server_compaction_adapters",
    "compaction_orchestrator_for",
    "compaction_request_from_v2",
    "enrich_orphan_upstream_warning",
    "native_item_from_payload",
    "CompactionModelResolver",
    "CompactionRequest",
    "CompactionAdapters",
    "NativeAttemptResult",
    "SummarizationAttemptResult",
    "build_byok_compact_body",
    "build_native_compact_body",
    "build_summarization_compact_body",
]

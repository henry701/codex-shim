from __future__ import annotations

from typing import Any

from .errors import describe_upstream_error, upstream_error_message
from .logging import log_compaction_fallback, log_compaction_phase, log_compaction_path, log_compaction_warnings
from .model_resolver import CompactionModelResolver, ResolvedCompactionModels
from .local import deterministic_fallback_summary, summary_is_usable
from .pipeline import PreparedInput, prepare_compaction_input
from .protocol import (
    apply_compaction_fallback_notice,
    compaction_item_from_response_payload,
    compaction_output_item,
    compaction_summary_from_output,
    decode_shim_compaction_summary,
    is_orphan_tool_call_upstream_error,
)
from .types import (
    CompactionAdapters,
    CompactionRequest,
    CompactionResult,
    NativeAttemptResult,
    SummarizationAttemptResult,
)


class CompactionOrchestrator:
    def __init__(self, adapters: CompactionAdapters) -> None:
        self._adapters = adapters

    async def run(self, request: CompactionRequest) -> CompactionResult:
        prepared = prepare_compaction_input(
            request.stripped_input,
            request.settings,
            client_instructions=(
                request.body.get("instructions")
                if isinstance(request.body.get("instructions"), str)
                else None
            ),
            compaction_model_context_window=request.compaction_model_context_window,
        )
        if prepared.warnings:
            log_compaction_warnings(prepared.warnings)
        log_compaction_phase(
            "post-prepare",
            items=len(prepared.native_input),
            summarization_items=len(prepared.summarization_input),
            rewritten=prepared.stats.get("rewritten_tool_outputs", 0),
            sanitization_dropped=prepared.stats.get("sanitization_dropped", 0),
            sanitization_preserved=prepared.stats.get("sanitization_preserved", 0),
        )

        if not prepared.native_input:
            native_message = "no compactable input remains after removing orphaned tool outputs"
            log_compaction_path(
                "sanitization_only",
                provider=request.provider,
                slug=request.requested_slug,
                native_items=0,
                summarization_items=0,
                native_error=native_message,
                dropped=prepared.stats.get("sanitization_dropped", 0),
                preserved=prepared.stats.get("sanitization_preserved", 0),
            )
            return CompactionResult(
                item=compaction_output_item(
                    apply_compaction_fallback_notice(
                        "No prior conversation state was available to compact.",
                        native_message,
                        provider=request.provider,
                        warnings=prepared.warnings or None,
                    )
                ),
                response_slug=request.requested_slug,
                warnings=list(prepared.warnings),
                native_error=native_message,
                phase="sanitization_only",
            )

        native_message = request.preset_native_message
        native = NativeAttemptResult(native_message=native_message)
        if not request.skip_native:
            native = await self._try_native(request, prepared)
            if native.item is not None:
                summary_chars = len(
                    decode_shim_compaction_summary(native.item.get("encrypted_content")) or ""
                )
                log_compaction_path(
                    "native",
                    provider=request.provider,
                    slug=request.requested_slug,
                    native_items=len(prepared.native_input),
                    summarization_items=len(prepared.summarization_input),
                    summary_chars=summary_chars,
                )
                return CompactionResult(
                    item=native.item,
                    usage=native.usage,
                    response_slug=request.requested_slug,
                    warnings=list(prepared.warnings),
                    phase="native",
                    legacy_payload=native.legacy_payload,
                )
            native_message = describe_upstream_error(
                native.error_response,
                context=native.upstream_context or None,
                fallback=native.native_message or "native compaction failed",
            )

        if not native_message:
            native_message = "native compaction failed"

        if not request.settings.fallback_enabled:
            raise CompactionOrchestratorError(
                native_message,
                error_response=native.error_response,
            )

        prepared_warnings = enrich_orphan_upstream_warning(native_message, prepared.warnings)
        if prepared_warnings != prepared.warnings:
            log_compaction_warnings(
                [w for w in prepared_warnings if w not in prepared.warnings]
            )

        summary_result = await self._try_summarization(request, prepared, native_message)
        accepted_summary = _accepted_summary(summary_result.summary, prepared)
        if summary_result.summary.strip() and not accepted_summary:
            log_compaction_phase(
                "summarization-unusable",
                provider=request.provider,
                slug=request.requested_slug,
                summary_chars=len(summary_result.summary.strip()),
            )
        if accepted_summary:
            log_compaction_path(
                "summarization",
                provider=request.provider,
                slug=request.requested_slug,
                native_items=len(prepared.native_input),
                summarization_items=len(prepared.summarization_input),
                summary_chars=len(accepted_summary),
                native_error=native_message,
            )
            return CompactionResult(
                item=compaction_output_item(
                    apply_compaction_fallback_notice(
                        accepted_summary,
                        native_message,
                        provider=request.provider,
                        warnings=prepared_warnings or None,
                    )
                ),
                usage=summary_result.usage,
                response_slug=request.requested_slug,
                warnings=list(prepared_warnings),
                native_error=native_message,
                phase="summarization",
            )

        summarization_message = describe_upstream_error(
            summary_result.error_response,
            context=summary_result.upstream_context or None,
            fallback="returned empty summary",
        )
        log_compaction_phase(
            "summarization-fail",
            provider=request.provider,
            slug=request.requested_slug,
            message=summarization_message,
        )

        tertiary_models = await self._resolve_models(request)
        tertiary = await self._try_tertiary(request, prepared, native_message, tertiary_models)
        accepted_tertiary = _accepted_summary(tertiary.summary, prepared)
        if tertiary.summary.strip() and not accepted_tertiary:
            log_compaction_phase(
                "tertiary-unusable",
                provider=request.provider,
                slug=request.requested_slug,
                summary_chars=len(tertiary.summary.strip()),
            )
        if accepted_tertiary:
            log_compaction_path(
                "tertiary",
                provider=request.provider,
                slug=request.requested_slug,
                native_items=len(prepared.native_input),
                summarization_items=len(prepared.summarization_input),
                summary_chars=len(accepted_tertiary),
                native_error=native_message,
            )
            return CompactionResult(
                item=compaction_output_item(
                    apply_compaction_fallback_notice(
                        accepted_tertiary,
                        native_message,
                        provider=request.provider,
                        warnings=prepared_warnings or None,
                    )
                ),
                usage=tertiary.usage,
                response_slug=request.requested_slug,
                warnings=list(prepared_warnings),
                native_error=native_message,
                phase="tertiary",
            )

        tertiary_message = describe_upstream_error(
            tertiary.error_response,
            context=tertiary.upstream_context or None,
            fallback="returned empty summary",
        )
        tertiary_slug = tertiary_models.tertiary_slug
        if tertiary_slug:
            log_compaction_phase(
                "tertiary-fail",
                provider=request.provider,
                slug=request.requested_slug,
                tertiary_slug=tertiary_slug,
                message=tertiary_message,
            )
        else:
            skip_fields: dict[str, object] = {
                "provider": request.provider,
                "slug": request.requested_slug,
                "reason": tertiary_models.tertiary_skip_reason or "not_configured",
            }
            if tertiary_models.tertiary_configured_slug:
                skip_fields["configured"] = tertiary_models.tertiary_configured_slug
            log_compaction_phase("tertiary-skip", **skip_fields)

        fallback = deterministic_fallback_summary(
            prepared.native_input,
            previous_summary=prepared.previous_summary,
            reason=summarization_message,
        )
        log_compaction_path(
            "local_fallback",
            provider=request.provider,
            slug=request.requested_slug,
            native_items=len(prepared.native_input),
            summarization_items=len(prepared.summarization_input),
            summary_chars=len(fallback),
            native_error=native_message,
        )
        fallback_warnings = list(prepared_warnings) + [
            "used deterministic local compaction fallback after unusable LLM summaries"
        ]
        return CompactionResult(
            item=compaction_output_item(
                apply_compaction_fallback_notice(
                    fallback,
                    native_message,
                    provider=request.provider,
                    warnings=fallback_warnings,
                )
            ),
            response_slug=request.requested_slug,
            warnings=fallback_warnings,
            native_error=native_message,
            phase="local_fallback",
        )

    async def _resolve_models(self, request: CompactionRequest) -> ResolvedCompactionModels:
        resolver = CompactionModelResolver(
            request.settings,
            route_fn=request.route_fn,
            has_credentials_fn=request.has_credentials_fn,
        )
        return await resolver.resolve(
            requested_slug=request.requested_slug,
            body=request.body,
            passthrough_fallback_slug=request.passthrough_fallback_slug,
        )

    async def _try_native(
        self,
        request: CompactionRequest,
        prepared: PreparedInput,
    ) -> NativeAttemptResult:
        log_compaction_phase("native-attempt", provider=request.provider)
        if request.provider == "chatgpt":
            return await self._adapters.native_chatgpt(request, prepared)
        if request.provider == "cursor":
            return await self._adapters.native_cursor(request, prepared)
        return await self._adapters.native_byok(request, prepared)

    async def _try_summarization(
        self,
        request: CompactionRequest,
        prepared: PreparedInput,
        native_message: str,
    ) -> SummarizationAttemptResult:
        log_compaction_fallback(
            request.requested_slug,
            provider=request.provider,
            native_status="fail",
            target="summarization",
            message=native_message,
        )
        if request.provider == "chatgpt":
            return await self._adapters.summarization_chatgpt(request, prepared, native_message)
        if request.provider == "cursor":
            return await self._adapters.summarization_cursor(request, prepared, native_message)
        return await self._adapters.summarization_byok(request, prepared, native_message)

    async def _try_tertiary(
        self,
        request: CompactionRequest,
        prepared: PreparedInput,
        native_message: str,
        resolved: ResolvedCompactionModels,
    ) -> SummarizationAttemptResult:
        if not resolved.tertiary_slug:
            return SummarizationAttemptResult()
        log_compaction_fallback(
            request.requested_slug,
            provider=request.provider,
            native_status="fail",
            target=f"tertiary:{resolved.tertiary_slug}",
            message=native_message,
        )
        return await self._adapters.tertiary_byok(
            request,
            prepared,
            native_message,
            resolved.tertiary_slug,
        )


class CompactionOrchestratorError(Exception):
    def __init__(self, message: str, *, error_response: Any = None) -> None:
        super().__init__(message)
        self.error_response = error_response


def _accepted_summary(summary: str, prepared: PreparedInput) -> str:
    text = (summary or "").strip()
    if summary_is_usable(
        text,
        items=prepared.native_input,
        previous_summary=prepared.previous_summary,
    ):
        return text
    return ""


def native_item_from_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    item = compaction_item_from_response_payload(payload)
    if item is not None:
        return item
    summary = compaction_summary_from_output(payload.get("output"))
    if summary.strip():
        return compaction_output_item(summary)
    return None


def enrich_orphan_upstream_warning(
    message: str,
    warnings: list[str],
) -> list[str]:
    if is_orphan_tool_call_upstream_error(message) and not warnings:
        return ["upstream rejected orphaned tool output(s); retrying via summarization fallback"]
    return list(warnings)

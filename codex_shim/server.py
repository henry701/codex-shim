from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
import re
import secrets
import sys
import time
import uuid
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Literal, Mapping
from urllib.parse import urljoin

from aiohttp import ClientError, ClientSession, ClientTimeout, WSMsgType, web

from .compaction import (
    CompactionTriggerError,
    SHIM_COMPACTION_PREFIX,
    apply_compaction_fallback_notice,
    compact_response_payload,
    compaction_item_from_response_payload,
    compaction_output_item,
    compaction_summary_from_output,
    decode_shim_compaction_summary,
    strip_terminal_compaction_trigger,
)
from .compaction.adapters import compaction_orchestrator_for, compaction_request_from_v2
from .compaction.logging import (
    log_compaction_cache_expansion,
    log_compaction_input_snapshot,
    log_compaction_upstream_body,
)
from .compaction.errors import byok_upstream_context
from .compaction.input_audit import summarize_compaction_input_items
from .compaction.model_resolver import CompactionModelResolver
from .compaction.orchestrator import (
    CompactionOrchestratorError,
    enrich_orphan_upstream_warning,
    native_item_from_payload,
)
from .compaction.pipeline import PreparedInput
from .compaction.strategies.bodies import (
    build_byok_compact_body,
    build_native_compact_body,
    build_summarization_compact_body,
)
from .compaction.types import CompactionRequest, NativeAttemptResult, SummarizationAttemptResult
from .cursor_bridge import (
    BridgeError,
    BridgeToolNotAllowedError,
    CursorBridgeSession,
    build_bridge_suffix,
    bridge_allowed_tools,
    cursor_bridge_enabled,
    cursor_bridge_registry,
    is_loopback_peer,
    shim_port_from_request_host,
)
from .cursor_passthrough import (
    CURSOR_MODEL_SLUG,
    CursorResponseCollector,
    build_cursor_prompt,
    cursor_passthrough_available,
    cursor_passthrough_display_names,
    cursor_upstream_model,
    is_cursor_passthrough_slug,
    iter_cursor_agent_events,
    resolve_cursor_workspace,
)
from .catalog import sort_catalog_entries
from .chatgpt_conversation_cache import ChatgptConversationCache, session_key_from_headers
from .responses_input_pipeline import apply_responses_input_pipeline_to_body
from . import router as router_module
from . import mcp_search
from . import tool_translate
from .header_passthrough import (
    apply_upstream_headers_to_response,
    chatgpt_passthrough_upstream_headers,
    chatgpt_passthrough_ws_upstream_headers,
    anthropic_upstream_headers,
    observe_upstream_response,
    openai_responses_ws_upstream_headers,
    openai_upstream_headers,
    prepare_downstream_sse_response,
    upstream_headers_from_response,
)
from .ws_passthrough import (
    CHATGPT_WS_URL,
    WsPassthroughConnectError,
    WsPassthroughSession,
    responses_websocket_url,
    ws_passthrough_enabled,
)
from .hostguard import build_allowed_hosts, host_guard_middleware
from .settings import (
    CHATGPT_MODEL_SLUG,
    DEFAULT_CHATGPT_CONVERSATIONS_DIR,
    DEFAULT_CODEX_AUTH,
    DEFAULT_SETTINGS,
    DEFAULT_HOST,
    DEFAULT_PORT,
    ModelSettings,
    ShimModel,
    available_model_slugs,
    chatgpt_passthrough_available,
    chatgpt_passthrough_display_names,
    chatgpt_passthrough_slugs,
    byok_model_has_credentials,
    chatgpt_upstream_model,
    is_chatgpt_passthrough_slug,
    usable_byok_models,
)
from .translate import (
    SHIM_ENCRYPTED_CONTENT_PREFIX,
    anthropic_messages_to_chat,
    anthropic_to_chat_response,
    anthropic_to_response,
    chat_completion_to_anthropic_message,
    chat_completion_to_response,
    chat_to_anthropic,
    normalize_responses_usage,
    prepare_codex_byok_responses_body,
    responses_to_anthropic,
    responses_to_chat,
    responses_tool_type_map,
    responses_tool_resolve_map,
    original_responses_tool_type,
    resolve_namespaced_tool_name,
    _chat_finish_to_anthropic_stop,
    _responses_usage_to_anthropic_usage,
)
from .upstream_compat import learn_parallel_tool_calls_compat_if_needed, prepare_openai_chat_body
from .upstream_io_trace import log_upstream_request, log_upstream_response, shim_io_log_enabled

DEBUG_DIR = Path(__file__).resolve().parents[1] / ".codex-shim"
CODEX_CONFIG_PATH = Path.home() / ".codex" / "config.toml"
PICKER_TOKEN_HEADER = "X-Codex-Shim-Picker-Token"
CHATGPT_ACCEPT_ENCODING = "zstd, gzip, deflate"
_MODELS_CACHE_TTL_SEC = 30.0
_STARTUP_REFRESH_TIMEOUT_SEC = 120.0


def _chatgpt_conversations_dir() -> Path:
    raw = os.environ.get("CODEX_SHIM_CHATGPT_CONVERSATIONS_DIR", "").strip()
    if raw:
        return Path(raw).expanduser()
    return DEFAULT_CHATGPT_CONVERSATIONS_DIR


def _chatgpt_expand_continuations_enabled() -> bool:
    """Expand delta-only continuations for ChatGPT passthrough.

    ChatGPT's Codex OAuth backend rejects ``previous_response_id`` (HTTP 400:
    "Unsupported parameter: previous_response_id"). When unset, expansion stays
    on. Set CODEX_SHIM_CHATGPT_EXPAND_CONTINUATIONS=0 to attempt native passthrough.
    """
    env = os.environ.get("CODEX_SHIM_CHATGPT_EXPAND_CONTINUATIONS", "")
    if env.lower() in {"0", "false", "no", "off"}:
        return False
    return True


def _chatgpt_passthrough_upstream_headers(
    request: web.Request,
    *,
    access_token: str,
    account_id: str,
    accept: str,
) -> dict[str, str]:
    return chatgpt_passthrough_upstream_headers(
        request.headers,
        access_token=access_token,
        account_id=account_id,
        accept=accept,
    )


class ShimServer:
    def __init__(self, settings_path: Path = DEFAULT_SETTINGS, host: str = DEFAULT_HOST):
        self.settings = ModelSettings(settings_path)
        self.host = host
        self.timeout = ClientTimeout(total=None, sock_connect=120, sock_read=None)
        self._models_cache: tuple[float, list[ShimModel]] | None = None
        self._health_snapshot: dict[str, Any] = {
            "ok": True,
            "models": 0,
            "chatgpt_passthrough": False,
            "cursor_passthrough": False,
            "auto_router": False,
        }
        self._chatgpt_conversation_cache = ChatgptConversationCache(_chatgpt_conversations_dir())
        self.picker_token = secrets.token_urlsafe(32)
        self._chatgpt_compaction_locks: dict[str, asyncio.Lock] = {}
        self._compact_timeout = ClientTimeout(total=300, sock_connect=120, sock_read=300)

    def _chatgpt_compaction_lock(self, session_key: str) -> asyncio.Lock:
        lock = self._chatgpt_compaction_locks.get(session_key)
        if lock is None:
            lock = asyncio.Lock()
            self._chatgpt_compaction_locks[session_key] = lock
        return lock

    def _session_key(self, request: web.Request) -> str:
        return session_key_from_headers(request.headers)

    async def _compaction_acquire_chatgpt_lock(self, request: CompactionRequest) -> asyncio.Lock:
        return self._chatgpt_compaction_lock(request.session_key)

    async def _post_chatgpt_native_compact(
        self,
        request: web.Request,
        compact_body: dict[str, Any],
        *,
        upstream_model: str,
        requested_slug: str,
    ) -> web.StreamResponse | web.Response:
        forwarded = _sanitize_chatgpt_compact_passthrough_body(compact_body)
        forwarded["model"] = upstream_model
        auth_path = DEFAULT_CODEX_AUTH.expanduser()
        try:
            auth = json.loads(auth_path.read_text())
        except FileNotFoundError:
            raise web.HTTPUnauthorized(text="~/.codex/auth.json not found")
        tokens = auth.get("tokens") or {}
        access_token = tokens.get("access_token")
        account_id = tokens.get("account_id") or ""
        if not access_token:
            raise web.HTTPUnauthorized(text="auth.json has no access_token")
        headers = _chatgpt_passthrough_upstream_headers(
            request,
            access_token=access_token,
            account_id=account_id,
            accept="application/json",
        )
        url = "https://chatgpt.com/backend-api/codex/responses/compact"
        _log_compaction_upstream_trace(phase="pre-native-compact", url=url, forwarded=forwarded)
        log_upstream_request("chatgpt-compact", url, forwarded)
        async with ClientSession(timeout=self._compact_timeout) as session:
            upstream = await session.post(url, json=forwarded, headers=headers)
            upstream_forward_headers = observe_upstream_response("chatgpt-compact-passthrough", upstream)
            if upstream.status >= 400:
                text = await upstream.text()
                status = upstream.status
                content_type = upstream.content_type or "text/plain"
                code, message = parse_upstream_error(text, status)
                print(f"[err] chatgpt-compact returned {status}: {message[:500]}", flush=True)
                _log_compaction_upstream_trace(
                    phase="native-compact-error",
                    url=url,
                    forwarded=forwarded,
                    status=status,
                    response_text=text,
                )
                log_upstream_response("chatgpt-compact", url, status, text, request_body=forwarded)
                upstream.release()
                return _upstream_text_response(
                    status,
                    text,
                    content_type=content_type,
                    upstream_headers=upstream_forward_headers,
                )
            payload = await upstream.json(content_type=None)
        usage = payload.get("usage") if isinstance(payload, dict) else None
        observe_upstream_response(
            "chatgpt-compact-passthrough",
            upstream,
            usage=usage if isinstance(usage, dict) else None,
        )
        _log_upstream_status(
            "chatgpt-compact",
            url,
            200,
            message=f"output_items={len(payload.get('output') or []) if isinstance(payload, dict) else 0}",
        )
        log_upstream_response(
            "chatgpt-compact",
            url,
            200,
            json.dumps(payload, default=str)[:12_000] if isinstance(payload, dict) else "",
            request_body=forwarded,
        )
        _rewrite_response_model(payload, requested_slug or None)
        response = web.json_response(payload)
        apply_upstream_headers_to_response(response, upstream_headers_from_response(upstream))
        upstream.release()
        return response

    async def _compaction_native_chatgpt(
        self,
        request: CompactionRequest,
        prepared: PreparedInput,
    ) -> NativeAttemptResult:
        upstream_model = request.upstream_model or chatgpt_upstream_model(request.requested_slug)
        compact_body = build_native_compact_body(
            prepared,
            body=request.body,
            upstream_model=upstream_model,
            requested_slug=request.requested_slug,
            settings=request.settings,
            session_key=request.session_key,
        )
        log_compaction_upstream_body(
            route="native-chatgpt",
            phase="request",
            input_items=compact_body.get("input") or [],
            model=request.requested_slug,
            sanitization_dropped=prepared.stats.get("sanitization_dropped", 0),
            sanitization_preserved=prepared.stats.get("sanitization_preserved", 0),
        )
        async with self._chatgpt_compaction_lock(request.session_key):
            response = await self._post_chatgpt_native_compact(
                request.http_request,
                compact_body,
                upstream_model=upstream_model,
                requested_slug=request.requested_slug,
            )
        if response.status >= 400:
            text = response.text or ""
            _, message = parse_upstream_error(text, response.status)
            merged_warnings = enrich_orphan_upstream_warning(message, prepared.warnings)
            for warning in merged_warnings:
                if warning not in prepared.warnings:
                    print(f"[warn] compaction: {warning}", flush=True)
            return NativeAttemptResult(
                native_status=response.status,
                native_message=message,
                error_response=response,
                upstream_context=(
                    f"{request.requested_slug} → {upstream_model} @ "
                    "https://chatgpt.com/backend-api/codex/responses/compact"
                ),
            )
        try:
            payload = json.loads(response.text or "{}")
        except json.JSONDecodeError:
            return NativeAttemptResult(error_response=response)
        if not isinstance(payload, dict):
            return NativeAttemptResult(error_response=response)
        item = native_item_from_payload(payload)
        if item is None:
            item = compaction_output_item("Compaction summary unavailable.")
        usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else None
        _log_upstream_status(
            "chatgpt-compact",
            "https://chatgpt.com/backend-api/codex/responses/compact",
            200,
        )
        return NativeAttemptResult(item=item, usage=usage, legacy_payload=payload)

    async def _compaction_native_cursor(
        self,
        request: CompactionRequest,
        prepared: PreparedInput,
    ) -> NativeAttemptResult:
        compact_body = build_native_compact_body(
            prepared,
            body=request.body,
            upstream_model=cursor_upstream_model(request.requested_slug),
            requested_slug=request.requested_slug,
            settings=request.settings,
            session_key=request.session_key,
        )
        compact_body["model"] = request.requested_slug
        response = await self._cursor_passthrough(
            request.http_request,
            compact_body,
            response_model_override=request.requested_slug,
            upstream_model=cursor_upstream_model(request.requested_slug),
            force_non_stream=True,
        )
        if response.status >= 400:
            text = response.text or ""
            _, message = parse_upstream_error(text, response.status)
            return NativeAttemptResult(
                native_status=response.status,
                native_message=message,
                error_response=response,
                upstream_context=(
                    f"{request.requested_slug} → {cursor_upstream_model(request.requested_slug)} "
                    "@ cursor-agent"
                ),
            )
        try:
            payload = json.loads(response.text or "{}")
        except json.JSONDecodeError:
            return NativeAttemptResult(error_response=response)
        if not isinstance(payload, dict):
            return NativeAttemptResult(error_response=response)
        item = native_item_from_payload(payload)
        usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else None
        return NativeAttemptResult(item=item, usage=usage)

    async def _compaction_native_byok(
        self,
        request: CompactionRequest,
        prepared: PreparedInput,
    ) -> NativeAttemptResult:
        route = request.route
        if route is None:
            route = await self._route(request.body)
        tool_types = request.tool_types or responses_tool_type_map(request.body.get("tools"))
        tool_resolve = request.tool_resolve or responses_tool_resolve_map(request.body.get("tools"))
        compact_body = build_byok_compact_body(
            prepared,
            body=request.body,
            upstream_model=route.model,
            settings=request.settings,
            session_key=request.session_key,
        )
        log_compaction_upstream_body(
            route="native-byok",
            phase="request",
            input_items=compact_body.get("input") or [],
            model=route.slug,
            sanitization_dropped=prepared.stats.get("sanitization_dropped", 0),
            sanitization_preserved=prepared.stats.get("sanitization_preserved", 0),
        )
        summary, usage, error_response = await self._fetch_byok_compact_summary(
            request.http_request,
            route,
            compact_body,
            tool_types=tool_types,
            tool_resolve=tool_resolve,
        )
        if error_response is not None:
            text = error_response.text if isinstance(error_response, web.Response) else ""
            status = error_response.status if isinstance(error_response, web.Response) else 502
            _, message = parse_upstream_error(text, status)
            return NativeAttemptResult(
                native_status=status,
                native_message=message,
                error_response=error_response,
                upstream_context=byok_upstream_context(route),
            )
        if summary.strip():
            return NativeAttemptResult(item=compaction_output_item(summary), usage=usage)
        return NativeAttemptResult(
            native_status=502,
            native_message="empty compaction summary from BYOK native compact",
            error_response=web.json_response(
                _responses_error_payload(route.slug, "upstream_error", "empty compaction summary"),
                status=502,
            ),
        )

    async def _compaction_summarization_chatgpt(
        self,
        request: CompactionRequest,
        prepared: PreparedInput,
        native_message: str,
    ) -> SummarizationAttemptResult:
        upstream_model = request.upstream_model or chatgpt_upstream_model(request.requested_slug)
        resolver = CompactionModelResolver(request.settings)
        resolved = resolver.resolve(requested_slug=request.requested_slug, body=request.body)
        summarization_slug = resolved.summarization_slug
        compact_body = build_summarization_compact_body(
            prepared,
            body=request.body,
            upstream_model=upstream_model,
            requested_slug=summarization_slug,
            settings=request.settings,
            session_key=request.session_key,
            stream=True,
        )
        log_compaction_upstream_body(
            route="summarization-chatgpt",
            phase="fallback-request",
            input_items=compact_body.get("input") or [],
            model=summarization_slug,
            sanitization_dropped=prepared.stats.get("sanitization_dropped", 0),
            sanitization_preserved=prepared.stats.get("sanitization_preserved", 0),
        )
        _log_compaction_upstream_trace(
            phase="pre-summarization-fallback",
            url="https://chatgpt.com/backend-api/codex/responses",
            forwarded=compact_body,
        )
        log_upstream_request("chatgpt-summarization-compact", "https://chatgpt.com/backend-api/codex/responses", compact_body)
        response = await self._chatgpt_passthrough(
            request.http_request,
            compact_body,
            response_model_override=summarization_slug,
            upstream_model=upstream_model,
            allow_byok_fallback=False,
            collect_stream=True,
        )
        return self._summarization_result_from_chatgpt_response(compact_body, response)

    def _summarization_result_from_chatgpt_response(
        self,
        compact_body: dict[str, Any],
        response: web.StreamResponse | web.Response,
    ) -> SummarizationAttemptResult:
        if not isinstance(response, web.Response) or response.status >= 400:
            text = response.text if isinstance(response, web.Response) else ""
            status = response.status if isinstance(response, web.Response) else None
            _log_compaction_upstream_trace(
                phase="summarization-fallback-error",
                url="https://chatgpt.com/backend-api/codex/responses",
                forwarded=compact_body,
                status=status,
                response_text=text,
            )
            return SummarizationAttemptResult(error_response=response)
        try:
            payload = json.loads(response.text or "{}")
        except json.JSONDecodeError:
            return SummarizationAttemptResult(error_response=response)
        if not isinstance(payload, dict):
            return SummarizationAttemptResult(error_response=response)
        summary = self._summary_from_compact_upstream_payload(payload)
        if not summary.strip():
            return SummarizationAttemptResult(error_response=response)
        usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else None
        cached = None
        if isinstance(usage, dict):
            details = usage.get("input_tokens_details")
            if isinstance(details, dict):
                cached = details.get("cached_tokens")
        _log_upstream_status(
            "chatgpt-summarization-compact",
            "https://chatgpt.com/backend-api/codex/responses",
            response.status,
            message=f"summary_chars={len(summary)} cached_tokens={cached}",
        )
        return SummarizationAttemptResult(summary=summary, usage=usage)

    async def _compaction_summarization_cursor(
        self,
        request: CompactionRequest,
        prepared: PreparedInput,
        native_message: str,
    ) -> SummarizationAttemptResult:
        resolver = CompactionModelResolver(request.settings)
        resolved = resolver.resolve(requested_slug=request.requested_slug, body=request.body)
        summarization_slug = resolved.summarization_slug
        compact_body = build_summarization_compact_body(
            prepared,
            body={**request.body, "model": summarization_slug},
            upstream_model=cursor_upstream_model(summarization_slug),
            requested_slug=summarization_slug,
            settings=request.settings,
            session_key=request.session_key,
            stream=False,
        )
        response = await self._cursor_passthrough(
            request.http_request,
            compact_body,
            response_model_override=summarization_slug,
            upstream_model=cursor_upstream_model(summarization_slug),
            force_non_stream=True,
        )
        if not isinstance(response, web.Response) or response.status >= 400:
            return SummarizationAttemptResult(error_response=response)
        try:
            payload = json.loads(response.text or "{}")
        except json.JSONDecodeError:
            return SummarizationAttemptResult(error_response=response)
        if not isinstance(payload, dict):
            return SummarizationAttemptResult(error_response=response)
        summary = compaction_summary_from_output(payload.get("output"))
        if not summary.strip():
            return SummarizationAttemptResult(error_response=response)
        usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else None
        return SummarizationAttemptResult(summary=summary, usage=usage)

    async def _compaction_summarization_byok(
        self,
        request: CompactionRequest,
        prepared: PreparedInput,
        native_message: str,
    ) -> SummarizationAttemptResult:
        route = request.route
        if route is None:
            route = await self._route(request.body)
        resolver = CompactionModelResolver(
            request.settings,
            route_fn=self._route,
            has_credentials_fn=byok_model_has_credentials,
        )
        resolved = resolver.resolve(requested_slug=request.requested_slug, body=request.body)
        summarization_slug = resolved.summarization_slug
        fallback_body = {**request.body, "model": summarization_slug}
        route = await self._route(fallback_body)
        tool_types = request.tool_types or responses_tool_type_map(request.body.get("tools"))
        tool_resolve = request.tool_resolve or responses_tool_resolve_map(request.body.get("tools"))
        compact_body = build_byok_compact_body(
            prepared,
            body=fallback_body,
            upstream_model=route.model,
            for_summarization=True,
            settings=request.settings,
            session_key=request.session_key,
        )
        log_compaction_upstream_body(
            route="summarization-byok",
            phase="fallback-request",
            input_items=compact_body.get("input") or [],
            model=route.slug,
            sanitization_dropped=prepared.stats.get("sanitization_dropped", 0),
            sanitization_preserved=prepared.stats.get("sanitization_preserved", 0),
        )
        summary, usage, error_response = await self._fetch_byok_compact_summary(
            request.http_request,
            route,
            compact_body,
            tool_types=tool_types,
            tool_resolve=tool_resolve,
        )
        if error_response is not None:
            return SummarizationAttemptResult(
                error_response=error_response,
                upstream_context=byok_upstream_context(route),
            )
        return SummarizationAttemptResult(summary=summary, usage=usage)

    async def _compaction_tertiary_byok(
        self,
        request: CompactionRequest,
        prepared: PreparedInput,
        native_message: str,
        tertiary_slug: str,
    ) -> SummarizationAttemptResult:
        fallback_body = {**request.body, "model": tertiary_slug, "input": prepared.summarization_input}
        route = await self._route(fallback_body)
        tool_types = request.tool_types or responses_tool_type_map(request.body.get("tools"))
        tool_resolve = request.tool_resolve or responses_tool_resolve_map(request.body.get("tools"))
        compact_body = build_byok_compact_body(
            prepared,
            body=fallback_body,
            upstream_model=route.model,
            for_summarization=True,
            settings=request.settings,
            session_key=request.session_key,
        )
        log_compaction_upstream_body(
            route="tertiary-byok",
            phase="fallback-request",
            input_items=compact_body.get("input") or [],
            model=route.slug,
            sanitization_dropped=prepared.stats.get("sanitization_dropped", 0),
            sanitization_preserved=prepared.stats.get("sanitization_preserved", 0),
        )
        summary, usage, error_response = await self._fetch_byok_compact_summary(
            request.http_request,
            route,
            compact_body,
            tool_types=tool_types,
            tool_resolve=tool_resolve,
        )
        if error_response is not None:
            return SummarizationAttemptResult(
                error_response=error_response,
                upstream_context=byok_upstream_context(route),
            )
        return SummarizationAttemptResult(summary=summary, usage=usage)

    async def _run_compaction_orchestrator(
        self,
        request: web.Request,
        body: dict[str, Any],
        stripped_input: list[Any],
        *,
        provider: Literal["chatgpt", "cursor", "byok"],
        requested_slug: str,
        upstream_model: str | None = None,
        route: ShimModel | None = None,
        tool_types: dict[str, str] | None = None,
        tool_resolve: dict[str, tuple[str | None, str]] | None = None,
        transport: Literal["v2", "legacy_compact"] = "v2",
        skip_native: bool = False,
        preset_native_message: str = "",
    ):
        compaction_request = compaction_request_from_v2(
            self,
            request,
            body,
            stripped_input,
            provider=provider,
            requested_slug=requested_slug,
            upstream_model=upstream_model,
            route=route,
            tool_types=tool_types,
            tool_resolve=tool_resolve,
            transport=transport,
            skip_native=skip_native,
            preset_native_message=preset_native_message,
        )
        orchestrator = compaction_orchestrator_for(self)
        return await orchestrator.run(compaction_request)

    def app(self) -> web.Application:
        allowed_hosts = build_allowed_hosts(self.host)
        app = web.Application(
            client_max_size=64 * 1024 * 1024,
            middlewares=[host_guard_middleware(allowed_hosts)],
        )
        app.on_startup.append(self._on_startup)
        app.router.add_get("/health", self.health)
        app.router.add_get("/v1/models", self.models)
        app.router.add_post("/v1/chat/completions", self.chat_completions)
        app.router.add_post("/v1/messages", self.anthropic_messages)
        app.router.add_get("/v1/responses", self.responses_websocket)
        app.router.add_post("/v1/responses", self.responses)
        app.router.add_post("/v1/responses/compact", self.responses_compact)
        app.router.add_get("/picker", self.picker_page)
        app.router.add_get("/api/models", self.api_models)
        app.router.add_post("/api/switch", self.switch_model)
        app.router.add_post("/_cursor_bridge/v1/invoke", self.cursor_bridge_invoke)
        return app

    async def _on_startup(self, _app: web.Application) -> None:
        try:
            models = await asyncio.wait_for(
                asyncio.to_thread(self.settings.load),
                timeout=_STARTUP_REFRESH_TIMEOUT_SEC,
            )
            self._models_cache = (time.monotonic(), models)
            self._health_snapshot = self._compute_health_snapshot_from_models(models)
        except Exception:
            try:
                self._health_snapshot = await asyncio.wait_for(
                    asyncio.to_thread(self._compute_health_snapshot),
                    timeout=_STARTUP_REFRESH_TIMEOUT_SEC,
                )
            except Exception:
                pass

    def _compute_health_snapshot_from_models(self, models: list[ShimModel]) -> dict[str, Any]:
        usable = usable_byok_models(models)
        chatgpt_ok = chatgpt_passthrough_available()
        cursor_ok = cursor_passthrough_available()
        passthrough_count = len(chatgpt_passthrough_slugs()) if chatgpt_ok else 0
        if cursor_ok:
            passthrough_count += len(cursor_passthrough_display_names())
        config = self.settings.load_router()
        auto_router = bool(
            config and router_module.router_is_active(config, available_model_slugs(models))
        )
        return {
            "ok": True,
            "models": len(usable) + passthrough_count,
            "chatgpt_passthrough": chatgpt_ok,
            "cursor_passthrough": cursor_ok,
            "auto_router": auto_router,
        }

    def _compute_health_snapshot(self) -> dict[str, Any]:
        return self._compute_health_snapshot_from_models(self.settings.load())

    async def _load_models(self) -> list[ShimModel]:
        now = time.monotonic()
        cached = self._models_cache
        if cached is not None and now - cached[0] < _MODELS_CACHE_TTL_SEC:
            return cached[1]
        models = await asyncio.to_thread(self.settings.load)
        self._models_cache = (now, models)
        return models

    async def picker_page(self, _request: web.Request) -> web.Response:
        return web.Response(text=_picker_html(self.picker_token), content_type="text/html")

    async def api_models(self, _request: web.Request) -> web.Response:
        current = _current_managed_model()
        data: list[dict[str, Any]] = []
        router_config = await self._active_router()
        if router_config is not None:
            data.append(
                {
                    "slug": router_config.slug,
                    "display_name": router_config.display_name,
                    "provider": "auto",
                    "active": current == router_config.slug,
                }
            )
        if chatgpt_passthrough_available():
            for slug, display_name in chatgpt_passthrough_display_names().items():
                data.append(
                    {
                        "slug": slug,
                        "display_name": display_name,
                        "provider": "chatgpt",
                        "active": current == slug,
                    }
                )
        if cursor_passthrough_available():
            for slug, display_name in cursor_passthrough_display_names().items():
                data.append(
                    {
                        "slug": slug,
                        "display_name": display_name,
                        "provider": "cursor",
                        "active": current == slug,
                    }
                )
        for m in usable_byok_models(await self._load_models()):
            data.append(
                {
                    "slug": m.slug,
                    "display_name": m.display_name,
                    "provider": m.provider,
                    "active": current == m.slug,
                }
            )
        return web.json_response(sort_catalog_entries(data))

    def _valid_picker_token(self, request: web.Request) -> bool:
        token = request.headers.get(PICKER_TOKEN_HEADER, "")
        return secrets.compare_digest(token, self.picker_token)

    async def switch_model(self, request: web.Request) -> web.Response:
        if not self._valid_picker_token(request):
            return web.json_response({"error": "forbidden"}, status=403)
        try:
            body = await request.json()
        except json.JSONDecodeError:
            return web.json_response({"error": "invalid JSON body"}, status=400)
        slug = str(body.get("slug") or "").strip()
        if not slug:
            return web.json_response({"error": "slug is required"}, status=400)
        models = usable_byok_models(await self._load_models())
        valid = {m.slug for m in models}
        display_for: dict[str, str] = {m.slug: m.display_name for m in models}
        router_config = await self._active_router()
        if router_config is not None:
            valid.add(router_config.slug)
            display_for[router_config.slug] = router_config.display_name
        if chatgpt_passthrough_available():
            valid.update(chatgpt_passthrough_slugs())
            display_for.update(chatgpt_passthrough_display_names())
        if cursor_passthrough_available():
            valid.update(cursor_passthrough_display_names())
            display_for.update(cursor_passthrough_display_names())
        if slug not in valid:
            return web.json_response({"error": f"unknown model: {slug}"}, status=404)
        _set_active_model(slug, display_for.get(slug, slug))
        restart = bool(body.get("restart_codex"))
        if restart:
            _restart_codex_app()
        return web.json_response({"ok": True, "model": slug, "restarted": restart})

    def _auto_router_active_now(self) -> bool:
        config = self.settings.load_router()
        if not config or not config.effective_enabled:
            return False
        cached = self._models_cache
        if cached is None:
            return bool(self._health_snapshot.get("auto_router"))
        return router_module.router_is_active(config, available_model_slugs(cached[1]))

    async def health(self, _request: web.Request) -> web.Response:
        snap = dict(self._health_snapshot)
        snap["auto_router"] = self._auto_router_active_now()
        return web.json_response(snap)

    async def models(self, _request: web.Request) -> web.Response:
        now = int(time.time())
        data: list[dict[str, Any]] = []
        router_config = await self._active_router()
        if router_config is not None:
            data.append(router_module.router_models_entry(router_config, now))
        if chatgpt_passthrough_available():
            data.extend(
                {"id": slug, "object": "model", "created": now, "owned_by": "chatgpt"}
                for slug in chatgpt_passthrough_slugs()
            )
        if cursor_passthrough_available():
            data.extend(
                {
                    "id": slug,
                    "object": "model",
                    "created": now,
                    "owned_by": "cursor",
                }
                for slug in cursor_passthrough_display_names()
            )
        data.extend(
            {
                "id": model.slug,
                "object": "model",
                "created": now,
                "owned_by": "codex-shim",
            }
            for model in usable_byok_models(await self._load_models())
        )
        return web.json_response({"object": "list", "data": sort_catalog_entries(data, slug_key="id")})

    async def chat_completions(self, request: web.Request) -> web.StreamResponse:
        body = await request.json()
        body = await self._maybe_apply_auto_router(body)
        route = await self._route(body)
        if route.is_openai_responses:
            raise web.HTTPBadGateway(
                text="openai-responses provider does not support /v1/chat/completions"
            )
        if route.is_openai_chat:
            forwarded = dict(body)
            forwarded["model"] = route.model
            if "messages" in forwarded:
                forwarded["messages"] = _normalize_roles(forwarded["messages"])
            return await self._post_openai_chat(request, route, forwarded, as_responses=False)
        if route.is_anthropic:
            forwarded = chat_to_anthropic(body, route.model, route.max_output_tokens)
            return await self._post_anthropic(request, route, forwarded, as_responses=False)
        raise web.HTTPBadGateway(text=f"Unsupported model provider: {route.provider}")

    async def anthropic_messages(self, request: web.Request) -> web.StreamResponse:
        body = await request.json()
        route = await self._route(body)
        if route.is_openai_chat:
            forwarded = anthropic_messages_to_chat(body, route.model, route.max_output_tokens)
            return await self._post_openai_chat_as_anthropic(request, route, forwarded)
        if route.is_anthropic:
            forwarded = dict(body)
            forwarded["model"] = route.model
            return await self._post_anthropic_messages(request, route, forwarded)
        raise web.HTTPBadGateway(text=f"Unsupported model provider: {route.provider}")

    async def responses_websocket(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(compress=True, heartbeat=30)
        await ws.prepare(request)
        passthrough: WsPassthroughSession | None = None
        async with ClientSession(timeout=self.timeout) as http_session:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    try:
                        payload = json.loads(msg.data)
                    except json.JSONDecodeError:
                        await _write_ws_error(ws, 400, "invalid_request_error", "invalid JSON websocket frame")
                        continue
                    if not isinstance(payload, dict):
                        await _write_ws_error(ws, 400, "invalid_request_error", "websocket frame must be a JSON object")
                        continue
                    if payload.get("type") != "response.create":
                        await _write_ws_error(
                            ws,
                            400,
                            "invalid_request_error",
                            "only response.create websocket frames are supported",
                        )
                        continue
                    body = {k: v for k, v in payload.items() if k != "type"}
                    if _shim_io_log_enabled() or _input_has_compaction_trigger(body.get("input")):
                        _log_client_request("/v1/responses/ws", body, transport="ws")
                    if await self._maybe_handle_ws_compaction_v2(request, ws, payload):
                        continue
                    target = await self._resolve_ws_passthrough_target(payload)
                    if target is not None and ws_passthrough_enabled():
                        if passthrough is None:
                            passthrough = WsPassthroughSession(client_session=http_session, client_ws=ws)
                        handled = await self._handle_ws_passthrough_response_create(
                            request,
                            passthrough,
                            payload,
                            target,
                        )
                        if handled:
                            continue
                    if target is not None and target.kind == "chatgpt":
                        await self._handle_chatgpt_response_create_websocket_http(
                            request,
                            ws,
                            payload,
                            target,
                            http_session=http_session,
                        )
                        continue
                    await self._handle_local_response_create_websocket(
                        request,
                        ws,
                        payload,
                        http_session=http_session,
                    )
                elif msg.type == WSMsgType.BINARY:
                    await _write_ws_error(ws, 400, "invalid_request_error", "binary websocket frames are not supported")
                elif msg.type == WSMsgType.ERROR:
                    break
            if passthrough is not None:
                await passthrough.close_upstream()
        return ws

    @dataclass(frozen=True)
    class _WsPassthroughTarget:
        kind: Literal["chatgpt", "openai_responses"]
        requested_slug: str
        upstream_model: str
        response_model_override: str | None
        route: ShimModel | None = None
        access_token: str | None = None
        account_id: str | None = None

    async def _resolve_ws_passthrough_target(self, payload: dict[str, Any]) -> _WsPassthroughTarget | None:
        requested = str(payload.get("model") or "")
        if is_chatgpt_passthrough_slug(requested):
            auth_path = DEFAULT_CODEX_AUTH.expanduser()
            try:
                auth = json.loads(auth_path.read_text())
            except (FileNotFoundError, json.JSONDecodeError):
                return None
            tokens = auth.get("tokens") or {}
            access_token = tokens.get("access_token")
            if not access_token:
                return None
            upstream_model = chatgpt_upstream_model(requested)
            return self._WsPassthroughTarget(
                kind="chatgpt",
                requested_slug=requested,
                upstream_model=upstream_model,
                response_model_override=requested if requested != upstream_model else None,
                access_token=access_token,
                account_id=tokens.get("account_id") or "",
            )
        body = await self._maybe_apply_auto_router(dict(payload))
        model = str(body.get("model") or requested)
        if is_cursor_passthrough_slug(model):
            return None
        if self._needs_image_gen(body) or self._needs_image_followup(body):
            return None
        try:
            route = await self._route(body)
        except web.HTTPException:
            return None
        if not route.is_openai_responses:
            return None
        return self._WsPassthroughTarget(
            kind="openai_responses",
            requested_slug=route.slug,
            upstream_model=route.model,
            response_model_override=route.slug if route.slug != route.model else None,
            route=route,
        )

    async def _handle_ws_passthrough_response_create(
        self,
        request: web.Request,
        passthrough: WsPassthroughSession,
        payload: dict[str, Any],
        target: _WsPassthroughTarget,
    ) -> bool:
        """Returns True when handled (including HTTP fallback). False to use local bridge."""
        if target.kind == "chatgpt":
            session_key = self._session_key(request)
            forwarded = self._prepare_chatgpt_passthrough_body(
                {k: v for k, v in payload.items() if k != "type"},
                session_key=session_key,
            )
            forwarded["model"] = target.upstream_model
            forwarded["stream"] = True
            headers = chatgpt_passthrough_ws_upstream_headers(
                request.headers,
                access_token=target.access_token or "",
                account_id=target.account_id or "",
            )
            _log_chatgpt_passthrough_trace(request, forwarded, headers, phase="pre-upstream-ws")
            upstream_url = CHATGPT_WS_URL
            source = "chatgpt-passthrough-ws"
            collector_factory = lambda fwd: ChatgptPassthroughResponseCollector(fwd)
            cache_enabled = _chatgpt_expand_continuations_enabled()
            store_cache = (
                (lambda response_id, items, *, terminal=False: self._store_chatgpt_passthrough_conversation(
                    session_key, response_id, items, terminal=terminal
                ))
                if cache_enabled
                else None
            )
        else:
            route = target.route
            assert route is not None
            body = prepare_codex_byok_responses_body(
                {k: v for k, v in payload.items() if k != "type"},
                request.headers,
            )
            forwarded = self._apply_responses_input_pipeline(body, request, context="turn")
            forwarded["model"] = route.model
            forwarded["stream"] = True
            headers = openai_responses_ws_upstream_headers(
                request.headers,
                api_key=route.api_key or None,
                extra_headers=route.extra_headers,
            )
            upstream_url = responses_websocket_url(route.base_url)
            source = f"byok-openai-responses-ws:{route.slug}"
            collector_factory = None
            cache_enabled = False
            store_cache = None

        try:
            if passthrough.upstream_ws is None or passthrough.upstream_ws.closed:
                await passthrough.connect_upstream(upstream_url, headers)
        except WsPassthroughConnectError as exc:
            print(f"[ws-passthrough] upstream connect failed; http-fallback url={upstream_url} err={exc}", flush=True)
            if target.kind == "chatgpt":
                await self._handle_chatgpt_response_create_websocket_http(
                    request,
                    passthrough.client_ws,
                    payload,
                    target,
                    http_session=passthrough.client_session,
                )
                return True
            return False

        collector = collector_factory(forwarded) if collector_factory else None
        cache_collected = store_cache if cache_enabled and store_cache is not None else None

        def on_event(event: dict[str, Any]) -> None:
            if collector is not None:
                collector.record(event)
                if cache_collected is not None and _should_cache_chatgpt_passthrough_event(event):
                    cache_collected(collector.response_id, collector.conversation_items(), terminal=False)

        await passthrough.send_response_create(forwarded)
        client_ws = passthrough.client_ws

        async def write_event(event: dict[str, Any]) -> None:
            await _write_ws_json(client_ws, event)

        await passthrough.relay_until_terminal(
            source=source,
            model_override=target.response_model_override,
            on_event=on_event if collector is not None else None,
            rewrite_model=_rewrite_response_model,
            write_event=write_event,
        )
        if collector is not None and cache_enabled and store_cache is not None:
            await self._store_chatgpt_passthrough_conversation_async(
                session_key,
                collector.response_id,
                collector.conversation_items(),
                terminal=True,
            )
        return True

    async def _handle_chatgpt_response_create_websocket_http(
        self,
        request: web.Request,
        ws: web.WebSocketResponse,
        payload: dict[str, Any],
        target: _WsPassthroughTarget,
        *,
        http_session: ClientSession,
    ) -> None:
        session_key = self._session_key(request)
        forwarded = self._prepare_chatgpt_passthrough_body(
            {k: v for k, v in payload.items() if k != "type"},
            session_key=session_key,
        )
        forwarded["model"] = target.upstream_model
        forwarded["stream"] = True
        headers = _chatgpt_passthrough_upstream_headers(
            request,
            access_token=target.access_token or "",
            account_id=target.account_id or "",
            accept="text/event-stream",
        )
        _log_chatgpt_passthrough_trace(request, forwarded, headers, phase="pre-upstream-ws-http-fallback")
        url = "https://chatgpt.com/backend-api/codex/responses"
        print(f"[ws-passthrough] http-fallback POST {url}", flush=True)
        log_upstream_request("chatgpt-passthrough-ws-http", url, forwarded)
        upstream = await http_session.post(url, json=forwarded, headers=headers)
        upstream_forward_headers = observe_upstream_response("chatgpt-passthrough-ws", upstream)
        if upstream.status >= 400:
            text = await upstream.text()
            code, message = parse_upstream_error(text, upstream.status)
            log_upstream_response(
                "chatgpt-passthrough-ws-http",
                url,
                upstream.status,
                text,
                request_body=forwarded,
                stream=True,
            )
            upstream.release()
            await _write_ws_error(ws, upstream.status, code, message)
            return
        collector = ChatgptPassthroughResponseCollector(forwarded)
        cache_store = (
            (lambda response_id, items, *, terminal=False: self._store_chatgpt_passthrough_conversation(
                session_key, response_id, items, terminal=terminal
            ))
            if _chatgpt_expand_continuations_enabled()
            else None
        )
        try:
            await _relay_sse_response_to_ws(
                upstream,
                request,
                ws,
                response_model_override=target.response_model_override,
                collector=collector,
                cache_collected=cache_store,
                upstream_forward_headers=upstream_forward_headers,
            )
        finally:
            if cache_store is not None:
                await self._store_chatgpt_passthrough_conversation_async(
                    session_key,
                    collector.response_id,
                    collector.conversation_items(),
                    terminal=True,
                )
            await _close_upstream(upstream)

    async def _handle_response_create_websocket(
        self,
        request: web.Request,
        ws: web.WebSocketResponse,
        payload: dict[str, Any],
    ) -> None:
        """Legacy entrypoint kept for tests; production uses responses_websocket loop."""
        target = await self._resolve_ws_passthrough_target(payload)
        if target is not None and ws_passthrough_enabled():
            async with ClientSession(timeout=self.timeout) as http_session:
                passthrough = WsPassthroughSession(client_session=http_session, client_ws=ws)
                if await self._handle_ws_passthrough_response_create(request, passthrough, payload, target):
                    await passthrough.close_upstream()
                    return
                await self._handle_local_response_create_websocket(
                    request,
                    ws,
                    payload,
                    http_session=http_session,
                )
            return
        if is_chatgpt_passthrough_slug(str(payload.get("model") or "")):
            auth_path = DEFAULT_CODEX_AUTH.expanduser()
            try:
                auth = json.loads(auth_path.read_text())
            except FileNotFoundError:
                await _write_ws_error(ws, 401, "unauthorized", "~/.codex/auth.json not found")
                return
            except json.JSONDecodeError:
                await _write_ws_error(ws, 401, "unauthorized", "auth.json is not valid JSON")
                return
            tokens = auth.get("tokens") or {}
            access_token = tokens.get("access_token")
            if not access_token:
                await _write_ws_error(ws, 401, "unauthorized", "auth.json has no access_token")
                return
            upstream_model = chatgpt_upstream_model(str(payload.get("model") or ""))
            requested = str(payload.get("model") or "")
            target = self._WsPassthroughTarget(
                kind="chatgpt",
                requested_slug=requested,
                upstream_model=upstream_model,
                response_model_override=requested if requested != upstream_model else None,
                access_token=access_token,
                account_id=tokens.get("account_id") or "",
            )
            async with ClientSession(timeout=self.timeout) as http_session:
                await self._handle_chatgpt_response_create_websocket_http(
                    request,
                    ws,
                    payload,
                    target,
                    http_session=http_session,
                )
            return
        async with ClientSession(timeout=self.timeout) as http_session:
            await self._handle_local_response_create_websocket(
                request,
                ws,
                payload,
                http_session=http_session,
            )

    async def _handle_local_response_create_websocket(
        self,
        request: web.Request,
        ws: web.WebSocketResponse,
        payload: dict[str, Any],
        *,
        http_session: ClientSession | None = None,
    ) -> None:
        forwarded = {k: v for k, v in payload.items() if k != "type"}
        forwarded["stream"] = True
        url = f"{request.scheme}://{request.host}/v1/responses"
        headers = openai_upstream_headers(
            request.headers,
            accept="text/event-stream",
        )
        session = http_session
        if session is None:
            async with ClientSession(timeout=self.timeout) as owned:
                await self._relay_local_response_create_over_http(owned, url, forwarded, headers, request, ws)
            return
        await self._relay_local_response_create_over_http(session, url, forwarded, headers, request, ws)

    async def _relay_local_response_create_over_http(
        self,
        session: ClientSession,
        url: str,
        forwarded: dict[str, Any],
        headers: dict[str, str],
        request: web.Request,
        ws: web.WebSocketResponse,
    ) -> None:
        upstream = await session.post(url, json=forwarded, headers=headers)
        if upstream.status >= 400:
            text = await upstream.text()
            code, message = parse_upstream_error(text, upstream.status)
            upstream.release()
            await _write_ws_error(ws, upstream.status, code, message)
            return
        try:
            await _relay_sse_response_to_ws(
                upstream,
                request,
                ws,
            )
        finally:
            await _close_upstream(upstream)

    async def responses(self, request: web.Request) -> web.StreamResponse:
        body = await request.json()
        _log_incoming_request("/v1/responses", body)
        body = await self._maybe_apply_auto_router(body)
        compaction_response = await self._maybe_handle_http_compaction_v2(request, body)
        if compaction_response is not None:
            return compaction_response
        model = str(body.get("model") or "")
        if is_chatgpt_passthrough_slug(model):
            upstream = chatgpt_upstream_model(model)
            override = model if model != upstream else None
            return await self._chatgpt_passthrough(
                request,
                body,
                response_model_override=override,
                upstream_model=upstream,
            )
        if is_cursor_passthrough_slug(model):
            return await self._cursor_passthrough(
                request,
                body,
                response_model_override=model,
                upstream_model=cursor_upstream_model(model),
            )
        if self._needs_image_gen(body) or self._needs_image_followup(body):
            return await self._chatgpt_passthrough(request, body, response_model_override=model)
        route = await self._route(body)
        tool_types = responses_tool_type_map(body.get("tools"))
        tool_resolve = responses_tool_resolve_map(body.get("tools"))
        body = prepare_codex_byok_responses_body(body, request.headers)
        body = self._apply_responses_input_pipeline(body, request, context="turn")
        if route.is_openai_responses:
            return await self._post_openai_responses(request, route, body)
        if route.is_openai_chat:
            forwarded = responses_to_chat(body, route.model)
            return await self._post_openai_chat(
                request, route, forwarded, as_responses=True, tool_types=tool_types, tool_resolve=tool_resolve
            )
        if route.is_anthropic:
            forwarded = responses_to_anthropic(body, route.model, route.max_output_tokens)
            return await self._post_anthropic(
                request, route, forwarded, as_responses=True, tool_types=tool_types, tool_resolve=tool_resolve
            )
        raise web.HTTPBadGateway(text=f"Unsupported model provider: {route.provider}")

    def _summary_from_compact_upstream_payload(self, payload: dict[str, Any]) -> str:
        item = compaction_item_from_response_payload(payload)
        if item is not None:
            return decode_shim_compaction_summary(item.get("encrypted_content")) or ""
        return compaction_summary_from_output(payload.get("output"))

    async def _maybe_handle_http_compaction_v2(
        self,
        request: web.Request,
        body: dict[str, Any],
    ) -> web.StreamResponse | web.Response | None:
        try:
            stripped_input = strip_terminal_compaction_trigger(body.get("input"))
        except CompactionTriggerError as exc:
            model = str(body.get("model") or "unknown")
            return web.json_response(
                _responses_error_payload(model, "invalid_request_error", str(exc)),
                status=400,
            )
        if stripped_input is None:
            return None
        _log_client_request("/v1/responses compaction-v2", body)
        return await self._responses_compaction_v2(request, body, stripped_input)

    async def _maybe_handle_ws_compaction_v2(
        self,
        request: web.Request,
        ws: web.WebSocketResponse,
        payload: dict[str, Any],
    ) -> bool:
        body = {k: v for k, v in payload.items() if k != "type"}
        body = await self._maybe_apply_auto_router(body)
        try:
            stripped_input = strip_terminal_compaction_trigger(body.get("input"))
        except CompactionTriggerError as exc:
            await _write_ws_error(ws, 400, "invalid_request_error", str(exc))
            return True
        if stripped_input is None:
            return False
        _log_client_request("ws compaction-v2", body, transport="ws")
        compaction_item, response_slug, usage, error_response = await self._resolve_compaction_v2_result(
            request,
            body,
            stripped_input,
        )
        if error_response is not None:
            text = error_response.text if isinstance(error_response, web.Response) else ""
            code, message = parse_upstream_error(text, error_response.status)
            _log_client_response(
                "ws compaction-v2",
                error_response.status,
                detail=message,
                code=code,
            )
            await _ws_error_from_http_response(ws, error_response)
            return True
        summary_chars = len(decode_shim_compaction_summary(compaction_item.get("encrypted_content")) or "")
        _log_client_response(
            "ws compaction-v2",
            200,
            detail=f"compaction item id={compaction_item.get('id')!r} summary_chars={summary_chars}",
        )
        await _write_compaction_v2_ws(ws, response_slug, compaction_item, usage)
        return True

    async def _resolve_compaction_v2_result(
        self,
        request: web.Request,
        body: dict[str, Any],
        stripped_input: list[Any],
        *,
        route: ShimModel | None = None,
        tool_types: dict[str, str] | None = None,
        tool_resolve: dict[str, tuple[str | None, str]] | None = None,
    ) -> tuple[dict[str, Any], str, dict[str, Any] | None, web.StreamResponse | web.Response | None]:
        model = str(body.get("model") or "")
        log_compaction_input_snapshot(
            "pre-expand",
            stripped_input,
            model=model,
            previous_response_id=body.get("previous_response_id"),
        )
        if is_chatgpt_passthrough_slug(model):
            stripped_input = self._expand_responses_stripped_input(
                request,
                body,
                stripped_input,
                context="compaction",
            )
        log_compaction_input_snapshot(
            "post-expand",
            stripped_input,
            model=model,
            previous_response_id=body.get("previous_response_id"),
        )
        print(
            f"[compaction-v2] resolve model={model!r} stripped_input_items={len(stripped_input)}",
            flush=True,
        )
        try:
            if is_chatgpt_passthrough_slug(model):
                result = await self._run_compaction_orchestrator(
                    request,
                    body,
                    stripped_input,
                    provider="chatgpt",
                    requested_slug=model,
                    upstream_model=chatgpt_upstream_model(model),
                )
                return result.item, model, result.usage, None
            if is_cursor_passthrough_slug(model):
                result = await self._run_compaction_orchestrator(
                    request,
                    body,
                    stripped_input,
                    provider="cursor",
                    requested_slug=model,
                )
                return result.item, model, result.usage, None
            if route is None:
                route = await self._route(body)
            if tool_types is None:
                tool_types = responses_tool_type_map(body.get("tools"))
            if tool_resolve is None:
                tool_resolve = responses_tool_resolve_map(body.get("tools"))
            result = await self._run_compaction_orchestrator(
                request,
                body,
                stripped_input,
                provider="byok",
                requested_slug=route.slug,
                route=route,
                tool_types=tool_types,
                tool_resolve=tool_resolve,
            )
            return result.item, route.slug, result.usage, None
        except CompactionOrchestratorError as exc:
            slug = model or (route.slug if route else "unknown")
            error_response = _compaction_orchestrator_error_response(slug, exc)
            return {}, slug, None, error_response

    async def _responses_compaction_v2(
        self,
        request: web.Request,
        body: dict[str, Any],
        stripped_input: list[Any],
        *,
        route: ShimModel | None = None,
        tool_types: dict[str, str] | None = None,
        tool_resolve: dict[str, tuple[str | None, str]] | None = None,
    ) -> web.StreamResponse:
        compaction_item, response_slug, usage, error_response = await self._resolve_compaction_v2_result(
            request,
            body,
            stripped_input,
            route=route,
            tool_types=tool_types,
            tool_resolve=tool_resolve,
        )
        if error_response is not None:
            response_slug = response_slug or str(body.get("model") or "unknown")
            text = error_response.text if isinstance(error_response, web.Response) else ""
            code, message = parse_upstream_error(text, error_response.status)
            _log_client_response(
                "http compaction-v2",
                error_response.status,
                detail=message,
                code=code,
            )
            if body.get("stream"):
                return await _stream_responses_error_from_http_response(
                    request,
                    response_slug,
                    error_response,
                    slug="compaction-v2",
                )
            text = error_response.text or ""
            code, message = parse_upstream_error(text, error_response.status)
            print(f"[err] compaction-v2 returned {error_response.status}: {message[:500]}", flush=True)
            return web.json_response(
                _responses_error_payload(response_slug, code, message),
                status=error_response.status,
            )
        summary_chars = len(decode_shim_compaction_summary(compaction_item.get("encrypted_content")) or "")
        _log_client_response(
            "http compaction-v2",
            200,
            detail=f"compaction item id={compaction_item.get('id')!r} summary_chars={summary_chars}",
        )
        return await _stream_compaction_v2_sse(request, response_slug, compaction_item, usage)

    async def _fetch_byok_compact_summary(
        self,
        request: web.Request,
        route: ShimModel,
        compact_body: dict[str, Any],
        *,
        tool_types: dict[str, str] | None,
        tool_resolve: dict[str, tuple[str | None, str]] | None,
    ) -> tuple[str, dict[str, Any] | None, web.StreamResponse | web.Response | None]:
        compact_body = prepare_codex_byok_responses_body(compact_body, request.headers)
        compact_body = self._apply_responses_input_pipeline(
            compact_body,
            request,
            context="compaction",
        )
        compact_body["stream"] = False
        _log_upstream_io_detail(
            surface="byok-compact",
            phase="pre-request",
            url=f"{route.slug}:{route.provider}",
            forwarded=compact_body,
        )
        if route.is_openai_responses:
            upstream_response = await self._post_openai_responses(request, route, compact_body)
            if not isinstance(upstream_response, web.Response) or upstream_response.status >= 400:
                _log_upstream_response_from_http(
                    "byok-compact-openai-responses",
                    route.slug,
                    upstream_response,
                )
                return "", None, upstream_response
            try:
                payload = json.loads(upstream_response.text or "{}")
            except json.JSONDecodeError:
                return "", None, upstream_response
            if not isinstance(payload, dict):
                return "", None, upstream_response
            summary = self._summary_from_compact_upstream_payload(payload)
            usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else None
            _log_upstream_status("byok-compact-openai-responses", route.slug, 200, message=f"summary_chars={len(summary)}")
            return summary, usage, None
        if route.is_openai_chat:
            forwarded = responses_to_chat(compact_body, route.model)
            forwarded["stream"] = False
            upstream_response = await self._post_openai_chat(
                request,
                route,
                forwarded,
                as_responses=True,
                tool_types=tool_types,
                tool_resolve=tool_resolve,
            )
            if not isinstance(upstream_response, web.Response) or upstream_response.status >= 400:
                _log_upstream_response_from_http("byok-compact-openai-chat", route.slug, upstream_response)
                return "", None, upstream_response
            try:
                payload = json.loads(upstream_response.text or "{}")
            except json.JSONDecodeError:
                return "", None, upstream_response
            if not isinstance(payload, dict):
                return "", None, upstream_response
            summary = self._summary_from_compact_upstream_payload(payload)
            usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else None
            _log_upstream_status("byok-compact-openai-chat", route.slug, 200, message=f"summary_chars={len(summary)}")
            return summary, usage, None
        if route.is_anthropic:
            forwarded = responses_to_anthropic(compact_body, route.model, route.max_output_tokens)
            forwarded["stream"] = False
            upstream_response = await self._post_anthropic(
                request,
                route,
                forwarded,
                as_responses=True,
                tool_types=tool_types,
                tool_resolve=tool_resolve,
            )
            if not isinstance(upstream_response, web.Response) or upstream_response.status >= 400:
                _log_upstream_response_from_http("byok-compact-anthropic", route.slug, upstream_response)
                return "", None, upstream_response
            try:
                payload = json.loads(upstream_response.text or "{}")
            except json.JSONDecodeError:
                return "", None, upstream_response
            if not isinstance(payload, dict):
                return "", None, upstream_response
            summary = self._summary_from_compact_upstream_payload(payload)
            usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else None
            _log_upstream_status("byok-compact-anthropic", route.slug, 200, message=f"summary_chars={len(summary)}")
            return summary, usage, None
        error = web.HTTPBadGateway(text=f"Unsupported model provider: {route.provider}")
        return "", None, web.json_response(
            _responses_error_payload(route.slug, "upstream_error", error.text),
            status=502,
        )

    async def responses_compact(self, request: web.Request) -> web.StreamResponse:
        body = await request.json()
        _log_incoming_request("/v1/responses/compact", body)
        body = await self._maybe_apply_auto_router(body)
        model = str(body.get("model") or "")
        input_items = body.get("input") or []
        if not isinstance(input_items, list):
            input_items = [input_items] if input_items is not None else []

        async def _compact_json_response(
            *,
            provider: Literal["chatgpt", "cursor", "byok"],
            requested_slug: str,
            upstream_model: str | None = None,
            route: ShimModel | None = None,
        ) -> web.StreamResponse | web.Response:
            try:
                result = await self._run_compaction_orchestrator(
                    request,
                    body,
                    input_items,
                    provider=provider,
                    requested_slug=requested_slug,
                    upstream_model=upstream_model,
                    route=route,
                    tool_types=responses_tool_type_map(body.get("tools")),
                    tool_resolve=responses_tool_resolve_map(body.get("tools")),
                    transport="legacy_compact",
                )
            except CompactionOrchestratorError as exc:
                return _compaction_orchestrator_error_response(requested_slug, exc)
            if result.legacy_payload is not None:
                payload = copy.deepcopy(result.legacy_payload)
                _rewrite_response_model(payload, requested_slug)
                for output_item in payload.get("output") or []:
                    if isinstance(output_item, dict) and "model" in output_item:
                        output_item["model"] = requested_slug
                return web.json_response(payload)
            summary = decode_shim_compaction_summary(result.item.get("encrypted_content")) or ""
            return web.json_response(compact_response_payload(requested_slug, summary, result.usage))

        if is_chatgpt_passthrough_slug(model):
            return await _compact_json_response(
                provider="chatgpt",
                requested_slug=model,
                upstream_model=chatgpt_upstream_model(model),
            )
        if is_cursor_passthrough_slug(model):
            return await _compact_json_response(provider="cursor", requested_slug=model)
        route = await self._route(body)
        return await _compact_json_response(provider="byok", requested_slug=route.slug, route=route)

    def _needs_image_gen(self, body: dict[str, Any]) -> bool:
        tools = body.get("tools") or []
        image_tool_names: set[str] = set()
        non_image_tool_count = 0
        for tool in tools:
            if not isinstance(tool, dict):
                non_image_tool_count += 1
                continue
            tool_type = str(tool.get("type") or "")
            fn = tool.get("function") or tool.get("name") or {}
            name = fn.get("name") if isinstance(fn, dict) else fn
            normalized = f"{tool_type} {name or ''}".lower()
            is_image_tool = tool_type in {"image_generation", "image_gen"} or ("image" in normalized and "gen" in normalized)
            if is_image_tool:
                image_tool_names.add(str(name or tool_type))
            else:
                non_image_tool_count += 1
        if not image_tool_names:
            return False

        tool_choice = body.get("tool_choice")
        if isinstance(tool_choice, str):
            if any(name.lower() in tool_choice.lower() for name in image_tool_names):
                return True
        elif isinstance(tool_choice, dict):
            fn = tool_choice.get("function") or {}
            choice_name = str(tool_choice.get("name") or (fn.get("name") if isinstance(fn, dict) else "") or tool_choice.get("type") or "").lower()
            if any(name.lower() in choice_name for name in image_tool_names):
                return True

        if non_image_tool_count == 0:
            return True

        latest = self._latest_user_text(body).lower()
        if not latest:
            return False
        image_intent_markers = (
            "@image",
            "imagegen",
            "image gen",
            "image_gen",
            "generate image",
            "generate an image",
            "generate a picture",
            "generate a photo",
            "generate an illustration",
            "create image",
            "create an image",
            "create a picture",
            "create a photo",
            "draw image",
            "draw an image",
            "make image",
            "make an image",
            "render image",
        )
        if any(marker in latest for marker in image_intent_markers):
            return True
        code_words = {"code", "component", "react", "tsx", "jsx", "html", "css", "svg", "file"}
        latest_words = {"".join(ch for ch in word if ch.isalnum()) for word in latest.split()}
        if latest_words & code_words:
            return False
        creative_objects = ("icon", "logo", "wallpaper", "poster", "banner", "avatar")
        creative_verbs = ("generate", "create", "draw", "design", "make", "render")
        return any(verb in latest for verb in creative_verbs) and any(obj in latest for obj in creative_objects)

    def _needs_image_followup(self, body: dict[str, Any]) -> bool:
        if not self._has_image_generation_history(body):
            return False
        latest = self._latest_user_text(body).lower()
        if not latest:
            return False
        direct_image_refs = ("image", "picture", "photo", "icon", "logo", "illustration")
        followup_actions = (
            "inspect",
            "look at",
            "view",
            "describe",
            "what do you see",
            "analyze",
            "modify",
            "edit",
            "change",
            "improve",
            "enhance",
            "upscale",
            "variation",
            "use",
            "based on",
            "same",
        )
        if any(ref in latest for ref in direct_image_refs) and any(action in latest for action in followup_actions):
            return True
        pronoun_followups = (
            "inspect it",
            "look at it",
            "view it",
            "describe it",
            "analyze it",
            "modify it",
            "edit it",
            "change it",
            "improve it",
            "enhance it",
            "upscale it",
            "make it brighter",
            "make it darker",
            "make it more",
            "use it",
            "based on it",
        )
        return any(marker in latest for marker in pronoun_followups)

    def _has_image_generation_history(self, body: dict[str, Any]) -> bool:
        inputs = body.get("input") or []
        if not isinstance(inputs, list):
            return False
        return any(isinstance(item, dict) and item.get("type") == "image_generation_call" for item in inputs)

    def _latest_user_text(self, body: dict[str, Any]) -> str:
        inputs = body.get("input") or []
        if isinstance(inputs, str):
            return inputs
        if not isinstance(inputs, list):
            return ""
        for item in reversed(inputs):
            if isinstance(item, str):
                return item
            if not isinstance(item, dict):
                continue
            if item.get("role") == "user":
                text = self._content_to_debug_text(item.get("content"))
                if text:
                    return text
            elif item.get("type") in {"input_text", "text"}:
                text = self._content_to_debug_text(item)
                if text:
                    return text
        return ""

    def _content_to_debug_text(self, content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict):
                    parts.append(str(part.get("text") or part.get("content") or ""))
                else:
                    parts.append(str(part))
            return "\n".join(part for part in parts if part)
        if isinstance(content, dict):
            return str(content.get("text") or content.get("content") or "")
        return str(content)

    def _apply_responses_input_pipeline(
        self,
        body: dict[str, Any],
        request: web.Request,
        *,
        context: str = "turn",
        strip_previous_response_id: bool = False,
    ) -> dict[str, Any]:
        return apply_responses_input_pipeline_to_body(
            body,
            cache=self._chatgpt_conversation_cache,
            session_key=self._session_key(request),
            expand_enabled=_chatgpt_expand_continuations_enabled(),
            context=context,
            strip_previous_response_id=strip_previous_response_id,
        )

    def _expand_responses_stripped_input(
        self,
        request: web.Request,
        body: dict[str, Any],
        stripped_input: list[Any],
        *,
        context: str,
    ) -> list[Any]:
        from .responses_input_pipeline import prepare_responses_input_items

        repaired, _warnings = prepare_responses_input_items(
            cache=self._chatgpt_conversation_cache,
            session_key=self._session_key(request),
            previous_response_id=body.get("previous_response_id"),
            input_items=stripped_input,
            expand_enabled=_chatgpt_expand_continuations_enabled(),
            context=context,
        )
        return repaired

    def _prepare_chatgpt_passthrough_body(self, body: dict[str, Any], *, session_key: str) -> dict[str, Any]:
        expand = _chatgpt_expand_continuations_enabled()
        sanitized = _sanitize_chatgpt_passthrough_body(body, strip_previous_response_id=False)
        return apply_responses_input_pipeline_to_body(
            sanitized,
            cache=self._chatgpt_conversation_cache,
            session_key=session_key,
            expand_enabled=expand,
            context="turn",
            strip_previous_response_id=expand,
        )

    def _store_chatgpt_passthrough_conversation(
        self,
        session_key: str,
        response_id: str | None,
        items: list[Any],
        *,
        terminal: bool = True,
    ) -> None:
        if not response_id or not items:
            return
        self._chatgpt_conversation_cache.put(session_key, response_id, items, terminal=terminal)

    async def _store_chatgpt_passthrough_conversation_async(
        self,
        session_key: str,
        response_id: str | None,
        items: list[Any],
        *,
        terminal: bool = True,
    ) -> None:
        if not response_id or not items:
            return
        if terminal:
            await asyncio.to_thread(
                self._chatgpt_conversation_cache.put,
                session_key,
                response_id,
                items,
                terminal=True,
            )
        else:
            self._chatgpt_conversation_cache.put(session_key, response_id, items, terminal=False)

    async def _chatgpt_passthrough(
        self,
        request: web.Request,
        body: dict[str, Any],
        response_model_override: str | None = None,
        upstream_model: str | None = None,
        *,
        allow_byok_fallback: bool = True,
        collect_stream: bool = False,
    ) -> web.StreamResponse:
        """Forward a Responses request to chatgpt.com using the user's Codex auth.

        Lets the picker expose OpenAI GPT models (ChatGPT subscription) as
        first-class models alongside configured BYOK entries.
        """
        requested = response_model_override or str(body.get("model") or "")
        auth_path = DEFAULT_CODEX_AUTH.expanduser()
        try:
            auth = json.loads(auth_path.read_text())
        except FileNotFoundError:
            fallback = await self._maybe_passthrough_byok_fallback(
                request,
                body,
                requested=requested,
                response_slug=requested,
                status=401,
                detail="~/.codex/auth.json not found",
            )
            if fallback is not None:
                return fallback
            raise web.HTTPUnauthorized(text="~/.codex/auth.json not found")
        tokens = auth.get("tokens") or {}
        access_token = tokens.get("access_token")
        account_id = tokens.get("account_id") or ""
        if not access_token:
            fallback = await self._maybe_passthrough_byok_fallback(
                request,
                body,
                requested=requested,
                response_slug=requested,
                status=401,
                detail="auth.json has no access_token",
            )
            if fallback is not None:
                return fallback
            raise web.HTTPUnauthorized(text="auth.json has no access_token")
        session_key = self._session_key(request)
        forwarded = self._prepare_chatgpt_passthrough_body(body, session_key=session_key)
        if collect_stream:
            forwarded["stream"] = True
        forwarded["model"] = upstream_model or CHATGPT_MODEL_SLUG
        headers = _chatgpt_passthrough_upstream_headers(
            request,
            access_token=access_token,
            account_id=account_id,
            accept="text/event-stream" if forwarded.get("stream") else "application/json",
        )
        _log_chatgpt_passthrough_trace(request, forwarded, headers, phase="pre-upstream")
        url = "https://chatgpt.com/backend-api/codex/responses"
        log_upstream_request("chatgpt-passthrough", url, forwarded)
        async with ClientSession(timeout=self.timeout) as session:
            upstream = await session.post(url, json=forwarded, headers=headers)
            if upstream.status >= 400:
                upstream_forward_headers = observe_upstream_response("chatgpt-passthrough", upstream)
                text = await upstream.text()
                status = upstream.status
                content_type = upstream.content_type or "text/plain"
                upstream.release()
                log_upstream_response(
                    "chatgpt-passthrough",
                    url,
                    status,
                    text,
                    request_body=forwarded,
                    stream=bool(forwarded.get("stream")),
                )
                if allow_byok_fallback:
                    fallback = await self._maybe_passthrough_byok_fallback(
                        request,
                        body,
                        requested=requested,
                        response_slug=requested,
                        status=status,
                        detail=text,
                    )
                    if fallback is not None:
                        return fallback
                return _upstream_text_response(
                    status,
                    text,
                    content_type=content_type,
                    upstream_headers=upstream_forward_headers,
                )
            if not forwarded.get("stream"):
                payload = await upstream.json(content_type=None)
                if _chatgpt_expand_continuations_enabled() and isinstance(payload, dict):
                    response_id = payload.get("id")
                    output = payload.get("output")
                    if isinstance(response_id, str) and isinstance(output, list):
                        await self._store_chatgpt_passthrough_conversation_async(
                            session_key,
                            response_id,
                            [*_chatgpt_input_items(forwarded.get("input")), *copy.deepcopy(output)],
                            terminal=True,
                        )
                usage = payload.get("usage") if isinstance(payload, dict) else None
                upstream_forward_headers = observe_upstream_response(
                    "chatgpt-passthrough",
                    upstream,
                    usage=usage if isinstance(usage, dict) else None,
                )
                _log_upstream_status(
                    "chatgpt-passthrough",
                    url,
                    200,
                    message=f"response_id={payload.get('id') if isinstance(payload, dict) else None!r}",
                )
                log_upstream_response(
                    "chatgpt-passthrough",
                    url,
                    200,
                    json.dumps(payload, default=str)[:12_000] if isinstance(payload, dict) else "",
                    request_body=forwarded,
                    stream=False,
                )
                _rewrite_response_model(payload, response_model_override)
                response = web.json_response(payload)
                apply_upstream_headers_to_response(response, upstream_headers_from_response(upstream))
                upstream.release()
                return response
            observe_upstream_response("chatgpt-passthrough", upstream)
            log_upstream_response(
                "chatgpt-passthrough",
                url,
                upstream.status,
                request_body=forwarded,
                stream=True,
            )
            if collect_stream:
                completed: dict[str, Any] | None = None
                failed_message = ""
                try:
                    async for line in _sse_lines(upstream, request):
                        if line == "[DONE]":
                            break
                        try:
                            payload = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if payload.get("type") == "response.failed":
                            response_obj = payload.get("response")
                            if isinstance(response_obj, dict):
                                err = response_obj.get("error")
                                if isinstance(err, dict):
                                    failed_message = str(err.get("message") or failed_message)
                            break
                        if payload.get("type") == "response.completed":
                            response_obj = payload.get("response")
                            if isinstance(response_obj, dict):
                                completed = response_obj
                            break
                finally:
                    await _close_upstream(upstream)
                if completed is None:
                    detail = failed_message or "ChatGPT passthrough stream ended without response.completed"
                    return _upstream_text_response(
                        502,
                        detail,
                        content_type="text/plain",
                    )
                usage = completed.get("usage") if isinstance(completed.get("usage"), dict) else None
                upstream_forward_headers = upstream_headers_from_response(upstream)
                log_upstream_response(
                    "chatgpt-passthrough",
                    url,
                    200,
                    json.dumps(completed, default=str)[:12_000],
                    request_body=forwarded,
                    stream=True,
                )
                _rewrite_response_model(completed, response_model_override)
                response = web.json_response(completed)
                apply_upstream_headers_to_response(response, upstream_forward_headers)
                return response
            response = prepare_downstream_sse_response(upstream)
            await response.prepare(request)
            collector = ChatgptPassthroughResponseCollector(forwarded)
            try:
                async for line in _sse_lines(upstream, request):
                    if line == "[DONE]":
                        await _safe_write(response, b"data: [DONE]\n\n")
                        break
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        await _safe_write(response, f"data: {line}\n\n".encode())
                        continue
                    if _chatgpt_expand_continuations_enabled():
                        collector.record(payload)
                        if _should_cache_chatgpt_passthrough_event(payload):
                            self._store_chatgpt_passthrough_conversation(
                                session_key,
                                collector.response_id,
                                collector.conversation_items(),
                                terminal=False,
                            )
                    if response_model_override:
                        _rewrite_response_model(payload, response_model_override)
                    if payload.get("type") == "response.completed":
                        response_obj = payload.get("response")
                        usage = response_obj.get("usage") if isinstance(response_obj, dict) else None
                        observe_upstream_response(
                            "chatgpt-passthrough",
                            upstream,
                            usage=usage if isinstance(usage, dict) else None,
                        )
                    await _write_sse(response, payload)
            except asyncio.CancelledError:
                print("[cancel] client disconnected during ChatGPT passthrough stream", flush=True)
                raise
            except ClientDisconnected:
                print("[cancel] client disconnected during ChatGPT passthrough stream", flush=True)
            finally:
                if _chatgpt_expand_continuations_enabled():
                    await self._store_chatgpt_passthrough_conversation_async(
                        session_key,
                        collector.response_id,
                        collector.conversation_items(),
                        terminal=True,
                    )
                await _close_upstream(upstream)
            try:
                await response.write_eof()
            except Exception:
                pass
            return response

    async def _chatgpt_compact_passthrough(
        self,
        request: web.Request,
        body: dict[str, Any],
        upstream_model: str | None = None,
    ) -> web.StreamResponse:
        requested = str(body.get("model") or "")
        auth_path = DEFAULT_CODEX_AUTH.expanduser()
        try:
            auth = json.loads(auth_path.read_text())
        except FileNotFoundError:
            fallback = await self._maybe_passthrough_byok_fallback(
                request,
                body,
                requested=requested,
                response_slug=requested,
                status=401,
                detail="~/.codex/auth.json not found",
                compact=True,
            )
            if fallback is not None:
                return fallback
            raise web.HTTPUnauthorized(text="~/.codex/auth.json not found")
        tokens = auth.get("tokens") or {}
        access_token = tokens.get("access_token")
        account_id = tokens.get("account_id") or ""
        if not access_token:
            fallback = await self._maybe_passthrough_byok_fallback(
                request,
                body,
                requested=requested,
                response_slug=requested,
                status=401,
                detail="auth.json has no access_token",
                compact=True,
            )
            if fallback is not None:
                return fallback
            raise web.HTTPUnauthorized(text="auth.json has no access_token")
        forwarded = _sanitize_chatgpt_compact_passthrough_body(body)
        original_model = str(forwarded.get("model") or "")
        response = await self._post_chatgpt_native_compact(
            request,
            forwarded,
            upstream_model=upstream_model or CHATGPT_MODEL_SLUG,
            requested_slug=original_model or requested,
        )
        if response.status >= 400:
            text = response.text or ""
            _, message = parse_upstream_error(text, response.status)
            input_items = body.get("input") or []
            if not isinstance(input_items, list):
                input_items = [input_items] if input_items is not None else []
            try:
                result = await self._run_compaction_orchestrator(
                    request,
                    body,
                    input_items,
                    provider="chatgpt",
                    requested_slug=original_model or requested,
                    upstream_model=upstream_model,
                    transport="legacy_compact",
                    skip_native=True,
                    preset_native_message=message,
                )
            except CompactionOrchestratorError as exc:
                enriched = _compaction_orchestrator_error_response(original_model or requested, exc)
                fallback = await self._maybe_passthrough_byok_fallback(
                    request,
                    body,
                    requested=requested,
                    response_slug=original_model or requested,
                    status=response.status,
                    detail=text,
                    compact=True,
                )
                if fallback is not None:
                    return self._apply_native_compaction_notice_to_compact_response(
                        fallback,
                        response_slug=original_model or requested,
                        native_message=message,
                    )
                return enriched
            summary = decode_shim_compaction_summary(result.item.get("encrypted_content")) or ""
            return web.json_response(
                compact_response_payload(original_model or requested, summary, result.usage)
            )
        return response

    async def _chatgpt_summarization_compact_response(
        self,
        request: web.Request,
        body: dict[str, Any],
        *,
        upstream_model: str | None,
        response_slug: str,
        native_message: str,
    ) -> web.Response | None:
        input_items = body.get("input")
        if not isinstance(input_items, list):
            input_items = [input_items] if input_items is not None else []
        try:
            result = await self._run_compaction_orchestrator(
                request,
                body,
                input_items,
                provider="chatgpt",
                requested_slug=response_slug,
                upstream_model=upstream_model or CHATGPT_MODEL_SLUG,
                transport="legacy_compact",
                skip_native=True,
                preset_native_message=native_message,
            )
        except CompactionOrchestratorError:
            return None
        summary = decode_shim_compaction_summary(result.item.get("encrypted_content")) or ""
        if not summary.strip():
            return None
        print(
            f"[fallback] {response_slug} native compaction -> ChatGPT summarization",
            flush=True,
        )
        return web.json_response(
            compact_response_payload(response_slug, summary, result.usage)
        )

    def _apply_native_compaction_notice_to_compact_response(
        self,
        response: web.StreamResponse | web.Response,
        *,
        response_slug: str,
        native_message: str,
    ) -> web.StreamResponse | web.Response:
        if not isinstance(response, web.Response) or response.status >= 400:
            return response
        try:
            payload = json.loads(response.text or "{}")
        except json.JSONDecodeError:
            return response
        if not isinstance(payload, dict):
            return response
        summary = self._summary_from_compact_upstream_payload(payload)
        if not summary.strip():
            return response
        return web.json_response(
            compact_response_payload(
                response_slug,
                apply_compaction_fallback_notice(summary, native_message),
                payload.get("usage"),
            )
        )

    async def _cursor_passthrough(
        self,
        request: web.Request,
        body: dict[str, Any],
        response_model_override: str | None = None,
        upstream_model: str | None = None,
        force_non_stream: bool = False,
    ) -> web.StreamResponse:
        """Route Composer through cursor-agent using Cursor subscription login."""
        if not cursor_passthrough_available():
            raise web.HTTPUnauthorized(
                text="Cursor subscription auth unavailable. Run `cursor-agent login`, then retry."
            )
        slug = response_model_override or CURSOR_MODEL_SLUG
        upstream = upstream_model or cursor_upstream_model(slug)
        prompt = build_cursor_prompt(body)
        cursor_workspace_path = resolve_cursor_workspace(
            body,
            request_headers=dict(request.headers),
            prompt=prompt,
        )
        stream = bool(body.get("stream")) and not force_non_stream
        tool_types = responses_tool_type_map(body.get("tools"))
        tool_resolve = responses_tool_resolve_map(body.get("tools"))
        bridge_session: CursorBridgeSession | None = None

        if cursor_bridge_enabled() and body.get("tools"):
            allowed = bridge_allowed_tools(body)
            if allowed:
                port = shim_port_from_request_host(request.headers.get("Host", ""))
                bridge_session = CursorBridgeSession.create(
                    allowed_tools=allowed,
                    tool_types=tool_types,
                    tool_resolve=tool_resolve,
                )
                await cursor_bridge_registry.register(bridge_session)
                prompt += "\n\n" + build_bridge_suffix(
                    bridge_session,
                    port,
                    workspace=cursor_workspace_path,
                )

        try:
            if not stream:
                collector = CursorResponseCollector(
                    tool_types=tool_types,
                    tool_resolve=tool_resolve,
                )
                if bridge_session is not None:
                    bridge_session.attach_collector(collector)
                usage: dict[str, Any] | None = None
                fallback_text = ""
                async for event in iter_cursor_agent_events(
                    prompt,
                    upstream,
                    workspace=cursor_workspace_path,
                ):
                    if event["type"] == "completed":
                        fallback_text = str(event.get("text") or fallback_text)
                    elif event["type"] == "usage":
                        usage = event.get("usage") if isinstance(event.get("usage"), dict) else None
                    elif event["type"] == "error":
                        raise web.HTTPBadGateway(text=str(event.get("message") or "cursor-agent failed"))
                    else:
                        collector.consume(event)
                output = collector.build_output(fallback_text=fallback_text)
                payload: dict[str, Any] = {
                    "id": f"resp_{int(time.time() * 1000)}",
                    "object": "response",
                    "model": slug,
                    "status": "completed",
                    "output": output,
                }
                normalized_usage = normalize_responses_usage(usage)
                if normalized_usage:
                    payload["usage"] = normalized_usage
                return web.json_response(payload)

            response = _sse_response()
            await response.prepare(request)
            state = ResponsesStreamState(slug, tool_types=tool_types, tool_resolve=tool_resolve)
            if bridge_session is not None:

                async def _stream_emit(
                    *,
                    name: str,
                    arguments: dict[str, Any],
                    call_id: str,
                    namespace: str | None = None,
                    chat_name: str | None = None,
                ) -> None:
                    await state.emit_synthetic_function_call(
                        response,
                        name=name,
                        arguments=arguments,
                        call_id=call_id,
                        namespace=namespace,
                        chat_name=chat_name,
                    )

                bridge_session.attach_stream(_stream_emit)
            try:
                await state.start(response)
                async for event in iter_cursor_agent_events(
                    prompt,
                    upstream,
                    workspace=cursor_workspace_path,
                ):
                    if event["type"] == "usage":
                        normalized_usage = normalize_responses_usage(event.get("usage"))
                        if normalized_usage:
                            state.usage = normalized_usage
                    elif event["type"] == "error":
                        message = str(event.get("message") or "cursor-agent failed")
                        await state.fail(response, message, code="upstream_error")
                        break
                    else:
                        await _apply_cursor_stream_event(state, response, event)
                await state.finish(response, upstream_saw_done=True)
            except ClientDisconnected:
                pass
            except Exception as exc:
                print(f"[err] cursor passthrough {slug}: {exc}", flush=True)
                raise web.HTTPBadGateway(text=str(exc)) from exc
            try:
                await response.write_eof()
            except Exception:
                pass
            return response
        finally:
            if bridge_session is not None:
                cursor_bridge_registry.close(bridge_session.bridge_id)

    async def cursor_bridge_invoke(self, request: web.Request) -> web.Response:
        if not is_loopback_peer(request.remote):
            raise web.HTTPForbidden(text="Forbidden: bridge invoke requires loopback peer")
        try:
            payload = await request.json()
        except json.JSONDecodeError as exc:
            raise web.HTTPBadRequest(text="Invalid JSON body") from exc
        if not isinstance(payload, dict):
            raise web.HTTPBadRequest(text="JSON body must be an object")

        bridge_id = str(payload.get("bridge") or "").strip()
        tool = str(payload.get("tool") or "").strip()
        arguments = payload.get("arguments")
        namespace_raw = payload.get("namespace")
        namespace = str(namespace_raw).strip() if namespace_raw is not None else None
        if namespace == "":
            namespace = None

        if not bridge_id:
            raise web.HTTPBadRequest(text="bridge is required")
        if not tool:
            raise web.HTTPBadRequest(text="tool is required")
        if not isinstance(arguments, dict):
            raise web.HTTPBadRequest(text="arguments must be a JSON object")

        session = cursor_bridge_registry.get(bridge_id)
        if session is None:
            raise web.HTTPNotFound(text="Unknown or expired bridge session")

        try:
            result = await session.invoke(
                tool=tool,
                arguments=arguments,
                namespace=namespace,
            )
        except BridgeToolNotAllowedError as exc:
            raise web.HTTPBadRequest(text=str(exc)) from exc
        except BridgeError as exc:
            raise web.HTTPBadRequest(text=str(exc)) from exc

        print(
            f"[cursor-bridge] invoke bridge={bridge_id} tool={tool} namespace={namespace or '-'} "
            f"call_id={result.get('codex_call_id')}",
            flush=True,
        )
        return web.json_response(result)

    # ------------------------------------------------------------------
    # Auto Router
    # ------------------------------------------------------------------
    async def _active_router(self):
        """Return the RouterConfig only when enabled and at least one candidate
        backend is usable, so discovery never advertises a dead Auto entry."""
        config = self.settings.load_router()
        if config is None:
            return None
        models = await self._load_models()
        if router_module.router_is_active(config, available_model_slugs(models)):
            return config
        return None

    async def _maybe_apply_auto_router(self, body: dict[str, Any]) -> dict[str, Any]:
        """If the request targets the Auto Router slug, classify the task and
        rewrite ``model`` to the concrete backend that should handle it. Any
        failure leaves the body untouched so the request still routes normally."""
        config = self.settings.load_router()
        if not config or not config.effective_enabled:
            return body
        if str(body.get("model") or "") != config.slug:
            return body
        resolved = await self._resolve_auto_model(config, body)
        if resolved and resolved != config.slug:
            if router_module.router_log_enabled():
                print(f"[router] {config.slug} -> {resolved}", flush=True)
            new_body = dict(body)
            new_body["model"] = resolved
            return new_body
        return body

    async def _resolve_auto_model(self, config, body: dict[str, Any]) -> str | None:
        models = await self._load_models()
        candidates = router_module.filter_available(config, available_model_slugs(models))
        if not candidates:
            return None
        classify = None
        if config.classifier:
            classifier_model = self.settings.by_slug_or_model(config.classifier)
            if (
                classifier_model is not None
                and byok_model_has_credentials(classifier_model)
                and (classifier_model.is_openai_chat or classifier_model.is_anthropic)
            ):
                classify = self._make_classifier(classifier_model, config)
        log = (lambda message: print(message, flush=True)) if router_module.router_log_enabled() else None
        resolved, _info = await router_module.resolve_auto(config, candidates, body, classify, log=log)
        return resolved or router_module.fallback_slug(
            config, candidates, has_image_task=router_module.has_images(body)
        )

    def _make_classifier(self, model: ShimModel, config):
        timeout = ClientTimeout(total=config.timeout + 5, sock_connect=config.timeout, sock_read=config.timeout)

        async def classify(system_prompt: str, user_content: str) -> str:
            async with ClientSession(timeout=timeout) as session:
                if model.is_anthropic:
                    url = _join_url(model.base_url, "/messages")
                    payload = {
                        "model": model.model,
                        "max_tokens": config.max_tokens,
                        "system": system_prompt,
                        "messages": [{"role": "user", "content": user_content}],
                    }
                    upstream = await session.post(url, json=payload, headers=_anthropic_headers({}, model))
                    upstream.raise_for_status()
                    data = await upstream.json(content_type=None)
                    return _anthropic_text(data)
                url = _join_url(model.base_url, "/chat/completions")
                payload = {
                    "model": model.model,
                    "stream": False,
                    "temperature": 0,
                    "max_tokens": config.max_tokens,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                }
                upstream = await session.post(url, json=payload, headers=_openai_headers({}, model))
                upstream.raise_for_status()
                data = await upstream.json(content_type=None)
                message = (data.get("choices") or [{}])[0].get("message") or {}
                return str(message.get("content") or "")

        return classify

    async def _route(self, body: dict[str, Any]) -> ShimModel:
        requested = str(body.get("model") or "")
        models = await self._load_models()
        by_slug = {model.slug: model for model in models}
        route = by_slug.get(requested)
        if route is None:
            matches = [model for model in models if model.model == requested]
            if len(matches) == 1:
                route = matches[0]
        if route is None:
            raise web.HTTPNotFound(text=f"Unknown model slug/model: {requested}")
        if not byok_model_has_credentials(route):
            raise web.HTTPUnauthorized(text=_missing_api_key_message(route))
        return route

    def _passthrough_fallback_slug(self, requested: str) -> str | None:
        mapping = self.settings.passthrough_error_fallback()
        if requested in mapping:
            return mapping[requested]
        if is_chatgpt_passthrough_slug(requested):
            upstream = chatgpt_upstream_model(requested)
            if upstream in mapping:
                return mapping[upstream]
        return None

    async def _dispatch_byok_responses(
        self,
        request: web.Request,
        body: dict[str, Any],
        *,
        response_slug: str | None = None,
    ) -> web.StreamResponse:
        route = await self._route(body)
        client_slug = response_slug or route.slug
        tool_types = responses_tool_type_map(body.get("tools"))
        tool_resolve = responses_tool_resolve_map(body.get("tools"))
        body = prepare_codex_byok_responses_body(body, request.headers)
        body = self._apply_responses_input_pipeline(body, request, context="turn")
        if route.is_openai_responses:
            return await self._post_openai_responses(request, route, body)
        if route.is_openai_chat:
            forwarded = responses_to_chat(body, route.model)
            return await self._post_openai_chat(
                request,
                route,
                forwarded,
                as_responses=True,
                response_slug=client_slug,
                tool_types=tool_types,
                tool_resolve=tool_resolve,
            )
        if route.is_anthropic:
            forwarded = responses_to_anthropic(body, route.model, route.max_output_tokens)
            return await self._post_anthropic(
                request,
                route,
                forwarded,
                as_responses=True,
                response_slug=client_slug,
                tool_types=tool_types,
                tool_resolve=tool_resolve,
            )
        raise web.HTTPBadGateway(text=f"Unsupported model provider: {route.provider}")

    async def _dispatch_byok_compact_responses(
        self,
        request: web.Request,
        body: dict[str, Any],
        *,
        response_slug: str | None = None,
    ) -> web.StreamResponse:
        route = await self._route(body)
        client_slug = response_slug or route.slug
        tool_types = responses_tool_type_map(body.get("tools"))
        tool_resolve = responses_tool_resolve_map(body.get("tools"))
        compact_body = _compact_request_body(body, route.model)
        compact_body = prepare_codex_byok_responses_body(compact_body, request.headers)
        compact_body = self._apply_responses_input_pipeline(
            compact_body,
            request,
            context="compaction",
        )
        if route.is_openai_responses:
            compact_body["stream"] = False
            return await self._post_openai_responses(request, route, compact_body)
        if route.is_openai_chat:
            forwarded = responses_to_chat(compact_body, route.model)
            forwarded["stream"] = False
            response = await self._post_openai_chat(
                request,
                route,
                forwarded,
                as_responses=True,
                response_slug=client_slug,
                tool_types=tool_types,
                tool_resolve=tool_resolve,
            )
            return await _as_compact_response(response, client_slug)
        if route.is_anthropic:
            forwarded = responses_to_anthropic(compact_body, route.model, route.max_output_tokens)
            forwarded["stream"] = False
            response = await self._post_anthropic(
                request,
                route,
                forwarded,
                as_responses=True,
                response_slug=client_slug,
                tool_types=tool_types,
                tool_resolve=tool_resolve,
            )
            return await _as_compact_response(response, client_slug)
        raise web.HTTPBadGateway(text=f"Unsupported model provider: {route.provider}")

    async def _maybe_passthrough_byok_fallback(
        self,
        request: web.Request,
        body: dict[str, Any],
        *,
        requested: str,
        response_slug: str,
        status: int,
        detail: str,
        compact: bool = False,
    ) -> web.StreamResponse | None:
        fallback_slug = self._passthrough_fallback_slug(requested)
        if not fallback_slug:
            return None
        print(
            f"[fallback] {requested} chatgpt passthrough {status} -> {fallback_slug}: {detail[:200]}",
            flush=True,
        )
        fallback_body = dict(body)
        fallback_body["model"] = fallback_slug
        try:
            if compact:
                return await self._dispatch_byok_compact_responses(
                    request, fallback_body, response_slug=response_slug
                )
            return await self._dispatch_byok_responses(
                request, fallback_body, response_slug=response_slug
            )
        except web.HTTPException as exc:
            print(
                f"[fallback] {requested} -> {fallback_slug} failed ({exc.status}): {exc.text[:200]}",
                flush=True,
            )
            return None

    async def _post_openai_responses(
        self,
        request: web.Request,
        route: ShimModel,
        body: dict[str, Any],
    ) -> web.StreamResponse:
        """Forward a Responses API request directly to the upstream /responses endpoint."""
        url = _join_url(route.base_url, "/responses")
        forwarded = dict(body)
        forwarded["model"] = route.model
        extra_headers = dict(route.extra_headers)
        extra_headers.setdefault("OpenAI-Beta", "responses=2026-02-06")
        headers = openai_upstream_headers(
            request.headers,
            api_key=route.api_key or None,
            extra_headers=extra_headers,
            accept="text/event-stream" if forwarded.get("stream") else None,
        )
        _dump_debug_request(route.slug, url, forwarded)
        log_upstream_request(f"byok-openai-responses:{route.slug}", url, forwarded)
        async with ClientSession(timeout=self.timeout) as session:
            upstream = await session.post(url, json=forwarded, headers=headers)
            if upstream.status >= 400:
                observe_upstream_response(f"byok-openai-responses:{route.slug}", upstream)
                text = await upstream.text()
                log_upstream_response(
                    f"byok-openai-responses:{route.slug}",
                    url,
                    upstream.status,
                    text,
                    request_body=forwarded,
                    stream=bool(forwarded.get("stream")),
                )
                code, message = parse_upstream_error(text, upstream.status)
                upstream.release()
                error = web.json_response(
                    _responses_error_payload(route.slug, code, message),
                    status=upstream.status,
                )
                apply_upstream_headers_to_response(error, upstream_headers_from_response(upstream))
                return error
            if not forwarded.get("stream"):
                payload = await upstream.json(content_type=None)
                usage = payload.get("usage") if isinstance(payload, dict) else None
                observe_upstream_response(
                    f"byok-openai-responses:{route.slug}",
                    upstream,
                    usage=usage if isinstance(usage, dict) else None,
                )
                response = web.json_response(payload)
                apply_upstream_headers_to_response(response, upstream_headers_from_response(upstream))
                log_upstream_response(
                    f"byok-openai-responses:{route.slug}",
                    url,
                    200,
                    json.dumps(payload, default=str)[:12_000] if isinstance(payload, dict) else "",
                    request_body=forwarded,
                )
                upstream.release()
                return response
            observe_upstream_response(f"byok-openai-responses:{route.slug}", upstream)
            log_upstream_response(
                f"byok-openai-responses:{route.slug}",
                url,
                upstream.status,
                request_body=forwarded,
                stream=True,
            )
            response = prepare_downstream_sse_response(upstream)
            await response.prepare(request)
            try:
                async for chunk in _iter_upstream_chunks(upstream.content, request):
                    await _safe_write(response, chunk)
            except ClientDisconnected:
                pass
            finally:
                upstream.release()
            try:
                await response.write_eof()
            except Exception:
                pass
            return response

    async def _post_openai_chat_completions(
        self,
        session: ClientSession,
        route: ShimModel,
        body: dict[str, Any],
        request_headers: Mapping[str, str] | None = None,
    ) -> tuple[Any, tuple[int, str, str, str] | None]:
        url = _join_url(route.base_url, "/chat/completions")
        headers = _openai_headers(
            request_headers or {},
            route,
            accept="text/event-stream" if body.get("stream") else None,
        )
        last_error: tuple[int, str, str, str] | None = None
        for attempt in range(2):
            prepared = prepare_openai_chat_body(route, body)
            log_upstream_request(f"byok-openai-chat:{route.slug}", url, prepared)
            upstream = await session.post(url, json=prepared, headers=headers)
            if upstream.status < 400:
                log_upstream_response(
                    f"byok-openai-chat:{route.slug}",
                    url,
                    upstream.status,
                    request_body=prepared,
                    stream=bool(body.get("stream")),
                )
                return upstream, None
            status = upstream.status
            text = await upstream.text()
            log_upstream_response(
                f"byok-openai-chat:{route.slug}",
                url,
                status,
                text,
                request_body=prepared,
                stream=bool(body.get("stream")),
            )
            code, message = parse_upstream_error(text, status)
            upstream.release()
            last_error = (status, text, code, message)
            if attempt == 0 and learn_parallel_tool_calls_compat_if_needed(route, status, message):
                print(
                    f"[compat] model={route.slug} retrying without parallel_tool_calls after upstream "
                    f"{status}: {message[:200]}",
                    flush=True,
                )
                continue
            break
        return None, last_error

    async def _post_openai_chat(
        self, request: web.Request, route: ShimModel, body: dict[str, Any], as_responses: bool,
        *,
        response_slug: str | None = None,
        tool_types: dict[str, str] | None = None,
        tool_resolve: dict[str, tuple[str | None, str]] | None = None,
    ) -> web.StreamResponse:
        client_slug = response_slug or route.slug
        url = _join_url(route.base_url, "/chat/completions")
        _dump_debug_request(route.slug, url, prepare_openai_chat_body(route, body))
        if body.get("stream"):
            return await self._stream_chat_loop(
                request,
                route,
                body,
                as_responses,
                response_slug=client_slug,
                tool_types=tool_types,
                tool_resolve=tool_resolve,
            )
        async with ClientSession(timeout=self.timeout) as session:
            upstream, err = await self._post_openai_chat_completions(session, route, body, request.headers)
            if err is not None:
                status, text, code, message = err
                if as_responses:
                    error = web.json_response(
                        _responses_error_payload(client_slug, code, message),
                        status=status,
                    )
                    return error
                return web.Response(
                    status=status,
                    text=text,
                    content_type="text/plain",
                )
            upstream_response_headers: Mapping[str, str] = {}
            try:
                payload = await upstream.json(content_type=None)
                usage = payload.get("usage") if isinstance(payload, dict) else None
                observe_upstream_response(
                    f"byok-openai-chat:{route.slug}",
                    upstream,
                    usage=usage if isinstance(usage, dict) else None,
                )
                upstream_response_headers = upstream_headers_from_response(upstream)
            finally:
                upstream.release()
        if as_responses:
            response_payload = chat_completion_to_response(
                payload, client_slug, tool_types, tool_resolve
            )
            response_payload = await mcp_search.augment_response_with_tool_search(response_payload)
            response = web.json_response(response_payload)
            apply_upstream_headers_to_response(response, upstream_response_headers)
            return response
        response = web.json_response(payload)
        apply_upstream_headers_to_response(response, upstream_response_headers)
        return response

    async def _stream_chat_loop(
        self,
        request: web.Request,
        route: ShimModel,
        body: dict[str, Any],
        as_responses: bool,
        *,
        response_slug: str | None = None,
        tool_types: dict[str, str] | None = None,
        tool_resolve: dict[str, tuple[str | None, str]] | None = None,
        max_turns: int = 6,
    ) -> web.StreamResponse:
        client_slug = response_slug or route.slug
        response: web.StreamResponse | None = None
        state = (
            ResponsesStreamState(client_slug, tool_types=tool_types, tool_resolve=tool_resolve)
            if as_responses
            else None
        )
        state_started = False
        messages = list(body.get("messages", []))
        chat_body = {k: v for k, v in body.items() if k != "stream"}
        chat_body["stream"] = True
        upstream_saw_done = False
        try:
            async with ClientSession(timeout=self.timeout) as session:
                for _ in range(max_turns):
                    turn_body = {**chat_body, "messages": messages}
                    upstream, err = await self._post_openai_chat_completions(
                        session, route, turn_body, request.headers
                    )
                    if err is not None:
                        status, _text, code, message = err
                        print(
                            f"[stream] model={route.slug} upstream HTTP {status}: {message[:500]}",
                            flush=True,
                        )
                        if response is None:
                            response = _sse_response()
                            await response.prepare(request)
                        if as_responses and state is not None:
                            if not state_started:
                                await state.start(response)
                                state_started = True
                            await state.fail(response, message, code=code)
                        else:
                            await _safe_write(
                                response,
                                json.dumps({"error": f"upstream {status}: {message[:200]}"}).encode()
                                + b"\n",
                            )
                        break
                    observe_upstream_response(f"byok-openai-chat-stream:{route.slug}", upstream)
                    if response is None:
                        response = prepare_downstream_sse_response(upstream)
                        await response.prepare(request)
                    if as_responses and state is not None and not state_started:
                        await state.start(response)
                        state_started = True
                    try:
                        async for line in _sse_lines(upstream, request):
                            if line == "[DONE]":
                                upstream_saw_done = True
                                break
                            try:
                                event = json.loads(line)
                            except json.JSONDecodeError:
                                continue
                            if isinstance(event, dict) and event.get("error"):
                                if as_responses and state is not None:
                                    code, message = parse_upstream_error(json.dumps(event), 502)
                                    await state.fail(response, message, code=code)
                                    break
                                await _write_sse(response, event)
                                continue
                            if as_responses and state is not None:
                                await state.write_chat_delta(response, event)
                            else:
                                await _write_sse(response, event)
                    except ClientError as exc:
                        print(
                            f"[stream] model={route.slug} upstream stream disconnected: "
                            f"{type(exc).__name__}: {exc}",
                            flush=True,
                        )
                        if as_responses and state is not None and not state.failed:
                            await state.fail(
                                response,
                                f"Upstream stream disconnected: {exc}",
                                code="upstream_disconnect",
                            )
                    finally:
                        await _close_upstream(upstream)
                    break
            if as_responses and state is not None and not state.failed:
                await state.finish(response, upstream_saw_done=upstream_saw_done)
            elif upstream_saw_done:
                await _safe_write(response, b"data: [DONE]\n\n")
        except asyncio.CancelledError:
            print("[cancel] client disconnected during BYOK stream", flush=True)
            raise
        except ClientDisconnected:
            print("[cancel] client disconnected during BYOK stream", flush=True)
        if response is None:
            response = _sse_response()
            await response.prepare(request)
        try:
            await response.write_eof()
        except Exception:
            pass
        return response

    async def _post_openai_chat_as_anthropic(
        self, request: web.Request, route: ShimModel, body: dict[str, Any]
    ) -> web.StreamResponse:
        url = _join_url(route.base_url, "/chat/completions")
        headers = _openai_headers(request.headers, route)
        _dump_debug_request(route.slug, url, body)
        async with ClientSession(timeout=self.timeout) as session:
            upstream = await session.post(url, json=body, headers=headers)
            if upstream.status >= 400:
                observe_upstream_response(f"byok-openai-chat-anthropic:{route.slug}", upstream)
                return await _anthropic_error_response(upstream)
            observe_upstream_response(f"byok-openai-chat-anthropic:{route.slug}", upstream)
            if body.get("stream"):
                return await self._stream_openai_chat_as_anthropic(request, upstream, route)
            payload = await upstream.json(content_type=None)
            upstream_response_headers = upstream_headers_from_response(upstream)
            upstream.release()
        response = web.json_response(chat_completion_to_anthropic_message(payload, route.slug))
        apply_upstream_headers_to_response(response, upstream_response_headers)
        return response

    async def _post_anthropic_messages(
        self, request: web.Request, route: ShimModel, body: dict[str, Any]
    ) -> web.StreamResponse:
        url = _join_url(route.base_url, "/messages")
        headers = _anthropic_headers(request.headers, route)
        log_upstream_request(f"byok-anthropic-messages:{route.slug}", url, body)
        async with ClientSession(timeout=self.timeout) as session:
            upstream = await session.post(url, json=body, headers=headers)
            if upstream.status >= 400:
                observe_upstream_response(f"byok-anthropic-messages:{route.slug}", upstream)
                text = await upstream.text()
                status = upstream.status
                content_type = upstream.content_type or "text/plain"
                log_upstream_response(
                    f"byok-anthropic-messages:{route.slug}",
                    url,
                    status,
                    text,
                    request_body=body,
                    stream=bool(body.get("stream")),
                )
                upstream.release()
                return _upstream_text_response(
                    status,
                    text,
                    content_type=content_type,
                    upstream_headers=upstream_headers_from_response(upstream),
                )
            observe_upstream_response(f"byok-anthropic-messages:{route.slug}", upstream)
            log_upstream_response(
                f"byok-anthropic-messages:{route.slug}",
                url,
                upstream.status,
                request_body=body,
                stream=bool(body.get("stream")),
            )
            if body.get("stream"):
                return await self._stream_raw_sse(request, upstream, route.slug)
            payload = await upstream.json(content_type=None)
            upstream_response_headers = upstream_headers_from_response(upstream)
            upstream.release()
        if isinstance(payload, dict):
            payload["model"] = route.slug
        response = web.json_response(payload)
        apply_upstream_headers_to_response(response, upstream_response_headers)
        return response

    async def _post_anthropic(
        self, request: web.Request, route: ShimModel, body: dict[str, Any], as_responses: bool,
        *,
        response_slug: str | None = None,
        tool_types: dict[str, str] | None = None,
        tool_resolve: dict[str, tuple[str | None, str]] | None = None,
    ) -> web.StreamResponse:
        client_slug = response_slug or route.slug
        url = _join_url(route.base_url, "/messages")
        headers = _anthropic_headers(
            request.headers,
            route,
            accept="text/event-stream" if body.get("stream") else None,
        )
        log_upstream_request(f"byok-anthropic:{route.slug}", url, body)
        async with ClientSession(timeout=self.timeout) as session:
            upstream = await session.post(url, json=body, headers=headers)
            if upstream.status >= 400:
                observe_upstream_response(f"byok-anthropic:{route.slug}", upstream)
                if as_responses and body.get("stream"):
                    return await _stream_responses_upstream_error(
                        request,
                        client_slug,
                        upstream,
                        slug=route.slug,
                    )
                text = await upstream.text()
                status = upstream.status
                content_type = upstream.content_type or "text/plain"
                log_upstream_response(
                    f"byok-anthropic:{route.slug}",
                    url,
                    status,
                    text,
                    request_body=body,
                    stream=bool(body.get("stream")),
                )
                upstream.release()
                return _upstream_text_response(
                    status,
                    text,
                    content_type=content_type,
                    upstream_headers=upstream_headers_from_response(upstream),
                )
            observe_upstream_response(f"byok-anthropic:{route.slug}", upstream)
            log_upstream_response(
                f"byok-anthropic:{route.slug}",
                url,
                upstream.status,
                request_body=body,
                stream=bool(body.get("stream")),
            )
            if body.get("stream"):
                return await self._stream_anthropic(
                    request,
                    upstream,
                    route,
                    as_responses,
                    response_slug=client_slug,
                    tool_types=tool_types,
                    tool_resolve=tool_resolve,
                )
            payload = await upstream.json(content_type=None)
            upstream_response_headers = upstream_headers_from_response(upstream)
            upstream.release()
        if as_responses:
            response = web.json_response(
                anthropic_to_response(payload, client_slug, tool_types, tool_resolve)
            )
            apply_upstream_headers_to_response(response, upstream_response_headers)
            return response
        response = web.json_response(anthropic_to_chat_response(payload, client_slug))
        apply_upstream_headers_to_response(response, upstream_response_headers)
        return response

    async def _stream_openai_chat_as_anthropic(
        self, request: web.Request, upstream, route: ShimModel
    ) -> web.StreamResponse:
        response = prepare_downstream_sse_response(upstream)
        await response.prepare(request)
        state = AnthropicMessagesStreamState(route.slug)
        try:
            await state.start(response)
            async for line in _sse_lines(upstream):
                if line == "[DONE]":
                    break
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                await state.write_chat_delta(response, event)
            await state.finish(response)
        except ClientDisconnected:
            pass
        finally:
            upstream.release()
        try:
            await response.write_eof()
        except Exception:
            pass
        return response

    async def _stream_anthropic(
        self,
        request: web.Request,
        upstream,
        route: ShimModel,
        as_responses: bool,
        *,
        response_slug: str | None = None,
        tool_types: dict[str, str] | None = None,
        tool_resolve: dict[str, tuple[str | None, str]] | None = None,
    ) -> web.StreamResponse:
        client_slug = response_slug or route.slug
        response = prepare_downstream_sse_response(upstream)
        await response.prepare(request)
        if as_responses:
            state = ResponsesStreamState(
                client_slug, tool_types=tool_types, tool_resolve=tool_resolve
            )
        upstream_saw_done = False
        try:
            if as_responses:
                await state.start(response)
            try:
                async for line in _sse_lines(upstream, request):
                    if line == "[DONE]":
                        upstream_saw_done = True
                        break
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(event, dict) and event.get("error"):
                        if as_responses and state is not None:
                            code, message = parse_upstream_error(json.dumps(event), 502)
                            await state.fail(response, message, code=code)
                            break
                        await _write_sse(response, event)
                        continue
                    if as_responses:
                        await state.write_anthropic_delta(response, event)
                    else:
                        await _write_sse(response, _anthropic_stream_to_chat_chunk(event, client_slug))
            except ClientError as exc:
                print(
                    f"[stream] model={route.slug} upstream stream disconnected: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
                if as_responses and not state.failed:
                    await state.fail(
                        response,
                        f"Upstream stream disconnected: {exc}",
                        code="upstream_disconnect",
                    )
            if as_responses and not state.failed:
                await state.finish(response, upstream_saw_done=upstream_saw_done)
            elif upstream_saw_done:
                await _safe_write(response, b"data: [DONE]\n\n")
        except asyncio.CancelledError:
            print("[cancel] client disconnected during Anthropic stream", flush=True)
            raise
        except ClientDisconnected:
            print("[cancel] client disconnected during Anthropic stream", flush=True)
        finally:
            await _close_upstream(upstream)
        try:
            await response.write_eof()
        except Exception:
            pass
        return response

    async def _stream_raw_sse(self, request: web.Request, upstream, model_slug: str | None = None) -> web.StreamResponse:
        response = prepare_downstream_sse_response(upstream)
        await response.prepare(request)
        try:
            async for line in _sse_lines(upstream):
                if model_slug and line.startswith("{"):
                    try:
                        event = json.loads(line)
                        if isinstance(event, dict) and event.get("type") == "message_start":
                            msg = event.get("message")
                            if isinstance(msg, dict):
                                msg["model"] = model_slug
                        await _write_anthropic_sse(response, event.get("type", "message"), event)
                        continue
                    except json.JSONDecodeError:
                        pass
                await _safe_write(response, f"data: {line}\n\n".encode())
        except ClientDisconnected:
            pass
        finally:
            upstream.release()
        try:
            await response.write_eof()
        except Exception:
            pass
        return response


_DROP_ITEM = object()


def _chatgpt_input_items(input_value: Any) -> list[Any]:
    if isinstance(input_value, list):
        return copy.deepcopy(input_value)
    if input_value is None:
        return []
    return [{"type": "message", "role": "user", "content": copy.deepcopy(input_value)}]


class ChatgptPassthroughResponseCollector:
    def __init__(self, forwarded: dict[str, Any]):
        self._input = _chatgpt_input_items(forwarded.get("input"))
        self.response_id: str | None = None
        self._output: list[Any] = []

    def record(self, event: dict[str, Any]) -> None:
        response = event.get("response")
        if isinstance(response, dict):
            response_id = response.get("id")
            if isinstance(response_id, str):
                self.response_id = response_id
            output = response.get("output")
            if isinstance(output, list) and output:
                self._output = copy.deepcopy(output)
                return
        if event.get("type") == "response.output_item.done":
            item = event.get("item")
            if isinstance(item, dict):
                self._output.append(copy.deepcopy(item))

    def conversation_items(self) -> list[Any]:
        return [*copy.deepcopy(self._input), *copy.deepcopy(self._output)]


def _should_cache_chatgpt_passthrough_event(event: dict[str, Any]) -> bool:
    if event.get("type") in {"response.output_item.done", "response.completed"}:
        return True
    response = event.get("response")
    return isinstance(response, dict) and bool(response.get("output"))


def _sanitize_chatgpt_passthrough_body(body: dict[str, Any], *, strip_previous_response_id: bool = False) -> dict[str, Any]:
    sanitized = _sanitize_chatgpt_passthrough_value(body)
    if not isinstance(sanitized, dict):
        return {}
    if strip_previous_response_id:
        sanitized.pop("previous_response_id", None)
    return _finalize_chatgpt_passthrough_body(sanitized)


_CHATGPT_UNSUPPORTED_REQUEST_KEYS = (
    "max_output_tokens",
    "max_tokens",
    "service_tier",
)


def _finalize_chatgpt_passthrough_body(body: dict[str, Any]) -> dict[str, Any]:
    forwarded = dict(body)
    for key in _CHATGPT_UNSUPPORTED_REQUEST_KEYS:
        forwarded.pop(key, None)
    forwarded["store"] = False
    return forwarded


_CHATGPT_COMPACT_UNSUPPORTED_REQUEST_KEYS = (
    *_CHATGPT_UNSUPPORTED_REQUEST_KEYS,
    "store",
    "stream",
)


def _finalize_chatgpt_compact_passthrough_body(body: dict[str, Any]) -> dict[str, Any]:
    forwarded = dict(body)
    for key in _CHATGPT_COMPACT_UNSUPPORTED_REQUEST_KEYS:
        forwarded.pop(key, None)
    return forwarded


def _sanitize_chatgpt_compact_passthrough_body(body: dict[str, Any]) -> dict[str, Any]:
    sanitized = _sanitize_chatgpt_passthrough_value(body)
    if not isinstance(sanitized, dict):
        return {}
    return _finalize_chatgpt_compact_passthrough_body(sanitized)


def _sanitize_chatgpt_passthrough_value(value: Any) -> Any:
    if isinstance(value, list):
        output = []
        for item in value:
            sanitized = _sanitize_chatgpt_passthrough_value(item)
            if sanitized is not _DROP_ITEM:
                output.append(sanitized)
        return output
    if isinstance(value, dict):
        if value.get("type") == "reasoning" and _has_shim_encrypted_content(value):
            return _DROP_ITEM
        if value.get("type") == "compaction":
            replaced = _replace_shim_compaction_for_chatgpt(value)
            if replaced is not None:
                return replaced
            return _DROP_ITEM
        output = {}
        for key, item in value.items():
            if key == "encrypted_content" and isinstance(item, str) and _is_shim_opaque_encrypted_content(item):
                continue
            sanitized = _sanitize_chatgpt_passthrough_value(item)
            if sanitized is not _DROP_ITEM:
                output[key] = sanitized
        return output
    return value


def _is_shim_opaque_encrypted_content(value: str) -> bool:
    return value.startswith(SHIM_ENCRYPTED_CONTENT_PREFIX) or value.startswith(SHIM_COMPACTION_PREFIX)


def _replace_shim_compaction_for_chatgpt(value: dict[str, Any]) -> Any:
    encrypted = value.get("encrypted_content")
    if not isinstance(encrypted, str) or not encrypted.startswith(SHIM_COMPACTION_PREFIX):
        return None
    summary = decode_shim_compaction_summary(encrypted)
    if not summary:
        return _DROP_ITEM
    return {
        "type": "message",
        "role": "developer",
        "content": [{"type": "input_text", "text": f"Compacted conversation state:\n{summary}"}],
    }


def _has_shim_encrypted_content(value: dict[str, Any]) -> bool:
    encrypted_content = value.get("encrypted_content")
    return isinstance(encrypted_content, str) and encrypted_content.startswith(SHIM_ENCRYPTED_CONTENT_PREFIX)


def _rewrite_response_model(payload: Any, model: str | None) -> None:
    if not model:
        return
    if isinstance(payload, dict):
        if payload.get("model") == CHATGPT_MODEL_SLUG:
            payload["model"] = model
        for value in payload.values():
            _rewrite_response_model(value, model)
    elif isinstance(payload, list):
        for item in payload:
            _rewrite_response_model(item, model)


# Desktop accumulates `response.reasoning_summary_text.delta` during the turn.
# llama.cpp often emits reasoning in one large chunk; split so the UI updates live.
_REASONING_DELTA_CHUNK_CHARS = 80


def _iter_reasoning_delta_chunks(text: str, chunk_size: int = _REASONING_DELTA_CHUNK_CHARS) -> list[str]:
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start = end
    return chunks


def _tool_call_arguments_complete(tc: dict[str, Any]) -> bool:
    name = tc.get("name") or ""
    if not name:
        return False
    args = tc.get("arguments") or ""
    if not isinstance(args, str) or not args.strip():
        return False
    if tc.get("output_type") == "custom_tool_call":
        return True
    try:
        json.loads(args)
    except json.JSONDecodeError:
        return False
    return True


def _responses_output_type_for_tool(name: str, tool_types: dict[str, str] | None = None) -> str:
    original_type = original_responses_tool_type(name, tool_types)
    if original_type == "apply_patch":
        return "custom_tool_call"
    if original_type.startswith("web_search"):
        return "web_search_call"
    return "function_call"


def _stream_tool_added_item(state: dict[str, Any], status: str = "in_progress") -> dict[str, Any]:
    output_type = state.get("output_type", "function_call")
    if output_type == "custom_tool_call":
        return {
            "id": state["id"],
            "type": "custom_tool_call",
            "status": status,
            "call_id": state["call_id"],
            "name": state["name"],
            "input": "" if status == "in_progress" else state.get("arguments", ""),
        }
    if output_type == "web_search_call":
        return _web_search_stream_item(state, status)
    item: dict[str, Any] = {
        "id": state["id"],
        "type": "function_call",
        "status": status,
        "call_id": state["call_id"],
        "name": state["name"],
        "arguments": "" if status == "in_progress" else state.get("arguments", ""),
    }
    if state.get("namespace"):
        item["namespace"] = state["namespace"]
    return item


def _web_search_stream_item(state: dict[str, Any], status: str) -> dict[str, Any]:
    query = ""
    raw_arguments = state.get("arguments") or ""
    try:
        parsed = json.loads(raw_arguments) if raw_arguments else {}
    except json.JSONDecodeError:
        parsed = {"query": raw_arguments}
    if isinstance(parsed, dict):
        query = str(parsed.get("query") or parsed.get("q") or parsed.get("search_query") or "")
    return {
        "id": state["id"],
        "type": "web_search_call",
        "status": status,
        "call_id": state["call_id"],
        "action": {"type": "search", "query": query},
    }


def _should_defer_tool_name(name: str) -> bool:
    if not name:
        return True
    if mcp_search.is_tool_search_call(name) or mcp_search.parse_mcp_tool_reference(name):
        return False
    return name.startswith("mcp")


class AnthropicMessagesStreamState:
    """Translates OpenAI chat-completions chunks into Anthropic Messages SSE."""

    def __init__(self, model: str):
        self.message_id = f"msg_{uuid.uuid4().hex[:24]}"
        self.model = model
        self.next_index = 0
        self.text_index: int | None = None
        self.reasoning_index: int | None = None
        self.text_open = False
        self.reasoning_open = False
        self.tool_calls: dict[int, dict[str, Any]] = {}
        self.usage: dict[str, Any] | None = None
        self.stop_reason = "end_turn"

    async def start(self, response: web.StreamResponse) -> None:
        await _write_anthropic_sse(
            response,
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": self.message_id,
                    "type": "message",
                    "role": "assistant",
                    "model": self.model,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                },
            },
        )

    async def write_chat_delta(self, response: web.StreamResponse, chunk: dict[str, Any]) -> None:
        usage = chunk.get("usage")
        if isinstance(usage, dict):
            self.usage = normalize_responses_usage(usage)
        choice = (chunk.get("choices") or [{}])[0]
        finish_reason = choice.get("finish_reason")
        if finish_reason:
            self.stop_reason = _chat_finish_to_anthropic_stop(finish_reason)
        delta = choice.get("delta") or {}
        reasoning = delta.get("reasoning_content") or delta.get("reasoning")
        if reasoning:
            await self._reasoning_delta(response, str(reasoning))
        content = delta.get("content")
        if content:
            if self.reasoning_open:
                await self._close_reasoning(response)
            await self._text_delta(response, str(content))
        for call in delta.get("tool_calls") or []:
            await self._tool_delta(response, call)

    async def finish(self, response: web.StreamResponse) -> None:
        if self.reasoning_open:
            await self._close_reasoning(response)
        if self.text_open:
            await self._close_text(response)
        for index in sorted(self.tool_calls):
            state = self.tool_calls[index]
            if not state.get("closed"):
                await self._close_tool(response, index, state)
        await _write_anthropic_sse(
            response,
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": self.stop_reason, "stop_sequence": None},
                "usage": _responses_usage_to_anthropic_usage(self.usage) or {"output_tokens": 0},
            },
        )
        await _write_anthropic_sse(response, "message_stop", {"type": "message_stop"})

    async def _text_delta(self, response: web.StreamResponse, text: str) -> None:
        if self.text_index is None:
            self.text_index = self.next_index
            self.next_index += 1
            self.text_open = True
            await _write_anthropic_sse(
                response,
                "content_block_start",
                {"type": "content_block_start", "index": self.text_index, "content_block": {"type": "text", "text": ""}},
            )
        await _write_anthropic_sse(
            response,
            "content_block_delta",
            {"type": "content_block_delta", "index": self.text_index, "delta": {"type": "text_delta", "text": text}},
        )

    async def _close_text(self, response: web.StreamResponse) -> None:
        if self.text_index is None:
            return
        await _write_anthropic_sse(response, "content_block_stop", {"type": "content_block_stop", "index": self.text_index})
        self.text_index = None
        self.text_open = False

    async def _reasoning_delta(self, response: web.StreamResponse, text: str) -> None:
        if self.reasoning_index is None:
            self.reasoning_index = self.next_index
            self.next_index += 1
            self.reasoning_open = True
            await _write_anthropic_sse(
                response,
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": self.reasoning_index,
                    "content_block": {"type": "thinking", "thinking": ""},
                },
            )
        await _write_anthropic_sse(
            response,
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": self.reasoning_index,
                "delta": {"type": "thinking_delta", "thinking": text},
            },
        )

    async def _close_reasoning(self, response: web.StreamResponse) -> None:
        if self.reasoning_index is None:
            return
        await _write_anthropic_sse(
            response,
            "content_block_stop",
            {"type": "content_block_stop", "index": self.reasoning_index},
        )
        self.reasoning_index = None
        self.reasoning_open = False

    async def _tool_delta(self, response: web.StreamResponse, call: dict[str, Any]) -> None:
        index = int(call.get("index", 0))
        fn = call.get("function") or {}
        state = self.tool_calls.setdefault(
            index,
            {
                "id": "",
                "name": "",
                "arguments": "",
                "emitted": 0,
                "block_index": None,
                "open": False,
                "closed": False,
            },
        )
        if call.get("id"):
            state["id"] = call["id"]
        if fn.get("name"):
            state["name"] += fn["name"]
        if fn.get("arguments"):
            state["arguments"] += fn["arguments"]
        if not state["open"] and state["name"]:
            if self.reasoning_open:
                await self._close_reasoning(response)
            if self.text_open:
                await self._close_text(response)
            await self._open_tool(response, index, state)
        if state["open"] and len(state["arguments"]) > state["emitted"]:
            delta = state["arguments"][state["emitted"] :]
            state["emitted"] = len(state["arguments"])
            await _write_anthropic_sse(
                response,
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": state["block_index"],
                    "delta": {"type": "input_json_delta", "partial_json": delta},
                },
            )

    async def _open_tool(self, response: web.StreamResponse, index: int, state: dict[str, Any]) -> None:
        state["block_index"] = self.next_index
        self.next_index += 1
        state["open"] = True
        if not state["id"]:
            state["id"] = f"call_{index}"
        await _write_anthropic_sse(
            response,
            "content_block_start",
            {
                "type": "content_block_start",
                "index": state["block_index"],
                "content_block": {
                    "type": "tool_use",
                    "id": state["id"],
                    "name": state["name"] or "tool",
                    "input": {},
                },
            },
        )

    async def _close_tool(self, response: web.StreamResponse, index: int, state: dict[str, Any]) -> None:
        if not state["open"]:
            await self._open_tool(response, index, state)
            if len(state["arguments"]) > state["emitted"]:
                delta = state["arguments"][state["emitted"] :]
                state["emitted"] = len(state["arguments"])
                await _write_anthropic_sse(
                    response,
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": state["block_index"],
                        "delta": {"type": "input_json_delta", "partial_json": delta},
                    },
                )
        await _write_anthropic_sse(
            response,
            "content_block_stop",
            {"type": "content_block_stop", "index": state["block_index"]},
        )
        state["open"] = False
        state["closed"] = True


class ResponsesStreamState:
    """Translates upstream chat-completions / anthropic stream events into the
    Codex Desktop Responses-API event sequence. Keeps the message item and
    each tool call as separate output items with stable indices, and emits
    proper .added / .delta / .done / .completed events plus a final
    proper .added / .delta / .done / .completed events. Emits ``response.completed``
    and client ``[DONE]`` only when upstream sent ``[DONE]`` and every open item
    was fully received."""

    def __init__(
        self,
        model: str,
        tool_types: dict[str, str] | None = None,
        tool_resolve: dict[str, tuple[str | None, str]] | None = None,
    ):
        self.response_id = f"resp_{int(time.time() * 1000)}"
        self.message_item_id = f"msg_{int(time.time() * 1000)}"
        self.model = model
        self.failed = False
        self.message_index: int | None = None  # output_index for the assistant message
        self.message_text = ""
        self.message_opened = False
        self.message_closed = False
        self.usage: dict[str, Any] | None = None
        self.tool_calls: dict[int, dict[str, Any]] = {}
        self.mcp_tool_calls: dict[int, dict[str, Any]] = {}
        self.tool_search_calls: dict[int, dict[str, Any]] = {}
        # Reasoning (extended thinking) blocks, keyed by upstream index.
        self.reasoning_blocks: dict[Any, dict[str, Any]] = {}
        self.finished_messages: list[tuple[int, dict[str, Any]]] = []
        self.next_output_index = 0
        self.completed_turns: list[dict[str, Any]] = []
        self.upstream_finish_reason: str | None = None
        self.tool_types = tool_types or {}
        self.tool_resolve = tool_resolve or {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    async def start(self, response: web.StreamResponse) -> None:
        await _write_sse(response, {"type": "response.created", "response": self._response("in_progress")})

    async def fail(
        self,
        response: web.StreamResponse,
        message: str,
        *,
        code: str = "upstream_error",
    ) -> None:
        if self.failed:
            return
        self.failed = True
        await self.close_turn_items(response)
        await _write_sse(
            response,
            {
                "type": "error",
                "code": code,
                "message": message,
            },
        )
        failed_response = self._response("failed", final=True)
        failed_response["error"] = {"code": code, "message": message}
        await _write_sse(
            response,
            {"type": "response.failed", "response": failed_response},
        )
        await response.write(b"data: [DONE]\n\n")

    async def finish(self, response: web.StreamResponse, *, upstream_saw_done: bool) -> None:
        if self.failed:
            return
        await self.close_turn_items(response)
        if upstream_saw_done and not self._has_open_incomplete_items():
            await _write_sse(
                response,
                {"type": "response.completed", "response": self._response("completed", final=True)},
            )
            await response.write(b"data: [DONE]\n\n")
        elif upstream_saw_done:
            self._log_incomplete_tool_calls()
            incomplete_response = self._response("incomplete", final=True)
            if self.upstream_finish_reason == "length":
                incomplete_response["incomplete_details"] = {"reason": "max_output_tokens"}
            await _write_sse(
                response,
                {"type": "response.incomplete", "response": incomplete_response},
            )
            await response.write(b"data: [DONE]\n\n")
        elif self._has_open_incomplete_items():
            self._log_incomplete_tool_calls()

    def _has_open_incomplete_items(self) -> bool:
        if self.message_opened and not self.message_closed:
            return True
        for state in self.reasoning_blocks.values():
            if not state.get("closed"):
                return True
        for tc in (
            *self.tool_calls.values(),
            *self.mcp_tool_calls.values(),
            *self.tool_search_calls.values(),
        ):
            if tc.get("closed"):
                continue
            if not _tool_call_arguments_complete(tc):
                return True
        return False

    async def close_turn_items(self, response: web.StreamResponse) -> None:
        for state in sorted(self.reasoning_blocks.values(), key=lambda s: s["output_index"]):
            if not state.get("closed"):
                await self._close_reasoning(response, state)
        if self.message_opened and not self.message_closed and self.message_text:
            await self._close_message(response)
        for state in sorted(self.tool_calls.values(), key=lambda s: s["output_index"]):
            if state.get("closed"):
                continue
            if mcp_search.is_tool_search_call(state.get("name") or ""):
                continue
            if mcp_search.parse_mcp_tool_reference(state.get("name") or ""):
                continue
            if _tool_call_arguments_complete(state):
                await self._close_tool(response, state)
        for state in sorted(self.mcp_tool_calls.values(), key=lambda s: s["output_index"]):
            if not state.get("closed") and _tool_call_arguments_complete(state):
                await self._close_mcp_tool(response, state)
        for state in sorted(self.tool_search_calls.values(), key=lambda s: s["output_index"]):
            if not state.get("closed") and _tool_call_arguments_complete(state):
                await self._close_tool_search(response, state)

    def _log_incomplete_tool_calls(self) -> None:
        incomplete: list[str] = []
        for tc in (
            *self.tool_calls.values(),
            *self.mcp_tool_calls.values(),
            *self.tool_search_calls.values(),
        ):
            if tc.get("closed"):
                continue
            name = tc.get("name") or "?"
            args = tc.get("arguments") or ""
            if _tool_call_arguments_complete(tc):
                continue
            incomplete.append(f"{name}({len(args)} arg chars)")
        if incomplete:
            print(
                f"[stream] model={self.model} incomplete_tool_calls_at_stream_end={incomplete}",
                flush=True,
            )

    def snapshot_turn(self) -> None:
        self.completed_turns.append(self._current_turn_dict())

    def reset_for_next_turn(self) -> None:
        turn_num = len(self.completed_turns)
        self.message_item_id = f"msg_{int(time.time() * 1000)}_{turn_num}"
        self.message_index = None
        self.message_text = ""
        self.message_opened = False
        self.message_closed = False
        self.tool_calls = {}
        self.mcp_tool_calls = {}
        self.tool_search_calls = {}
        self.reasoning_blocks = {}
        self.finished_messages = []

    def _current_turn_dict(self) -> dict[str, Any]:
        return {
            "message_item_id": self.message_item_id,
            "message_index": self.message_index,
            "message_text": self.message_text,
            "message_opened": self.message_opened,
            "message_closed": self.message_closed,
            "tool_calls": dict(self.tool_calls),
            "mcp_tool_calls": dict(self.mcp_tool_calls),
            "tool_search_calls": dict(self.tool_search_calls),
            "reasoning_blocks": dict(self.reasoning_blocks),
        }

    # ------------------------------------------------------------------
    # Chat-completions (OpenAI-style) deltas
    # ------------------------------------------------------------------
    async def write_chat_delta(
        self,
        response: web.StreamResponse,
        chunk: dict[str, Any],
    ) -> None:
        usage = chunk.get("usage")
        if isinstance(usage, dict):
            self.usage = normalize_responses_usage(usage)
        choice = (chunk.get("choices") or [{}])[0]
        finish_reason = choice.get("finish_reason")
        if finish_reason is not None:
            self.upstream_finish_reason = str(finish_reason)
        delta = choice.get("delta") or {}
        reasoning = delta.get("reasoning_content") or delta.get("reasoning")
        if reasoning:
            await self._chat_reasoning_delta(response, reasoning)
        for call in delta.get("tool_calls") or []:
            await self._chat_tool_delta(response, call)
        message = choice.get("message") or {}
        for call in message.get("tool_calls") or []:
            await self._chat_tool_delta(response, call)
        content = delta.get("content")
        if content:
            for state in list(self.reasoning_blocks.values()):
                if not state.get("closed"):
                    await self._close_reasoning(response, state)
            await self._text_delta(response, content)

    async def _emit_reasoning_summary_deltas(
        self, response: web.StreamResponse, state: dict[str, Any], text: str
    ) -> None:
        for delta in _iter_reasoning_delta_chunks(text):
            await _write_sse(
                response,
                {
                    "type": "response.reasoning_summary_text.delta",
                    "item_id": state["id"],
                    "output_index": state["output_index"],
                    "summary_index": 0,
                    "delta": delta,
                },
            )

    async def _chat_reasoning_delta(self, response: web.StreamResponse, text: str) -> None:
        state = self.reasoning_blocks.get(("chat",))
        if state is None:
            state = await self._open_reasoning(response, key=("chat",))
        state["text"] += text
        await self._emit_reasoning_summary_deltas(response, state, text)

    async def _chat_tool_delta(
        self,
        response: web.StreamResponse,
        call: dict[str, Any],
    ) -> None:
        index = int(call.get("index", 0))
        fn = call.get("function") or {}
        pending = self.tool_calls.get(index)
        if pending is not None and not pending.get("emitted"):
            pending["name"] = mcp_search.normalize_upstream_tool_name(
                (pending.get("name") or "") + (fn.get("name") or "")
            )
            arg_delta = fn.get("arguments") or ""
            if arg_delta:
                pending["arguments"] += arg_delta
            await self._finalize_pending_tool(response, index)
            if index in self.mcp_tool_calls:
                await self._chat_mcp_tool_delta(
                    response,
                    index,
                    call,
                    fn,
                    self.mcp_tool_calls[index]["name"],
                )
                return
            if index in self.tool_search_calls:
                await self._chat_tool_search_delta(
                    response,
                    index,
                    call,
                    fn,
                    self.tool_search_calls[index]["name"],
                )
                return
            pending = self.tool_calls.get(index)
            if pending is None or not pending.get("emitted"):
                return

        existing = (
            self.tool_calls.get(index)
            or self.mcp_tool_calls.get(index)
            or self.tool_search_calls.get(index)
        )
        name = existing.get("name") if existing else mcp_search.normalize_upstream_tool_name(fn.get("name") or "")
        if existing is not None and fn.get("name") and existing is not pending:
            existing["name"] = mcp_search.normalize_upstream_tool_name(
                (existing.get("name") or "") + (fn.get("name") or "")
            )
            name = existing["name"]
        if mcp_search.is_tool_search_call(name) or index in self.tool_search_calls:
            await self._chat_tool_search_delta(response, index, call, fn, name)
            return
        if mcp_search.parse_mcp_tool_reference(name) or index in self.mcp_tool_calls:
            await self._chat_mcp_tool_delta(response, index, call, fn, name)
            return
        state = self.tool_calls.get(index)
        if state is None:
            call_id = call.get("id") or f"call_{index}"
            if _should_defer_tool_name(name):
                self.tool_calls[index] = {
                    "id": call_id,
                    "call_id": call_id,
                    "name": name,
                    "arguments": fn.get("arguments") or "",
                    "output_type": _responses_output_type_for_tool(name, self.tool_types),
                    "closed": False,
                    "emitted": False,
                }
                await self._finalize_pending_tool(response, index)
                return
            namespace, tool_name = resolve_namespaced_tool_name(name, self.tool_resolve)
            state = await self._open_tool(
                response,
                key=index,
                call_id=call_id,
                name=tool_name,
                namespace=namespace,
            )
            state["emitted"] = True
            arg_delta = fn.get("arguments") or ""
            if arg_delta:
                state["arguments"] = arg_delta
                await _write_sse(
                    response,
                    {
                        "type": "response.function_call_arguments.delta",
                        "item_id": state["id"],
                        "output_index": state["output_index"],
                        "delta": arg_delta,
                    },
                )
        else:
            if fn.get("name"):
                state["name"] = name
            arg_delta = fn.get("arguments") or ""
            if arg_delta:
                state["arguments"] += arg_delta
            if not state.get("emitted"):
                await self._finalize_pending_tool(response, index)
                state = self.tool_calls.get(index)
                if state is None or not state.get("emitted"):
                    return
            if arg_delta:
                await _write_sse(
                    response,
                    {
                        "type": "response.function_call_arguments.delta",
                        "item_id": state["id"],
                        "output_index": state["output_index"],
                        "delta": arg_delta,
                    },
                )
        state = self.tool_calls.get(index)
        if (
            state
            and state.get("emitted")
            and _tool_call_arguments_complete(state)
            and not state.get("closed")
        ):
            await self._close_tool(response, state)

    async def _finalize_pending_tool(self, response: web.StreamResponse, index: int) -> None:
        state = self.tool_calls.get(index)
        if state is None or state.get("emitted"):
            return
        name = state.get("name") or ""
        if mcp_search.is_tool_search_call(name):
            pending = self.tool_calls.pop(index)
            self.tool_search_calls[index] = {
                **pending,
                "opened": False,
                "closed": False,
            }
            return
        if mcp_search.parse_mcp_tool_reference(name):
            pending = self.tool_calls.pop(index)
            self.mcp_tool_calls[index] = {
                **pending,
                "opened": False,
                "closed": False,
            }
            return
        if _should_defer_tool_name(name):
            return
        if self.message_opened and not self.message_closed:
            await self._close_message(response)
        output_index = self.next_output_index
        self.next_output_index += 1
        namespace, tool_name = resolve_namespaced_tool_name(name, self.tool_resolve)
        if namespace is not None:
            state["namespace"] = namespace
            state["name"] = tool_name
            name = tool_name
        state.update(
            {
                "output_index": output_index,
                "emitted": True,
                "output_type": _responses_output_type_for_tool(name, self.tool_types),
            }
        )
        item = _stream_tool_added_item(state)
        await _write_sse(
            response,
            {
                "type": "response.output_item.added",
                "output_index": output_index,
                "item": item,
            },
        )
        if _tool_call_arguments_complete(state) and not state.get("closed"):
            await self._close_tool(response, state)

    async def _chat_tool_search_delta(
        self,
        response: web.StreamResponse,
        index: int,
        call: dict[str, Any],
        fn: dict[str, Any],
        name: str,
    ) -> None:
        state = self.tool_search_calls.get(index)
        if state is None:
            call_id = call.get("id") or f"call_{index}"
            state = {
                "id": call_id,
                "call_id": call_id,
                "name": fn.get("name") or "",
                "arguments": "",
                "closed": False,
                "opened": False,
            }
            self.tool_search_calls[index] = state
        elif fn.get("name"):
            state["name"] += fn.get("name")
        if not state.get("opened") and mcp_search.is_tool_search_call(state.get("name") or ""):
            state = await self._open_tool_search(response, index, state)
        arg_delta = fn.get("arguments") or ""
        if arg_delta:
            state["arguments"] += arg_delta
        if _tool_call_arguments_complete(state) and not state.get("closed"):
            await self._close_tool_search(response, state)

    async def _open_tool_search(
        self,
        response: web.StreamResponse,
        index: int,
        state: dict[str, Any],
    ) -> dict[str, Any]:
        for reasoning in list(self.reasoning_blocks.values()):
            if not reasoning.get("closed"):
                await self._close_reasoning(response, reasoning)
        output_index = self.next_output_index
        self.next_output_index += 1
        state.update({"output_index": output_index, "opened": True})
        self.tool_search_calls[index] = state
        await _write_sse(
            response,
            {
                "type": "response.output_item.added",
                "output_index": output_index,
                "item": tool_translate.tool_search_call_from_raw(
                    state["id"],
                    "",
                    "in_progress",
                ),
            },
        )
        return state

    async def _close_tool_search(
        self, response: web.StreamResponse, state: dict[str, Any]
    ) -> None:
        if state.get("closed"):
            return
        if not state.get("opened"):
            for index, candidate in self.tool_search_calls.items():
                if candidate is state:
                    state = await self._open_tool_search(response, index, state)
                    break
        state["closed"] = True
        raw_arguments = state.get("arguments") or ""
        await _write_sse(
            response,
            {
                "type": "response.output_item.done",
                "output_index": state["output_index"],
                "item": tool_translate.tool_search_call_from_raw(
                    state["id"],
                    raw_arguments,
                    "completed",
                ),
            },
        )

    async def _chat_mcp_tool_delta(
        self,
        response: web.StreamResponse,
        index: int,
        call: dict[str, Any],
        fn: dict[str, Any],
        name: str,
    ) -> None:
        state = self.mcp_tool_calls.get(index)
        if state is None:
            call_id = call.get("id") or f"call_{index}"
            state = {
                "id": call_id,
                "call_id": call_id,
                "name": name or fn.get("name") or "",
                "arguments": "",
                "closed": False,
                "opened": False,
            }
            self.mcp_tool_calls[index] = state
        elif mcp_search.parse_mcp_tool_reference(name):
            state["name"] = name
        elif fn.get("name"):
            state["name"] = mcp_search.normalize_upstream_tool_name(
                (state.get("name") or "") + fn.get("name")
            )
        if not state.get("opened") and mcp_search.parse_mcp_tool_reference(state.get("name") or ""):
            state = await self._open_mcp_tool(response, index, state)
        arg_delta = fn.get("arguments") or ""
        if arg_delta:
            state["arguments"] += arg_delta
            if state.get("opened"):
                await _write_sse(
                    response,
                    {
                        "type": "response.function_call_arguments.delta",
                        "item_id": state["id"],
                        "output_index": state["output_index"],
                        "delta": arg_delta,
                    },
                )
        if _tool_call_arguments_complete(state) and not state.get("closed"):
            await self._close_mcp_tool(response, state)

    async def _open_mcp_tool(
        self,
        response: web.StreamResponse,
        index: int,
        state: dict[str, Any],
    ) -> dict[str, Any]:
        parsed = mcp_search.parse_mcp_tool_reference(state.get("name") or "")
        if parsed is None:
            return state
        server, tool = parsed
        for reasoning in list(self.reasoning_blocks.values()):
            if not reasoning.get("closed"):
                await self._close_reasoning(response, reasoning)
        output_index = self.next_output_index
        self.next_output_index += 1
        state.update(
            {
                "output_index": output_index,
                "server": server,
                "tool": tool,
                "opened": True,
            }
        )
        self.mcp_tool_calls[index] = state
        await _write_sse(
            response,
            {
                "type": "response.output_item.added",
                "output_index": output_index,
                "item": tool_translate.mcp_function_call_item(
                    state["id"],
                    server,
                    tool,
                    "",
                    "in_progress",
                ),
            },
        )
        prior_args = state.get("arguments") or ""
        if prior_args:
            await _write_sse(
                response,
                {
                    "type": "response.function_call_arguments.delta",
                    "item_id": state["id"],
                    "output_index": output_index,
                    "delta": prior_args,
                },
            )
        return state

    async def _close_mcp_tool(
        self, response: web.StreamResponse, state: dict[str, Any]
    ) -> None:
        if state.get("closed"):
            return
        if not _tool_call_arguments_complete(state):
            return
        if not state.get("opened"):
            for index, candidate in self.mcp_tool_calls.items():
                if candidate is state:
                    state = await self._open_mcp_tool(response, index, state)
                    break
        state["closed"] = True
        arguments = state.get("arguments") or ""
        await _write_sse(
            response,
            {
                "type": "response.function_call_arguments.done",
                "item_id": state["id"],
                "output_index": state["output_index"],
                "arguments": arguments,
            },
        )
        await _write_sse(
            response,
            {
                "type": "response.output_item.done",
                "output_index": state["output_index"],
                "item": tool_translate.mcp_function_call_item(
                    state["id"],
                    state["server"],
                    state["tool"],
                    arguments,
                    "completed",
                ),
            },
        )

    def all_tool_calls(self) -> dict[int, dict[str, Any]]:
        return dict(self.tool_calls)

    # ------------------------------------------------------------------
    # Anthropic deltas
    # ------------------------------------------------------------------
    async def write_anthropic_delta(self, response: web.StreamResponse, event: dict[str, Any]) -> None:
        event_type = event.get("type")
        if event_type == "message_start":
            message = event.get("message") or {}
            usage = message.get("usage")
            if isinstance(usage, dict):
                self.usage = normalize_responses_usage(usage)
        if event_type == "content_block_start":
            block = event.get("content_block") or {}
            idx = int(event.get("index", 0))
            btype = block.get("type")
            if btype == "text":
                seed = block.get("text") or ""
                if seed:
                    await self._text_delta(response, seed)
            elif btype == "tool_use":
                raw_name = block.get("name") or ""
                namespace, tool_name = resolve_namespaced_tool_name(raw_name, self.tool_resolve)
                await self._open_tool(
                    response,
                    key=("anthropic", idx),
                    call_id=block.get("id") or f"call_{idx}",
                    name=tool_name,
                    namespace=namespace,
                )
            elif btype in {"thinking", "redacted_thinking"}:
                await self._open_reasoning(
                    response,
                    key=("anthropic_thinking", idx),
                    initial_text=block.get("thinking") or "",
                    initial_signature=block.get("signature") or "",
                    redacted=(btype == "redacted_thinking"),
                    redacted_data=block.get("data") or "",
                )
        elif event_type == "content_block_delta":
            idx = int(event.get("index", 0))
            delta = event.get("delta") or {}
            dtype = delta.get("type")
            if dtype == "text_delta":
                await self._text_delta(response, delta.get("text", ""))
            elif dtype == "input_json_delta":
                state = self.tool_calls.get(("anthropic", idx))
                if state is not None:
                    arg_delta = delta.get("partial_json") or ""
                    if arg_delta:
                        state["arguments"] += arg_delta
                        await _write_sse(
                            response,
                            {
                                "type": "response.function_call_arguments.delta",
                                "item_id": state["id"],
                                "output_index": state["output_index"],
                                "delta": arg_delta,
                            },
                        )
            elif dtype == "thinking_delta":
                state = self.reasoning_blocks.get(("anthropic_thinking", idx))
                if state is None:
                    state = await self._open_reasoning(response, key=("anthropic_thinking", idx))
                txt = delta.get("thinking") or ""
                if txt:
                    state["text"] += txt
                    await self._emit_reasoning_summary_deltas(response, state, txt)
            elif dtype == "signature_delta":
                state = self.reasoning_blocks.get(("anthropic_thinking", idx))
                if state is None:
                    state = await self._open_reasoning(response, key=("anthropic_thinking", idx))
                state["signature"] += delta.get("signature") or ""
        elif event_type == "message_delta":
            usage = event.get("usage")
            if isinstance(usage, dict):
                if self.usage is None or any(
                    key in usage for key in ("input_tokens", "prompt_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")
                ):
                    normalized = normalize_responses_usage(usage)
                    if normalized is not None:
                        self.usage = normalized if self.usage is None else {**self.usage, **normalized}
                output_tokens = usage.get("output_tokens")
                if isinstance(output_tokens, int) and not isinstance(output_tokens, bool):
                    if self.usage is None:
                        self.usage = normalize_responses_usage(usage)
                    else:
                        self.usage["output_tokens"] = output_tokens
                        self.usage["total_tokens"] = int(self.usage.get("input_tokens") or 0) + output_tokens
        elif event_type == "content_block_stop":
            idx = int(event.get("index", 0))
            tool_state = self.tool_calls.get(("anthropic", idx))
            if tool_state is not None and not tool_state.get("closed"):
                await self._close_tool(response, tool_state)
            r_state = self.reasoning_blocks.get(("anthropic_thinking", idx))
            if r_state is not None and not r_state.get("closed"):
                await self._close_reasoning(response, r_state)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    async def _open_message(self, response: web.StreamResponse) -> None:
        self.message_index = self.next_output_index
        self.next_output_index += 1
        self.message_opened = True
        await _write_sse(
            response,
            {
                "type": "response.output_item.added",
                "output_index": self.message_index,
                "item": {
                    "id": self.message_item_id,
                    "type": "message",
                    "status": "in_progress",
                    "role": "assistant",
                    "content": [],
                },
            },
        )
        await _write_sse(
            response,
            {
                "type": "response.content_part.added",
                "item_id": self.message_item_id,
                "output_index": self.message_index,
                "content_index": 0,
                "part": {"type": "output_text", "text": "", "annotations": []},
            },
        )

    async def _close_message(self, response: web.StreamResponse) -> None:
        if not self.message_opened or self.message_closed:
            return
        self.message_closed = True
        await _write_sse(
            response,
            {
                "type": "response.output_text.done",
                "item_id": self.message_item_id,
                "output_index": self.message_index,
                "content_index": 0,
                "text": self.message_text,
            },
        )
        await _write_sse(
            response,
            {
                "type": "response.content_part.done",
                "item_id": self.message_item_id,
                "output_index": self.message_index,
                "content_index": 0,
                "part": {"type": "output_text", "text": self.message_text, "annotations": []},
            },
        )
        await _write_sse(
            response,
            {
                "type": "response.output_item.done",
                "output_index": self.message_index,
                "item": self._message_item("completed"),
            },
        )
        if self.message_index is not None:
            self.finished_messages.append((self.message_index, self._message_item("completed")))

    async def close_message_segment(self, response: web.StreamResponse) -> None:
        if self.message_opened and not self.message_closed and self.message_text:
            await self._close_message(response)

    async def reopen_message_segment(self, response: web.StreamResponse) -> None:
        self.message_item_id = f"msg_{int(time.time() * 1000)}_{self.next_output_index}"
        self.message_text = ""
        self.message_closed = False
        self.message_opened = False
        self.message_index = None
        await self._open_message(response)

    async def open_cursor_tool_activity(
        self,
        response: web.StreamResponse,
        call_id: str,
        markdown: str,
    ) -> None:
        key = ("cursor_tool", call_id)
        await self._open_reasoning(response, key=key, initial_text=markdown)

    async def append_cursor_tool_activity(
        self,
        response: web.StreamResponse,
        call_id: str,
        markdown: str,
    ) -> None:
        if not markdown:
            return
        key = ("cursor_tool", call_id)
        state = self.reasoning_blocks.get(key)
        if state is None:
            await self.open_cursor_tool_activity(response, call_id, markdown)
            return
        state["text"] = str(state.get("text") or "") + markdown
        await self._emit_reasoning_summary_deltas(response, state, markdown)

    async def close_cursor_tool_activity(
        self,
        response: web.StreamResponse,
        call_id: str,
    ) -> None:
        key = ("cursor_tool", call_id)
        state = self.reasoning_blocks.get(key)
        if state is not None and not state.get("closed"):
            await self._close_reasoning(response, state)

    async def append_cursor_thinking_activity(
        self,
        response: web.StreamResponse,
        delta: str,
    ) -> None:
        if not delta:
            return
        key = ("cursor_thinking",)
        state = self.reasoning_blocks.get(key)
        if state is None:
            await self._open_reasoning(
                response,
                key=key,
                initial_text=f"**cursor-agent · thinking**\n\n{delta}",
            )
            return
        if state.get("closed"):
            return
        state["text"] = str(state.get("text") or "") + delta
        await self._emit_reasoning_summary_deltas(response, state, delta)

    async def close_cursor_thinking_activity(self, response: web.StreamResponse) -> None:
        key = ("cursor_thinking",)
        state = self.reasoning_blocks.get(key)
        if state is not None and not state.get("closed"):
            await self._close_reasoning(response, state)

    async def interrupt_cursor_tool_activities(
        self,
        response: web.StreamResponse,
        message: str,
    ) -> None:
        for key, state in list(self.reasoning_blocks.items()):
            if not isinstance(key, tuple) or key[0] != "cursor_tool":
                continue
            if state.get("closed"):
                continue
            state["text"] = str(state.get("text") or "") + message
            await self._emit_reasoning_summary_deltas(response, state, message)
            await self._close_reasoning(response, state)

    async def _text_delta(self, response: web.StreamResponse, text: str) -> None:
        if not text:
            return
        if self.message_closed:
            await self.reopen_message_segment(response)
        elif not self.message_opened:
            await self._open_message(response)
        self.message_text += text
        await _write_sse(
            response,
            {
                "type": "response.output_text.delta",
                "item_id": self.message_item_id,
                "output_index": self.message_index,
                "content_index": 0,
                "delta": text,
            },
        )

    async def emit_synthetic_function_call(
        self,
        response: web.StreamResponse,
        *,
        name: str,
        arguments: dict[str, Any],
        call_id: str,
        namespace: str | None = None,
        chat_name: str | None = None,
    ) -> None:
        await self.close_cursor_thinking_activity(response)
        await self.close_message_segment(response)
        lookup_name = chat_name or name
        state = await self._open_tool(
            response,
            key=call_id,
            call_id=call_id,
            name=name,
            namespace=namespace,
        )
        state["output_type"] = _responses_output_type_for_tool(lookup_name, self.tool_types)
        state["arguments"] = json.dumps(arguments, separators=(",", ":"))
        await self._close_tool(response, state)

    async def _open_tool(
        self,
        response: web.StreamResponse,
        *,
        key: Any,
        call_id: str,
        name: str,
        namespace: str | None = None,
    ) -> dict[str, Any]:
        if self.message_opened and not self.message_closed:
            await self._close_message(response)
        output_index = self.next_output_index
        self.next_output_index += 1
        state: dict[str, Any] = {
            "id": call_id,
            "call_id": call_id,
            "name": name,
            "namespace": namespace,
            "arguments": "",
            "output_index": output_index,
            "output_type": _responses_output_type_for_tool(name, self.tool_types),
            "closed": False,
        }
        self.tool_calls[key] = state
        item = _stream_tool_added_item(state)
        await _write_sse(
            response,
            {
                "type": "response.output_item.added",
                "output_index": output_index,
                "item": item,
            },
        )
        return state

    async def _close_tool(self, response: web.StreamResponse, state: dict[str, Any]) -> None:
        if state.get("closed") or not _tool_call_arguments_complete(state):
            return
        state["closed"] = True
        if state.get("output_type", "function_call") == "function_call":
            await _write_sse(
                response,
                {
                    "type": "response.function_call_arguments.done",
                    "item_id": state["id"],
                    "output_index": state["output_index"],
                    "arguments": state["arguments"],
                },
            )
        await _write_sse(
            response,
            {
                "type": "response.output_item.done",
                "output_index": state["output_index"],
                "item": self._tool_item(state, "completed"),
            },
        )

    async def _open_reasoning(
        self,
        response: web.StreamResponse,
        *,
        key: Any,
        initial_text: str = "",
        initial_signature: str = "",
        redacted: bool = False,
        redacted_data: str = "",
    ) -> dict[str, Any]:
        # Reasoning items are emitted before the assistant message/tool calls
        # so we open them eagerly. If a message/tool was already opened we
        # still slot them in at the next available output_index; Codex orders
        # by output_index when reconciling.
        output_index = self.next_output_index
        self.next_output_index += 1
        item_id = f"rs_{int(time.time() * 1000)}_{output_index}"
        state: dict[str, Any] = {
            "id": item_id,
            "output_index": output_index,
            "text": initial_text,
            "signature": initial_signature,
            "redacted": redacted,
            "redacted_data": redacted_data,
            "closed": False,
        }
        self.reasoning_blocks[key] = state
        await _write_sse(
            response,
            {
                "type": "response.output_item.added",
                "output_index": output_index,
                "item": {
                    "id": item_id,
                    "type": "reasoning",
                    "status": "in_progress",
                    "summary": [],
                    "encrypted_content": None,
                },
            },
        )
        if initial_text:
            await self._emit_reasoning_summary_deltas(response, state, initial_text)
        return state

    async def _close_reasoning(self, response: web.StreamResponse, state: dict[str, Any]) -> None:
        state["closed"] = True
        # Emit summary_text.done so renderers can finalize the reasoning bubble.
        await _write_sse(
            response,
            {
                "type": "response.reasoning_summary_text.done",
                "item_id": state["id"],
                "output_index": state["output_index"],
                "summary_index": 0,
                "text": state["text"],
            },
        )
        await _write_sse(
            response,
            {
                "type": "response.output_item.done",
                "output_index": state["output_index"],
                "item": self._reasoning_item(state, "completed"),
            },
        )

    def _reasoning_item(self, state: dict[str, Any], status: str) -> dict[str, Any]:
        # Encode the original Anthropic thinking block in encrypted_content so
        # we can roundtrip it back on the next turn. Codex preserves this
        # field verbatim across turns.
        if state.get("redacted"):
            payload = {"type": "redacted_thinking", "data": state.get("redacted_data", "")}
        else:
            payload = {
                "type": "thinking",
                "thinking": state.get("text", ""),
                "signature": state.get("signature", ""),
            }
        encrypted = _encode_thinking_payload(payload)
        # Streamed deltas drive live UI; completed summary must be populated
        # because Desktop replaces the whole reasoning item on item/completed.
        return {
            "id": state["id"],
            "type": "reasoning",
            "status": status,
            "summary": (
                [{"type": "summary_text", "text": state.get("text", "")}]
                if state.get("text") and not state.get("redacted")
                else []
            ),
            "encrypted_content": encrypted,
        }

    def _message_item(self, status: str) -> dict[str, Any]:
        return {
            "id": self.message_item_id,
            "type": "message",
            "status": status,
            "role": "assistant",
            "content": [
                {"type": "output_text", "text": self.message_text, "annotations": []}
            ] if self.message_text else [],
        }

    def _mcp_tool_item(self, state: dict[str, Any], status: str = "completed") -> dict[str, Any]:
        return tool_translate.mcp_function_call_item(
            state["id"],
            state["server"],
            state["tool"],
            state.get("arguments") or "",
            status,
        )

    def _tool_search_item(self, state: dict[str, Any], status: str = "completed") -> dict[str, Any]:
        return tool_translate.tool_search_call_from_raw(
            state["id"],
            state.get("arguments") or "",
            status,
        )

    def _tool_item(self, state: dict[str, Any], status: str) -> dict[str, Any]:
        output_type = state.get("output_type", "function_call")
        if output_type in {"custom_tool_call", "web_search_call"}:
            return _stream_tool_added_item(state, status)
        item: dict[str, Any] = {
            "id": state["id"],
            "type": "function_call",
            "status": status,
            "call_id": state["call_id"],
            "name": state["name"],
            "arguments": state["arguments"],
        }
        if state.get("namespace"):
            item["namespace"] = state["namespace"]
        return item

    def _response(self, status: str, *, final: bool = False) -> dict[str, Any]:
        output: list[dict[str, Any]] = []
        if final:
            collected: list[tuple[int, dict[str, Any]]] = []
            for turn in [*self.completed_turns, self._current_turn_dict()]:
                for state in turn["reasoning_blocks"].values():
                    collected.append((state["output_index"], self._reasoning_item(state, "completed")))
                if (
                    turn["message_opened"]
                    and turn["message_text"]
                    and turn["message_index"] is not None
                    and not turn["message_closed"]
                ):
                    collected.append((turn["message_index"], self._message_item_for_turn(turn, "completed")))
                for state in turn["tool_calls"].values():
                    if mcp_search.parse_mcp_tool_reference(state.get("name") or ""):
                        continue
                    if mcp_search.is_tool_search_call(state.get("name") or ""):
                        continue
                    collected.append((state["output_index"], self._tool_item(state, "completed")))
                for state in turn.get("mcp_tool_calls", {}).values():
                    collected.append((state["output_index"], self._mcp_tool_item(state)))
                for state in turn.get("tool_search_calls", {}).values():
                    collected.append((state["output_index"], self._tool_search_item(state)))
            for output_index, item in self.finished_messages:
                collected.append((output_index, item))
            collected.sort(key=lambda pair: pair[0])
            output = [item for _, item in collected]
        payload = {
            "id": self.response_id,
            "object": "response",
            "created_at": int(time.time()),
            "status": status,
            "model": self.model,
            "output": output,
        }
        if self.usage is not None:
            payload["usage"] = self.usage
        elif final:
            payload["usage"] = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        return payload

    def _message_item_for_turn(self, turn: dict[str, Any], status: str) -> dict[str, Any]:
        return {
            "id": turn["message_item_id"],
            "type": "message",
            "status": status,
            "role": "assistant",
            "content": [
                {"type": "output_text", "text": turn["message_text"], "annotations": []}
            ] if turn["message_text"] else [],
        }


async def _apply_cursor_stream_event(
    state: ResponsesStreamState,
    response: web.StreamResponse,
    event: dict[str, Any],
) -> None:
    event_type = event.get("type")
    if event_type == "text_delta":
        await state.close_cursor_thinking_activity(response)
        await state.write_chat_delta(
            response,
            {"choices": [{"delta": {"content": event.get("delta") or ""}}]},
        )
        return
    if event_type == "segment_boundary":
        await state.close_cursor_thinking_activity(response)
        await state.close_message_segment(response)
        return
    if event_type == "tool_started":
        await state.close_cursor_thinking_activity(response)
        await state.close_message_segment(response)
        await state.open_cursor_tool_activity(
            response,
            str(event.get("call_id") or ""),
            str(event.get("markdown") or ""),
        )
        return
    if event_type == "tool_completed":
        call_id = str(event.get("call_id") or "")
        await state.append_cursor_tool_activity(
            response,
            call_id,
            str(event.get("markdown") or ""),
        )
        await state.close_cursor_tool_activity(response, call_id)
        return
    if event_type == "thinking_delta":
        await state.append_cursor_thinking_activity(response, str(event.get("delta") or ""))
        return
    if event_type == "thinking_completed":
        await state.close_cursor_thinking_activity(response)
        return
    if event_type == "connection_interrupted":
        await state.interrupt_cursor_tool_activities(
            response,
            str(event.get("message") or ""),
        )
        return
    if event_type == "completed":
        await state.close_message_segment(response)


_THINKING_MAGIC = "anthropic-thinking-v1:"


def _encode_thinking_payload(payload: dict[str, Any]) -> str:
    import base64

    raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    return _THINKING_MAGIC + base64.urlsafe_b64encode(raw).decode("ascii")


def _decode_thinking_payload(encoded: str) -> dict[str, Any] | None:
    import base64

    if not isinstance(encoded, str) or not encoded.startswith(_THINKING_MAGIC):
        return None
    blob = encoded[len(_THINKING_MAGIC) :]
    try:
        raw = base64.urlsafe_b64decode(blob.encode("ascii"))
        data = json.loads(raw.decode("utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


_VERSIONED_BASE_RE = re.compile(r"(?:^|/)v\d+$")


def _join_url(base_url: str, endpoint: str) -> str:
    base = base_url.rstrip("/")
    if _VERSIONED_BASE_RE.search(base):
        # Already ends with /v<n> (e.g. /v1, /api/coding/v3) — append
        # the endpoint as-is rather than injecting another /v1/.
        return base + endpoint
    if endpoint == "/messages":
        return base + "/v1/messages"
    return urljoin(base + "/", "v1" + endpoint)


def _openai_headers(
    request_headers: Mapping[str, str],
    route: ShimModel,
    *,
    accept: str | None = None,
) -> dict[str, str]:
    return openai_upstream_headers(
        request_headers,
        api_key=route.api_key or None,
        extra_headers=route.extra_headers,
        accept=accept,
    )


def _anthropic_headers(
    request_headers: Mapping[str, str],
    route: ShimModel,
    *,
    accept: str | None = None,
) -> dict[str, str]:
    return anthropic_upstream_headers(
        request_headers,
        api_key=route.api_key or None,
        extra_headers=route.extra_headers,
        accept=accept,
    )


def _anthropic_text(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""
    parts = [
        str(block.get("text") or "")
        for block in (payload.get("content") or [])
        if isinstance(block, dict) and block.get("type") == "text"
    ]
    return "".join(parts)


def _sse_response() -> web.StreamResponse:
    response = web.StreamResponse(
        status=200,
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
    return response


async def _safe_write(response: web.StreamResponse, data: bytes) -> None:
    try:
        await response.write(data)
    except (ConnectionResetError, ConnectionError):
        raise ClientDisconnected()
    except Exception as exc:
        if exc.__class__.__name__ in {
            "ClientConnectionResetError",
            "ClientConnectionError",
            "ClientPayloadError",
        }:
            raise ClientDisconnected() from exc
        raise


def _stream_log_enabled() -> bool:
    return os.environ.get("CODEX_SHIM_STREAM_LOG", "").lower() in {"1", "true", "yes", "on"}


def _log_stream_event(payload: dict[str, Any]) -> None:
    if not _stream_log_enabled():
        return
    event_type = payload.get("type", "?")
    detail = ""
    if event_type == "response.output_item.added":
        item = payload.get("item") or {}
        extra = item.get("name") or ""
        if not extra and item.get("type") == "function_call" and item.get("namespace"):
            extra = f"{item.get('namespace')}/{item.get('name')}"
        detail = f" item={item.get('type')} name={extra}"
    elif event_type == "response.output_item.done":
        item = payload.get("item") or {}
        extra = item.get("name") or ""
        if not extra and item.get("type") == "function_call" and item.get("namespace"):
            extra = f"{item.get('namespace')}/{item.get('name')}"
        elif not extra:
            extra = (item.get("action") or {}).get("query", "")
        detail = f" item={item.get('type')} name={extra}"
    elif event_type == "response.reasoning_summary_text.delta":
        detail = f" len={len(payload.get('delta') or '')}"
    print(f"[sse] {event_type}{detail}", flush=True)


async def _write_sse(response: web.StreamResponse, payload: dict[str, Any]) -> None:
    _log_stream_event(payload)
    try:
        await response.write(f"data: {json.dumps(payload, separators=(',', ':'))}\n\n".encode())
    except (ConnectionResetError, ConnectionError) as exc:
        raise ClientDisconnected() from exc
    except Exception as exc:
        # aiohttp raises ClientConnectionResetError (an OSError subclass on
        # some versions, a ClientConnectionError on others). Trap both.
        if exc.__class__.__name__ in {
            "ClientConnectionResetError",
            "ClientConnectionError",
            "ClientPayloadError",
        }:
            raise ClientDisconnected() from exc
        raise


async def _write_ws_json(ws: web.WebSocketResponse, payload: dict[str, Any]) -> None:
    _log_stream_event(payload)
    await ws.send_str(json.dumps(payload, separators=(",", ":")))


async def _write_compaction_v2_ws(
    ws: web.WebSocketResponse,
    response_slug: str,
    compaction_item: dict[str, Any],
    usage: dict[str, Any] | None,
) -> None:
    response_id = f"resp_compact_{int(time.time() * 1000)}"
    completed_response: dict[str, Any] = {
        "id": response_id,
        "object": "response",
        "status": "completed",
        "model": response_slug,
        "output": [compaction_item],
    }
    if usage is not None:
        completed_response["usage"] = usage
    await _write_ws_json(
        ws,
        {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": compaction_item,
        },
    )
    await _write_ws_json(
        ws,
        {
            "type": "response.completed",
            "response": completed_response,
        },
    )


async def _ws_error_from_http_response(ws: web.WebSocketResponse, response: web.StreamResponse | web.Response) -> None:
    text = ""
    if isinstance(response, web.Response):
        text = response.text or ""
    code, message = parse_upstream_error(text, response.status)
    await _write_ws_error(ws, response.status, code, message)


async def _stream_compaction_v2_sse(
    request: web.Request,
    response_slug: str,
    compaction_item: dict[str, Any],
    usage: dict[str, Any] | None,
) -> web.StreamResponse:
    response = _sse_response()
    await response.prepare(request)
    response_id = f"resp_compact_{int(time.time() * 1000)}"
    completed_response: dict[str, Any] = {
        "id": response_id,
        "object": "response",
        "status": "completed",
        "model": response_slug,
        "output": [compaction_item],
    }
    if usage is not None:
        completed_response["usage"] = usage
    try:
        await _write_sse(
            response,
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": compaction_item,
            },
        )
        await _write_sse(
            response,
            {
                "type": "response.completed",
                "response": completed_response,
            },
        )
        await response.write(b"data: [DONE]\n\n")
    except ClientDisconnected:
        pass
    try:
        await response.write_eof()
    except Exception:
        pass
    return response


async def _write_ws_error(ws: web.WebSocketResponse, status: int, code: str, message: str) -> None:
    await _write_ws_json(
        ws,
        {
            "type": "error",
            "status": status,
            "error": {
                "type": code,
                "code": code,
                "message": message,
            },
        },
    )


async def _relay_sse_response_to_ws(
    upstream,
    request: web.Request,
    ws: web.WebSocketResponse,
    *,
    response_model_override: str | None = None,
    collector: ChatgptPassthroughResponseCollector | None = None,
    cache_collected: Any = None,
    upstream_forward_headers: dict[str, str] | None = None,
) -> None:
    _ = upstream_forward_headers
    async for line in _sse_lines(upstream, request):
        if line == "[DONE]":
            break
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            await _write_ws_error(ws, 502, "upstream_protocol_error", "upstream emitted non-JSON SSE data")
            continue
        if collector is not None:
            collector.record(event)
            if cache_collected is not None and _should_cache_chatgpt_passthrough_event(event):
                cache_collected(collector.response_id, collector.conversation_items(), terminal=False)
        if response_model_override:
            _rewrite_response_model(event, response_model_override)
        if event.get("type") == "response.completed":
            response_obj = event.get("response")
            usage = response_obj.get("usage") if isinstance(response_obj, dict) else None
            observe_upstream_response(
                "chatgpt-passthrough-ws",
                upstream,
                usage=usage if isinstance(usage, dict) else None,
            )
        await _write_ws_json(ws, event)


async def _write_anthropic_sse(response: web.StreamResponse, event: str, payload: dict[str, Any]) -> None:
    data = json.dumps(payload, separators=(",", ":"))
    try:
        await response.write(f"event: {event}\ndata: {data}\n\n".encode())
    except (ConnectionResetError, ConnectionError) as exc:
        raise ClientDisconnected() from exc
    except Exception as exc:
        if exc.__class__.__name__ in {
            "ClientConnectionResetError",
            "ClientConnectionError",
            "ClientPayloadError",
        }:
            raise ClientDisconnected() from exc
        raise


class ClientDisconnected(Exception):
    """Raised when the downstream Codex client closes the SSE connection."""


def _passthrough_trace_enabled() -> bool:
    return os.environ.get("CODEX_SHIM_PASSTHROUGH_TRACE", "").lower() in {"1", "true", "yes", "on"}


def _shim_io_log_enabled() -> bool:
    return shim_io_log_enabled()


def _input_has_compaction_trigger(input_items: Any) -> bool:
    if not isinstance(input_items, list):
        return False
    return any(isinstance(item, dict) and item.get("type") == "compaction_trigger" for item in input_items)


def _summarize_input_items(input_items: Any, *, tail: int = 6) -> tuple[int, list[str]]:
    if not isinstance(input_items, list):
        return 0, []
    summary: list[str] = []
    for item in input_items[-tail:]:
        if not isinstance(item, dict):
            summary.append("?")
            continue
        item_type = str(item.get("type") or item.get("role") or "?")
        extra = ""
        if item_type == "function_call":
            extra = f"({item.get('name', '?')})"
        elif item_type == "function_call_output":
            extra = f"(call_id={str(item.get('call_id', ''))[:24]})"
        elif item_type == "web_search_call":
            action = item.get("action") or {}
            extra = f"(query={str(action.get('query', ''))[:40]})"
        elif item_type == "mcp_tool_call":
            extra = f"({item.get('server', '?')}/{item.get('tool', '?')})"
        elif item_type == "message":
            role = item.get("role")
            if role:
                extra = f"(role={role})"
        summary.append(f"{item_type}{extra}")
    return len(input_items), summary


def _log_client_request(endpoint: str, body: dict[str, Any], *, transport: str = "http") -> None:
    try:
        tools = body.get("tools") or []
        names = []
        for tool in tools[:80]:
            if isinstance(tool, dict):
                name = tool.get("name") or (tool.get("function") or {}).get("name") or tool.get("type")
                if name:
                    names.append(str(name))
        tail = 12 if _input_has_compaction_trigger(body.get("input")) else 6
        input_count, input_summary = _summarize_input_items(body.get("input"), tail=tail)
        print(
            f"[req] {endpoint} transport={transport} model={body.get('model')!r} stream={body.get('stream')!r} "
            f"previous_response_id={body.get('previous_response_id')!r} "
            f"tools={len(tools)} ({names[:8]}) "
            f"input={input_count} ({input_summary})",
            flush=True,
        )
    except Exception as exc:
        print(f"[req] failed to log: {exc}", flush=True)


def _log_client_response(
    surface: str,
    status: int,
    *,
    detail: str | None = None,
    code: str | None = None,
) -> None:
    parts = [f"[resp] {surface} status={status}"]
    if code:
        parts.append(f"code={code!r}")
    if detail:
        parts.append(f"detail={detail[:500]!r}")
    print(" ".join(parts), flush=True)


def _log_upstream_status(
    surface: str,
    url: str,
    status: int,
    *,
    message: str | None = None,
) -> None:
    line = f"[upstream] {surface} url={url} status={status}"
    if message:
        line += f" {message[:500]}"
    print(line, flush=True)


def _log_upstream_response_from_http(
    surface: str,
    route_slug: str,
    response: web.StreamResponse | web.Response | None,
) -> None:
    if response is None:
        _log_upstream_status(surface, route_slug, 0, message="no response")
        return
    text = response.text if isinstance(response, web.Response) else ""
    _, message = parse_upstream_error(text, response.status)
    _log_upstream_status(surface, route_slug, response.status, message=message)
    _log_upstream_io_detail(
        surface=surface,
        phase="response",
        url=route_slug,
        status=response.status,
        response_text=text,
    )


def _log_upstream_io_detail(
    *,
    surface: str,
    phase: str,
    url: str,
    forwarded: dict[str, Any] | None = None,
    status: int | None = None,
    response_text: str | None = None,
) -> None:
    if not _shim_io_log_enabled():
        return
    payload: dict[str, Any] = {"surface": surface, "phase": phase, "url": url}
    if forwarded is not None:
        input_items = forwarded.get("input")
        input_types = _summarize_compaction_input_items(input_items)
        payload.update(
            {
                "model": forwarded.get("model"),
                "stream": forwarded.get("stream"),
                "input_item_count": len(input_items) if isinstance(input_items, list) else 0,
                "input_item_types": input_types,
                "instructions_chars": len(str(forwarded.get("instructions") or "")),
                "max_output_tokens": forwarded.get("max_output_tokens"),
            }
        )
    if status is not None:
        payload["status"] = status
    if response_text is not None:
        payload["response_text"] = response_text[:2000]
    print(f"[io] {json.dumps(payload, separators=(',', ':'), default=str)}", flush=True)


def _log_chatgpt_passthrough_trace(
    request: web.Request,
    forwarded: dict[str, Any],
    upstream_headers: dict[str, str],
    *,
    phase: str,
) -> None:
    if not _passthrough_trace_enabled():
        return
    trace_headers = {}
    for key in request.headers:
        lowered = key.lower()
        if lowered in {
            "session_id",
            "chatgpt-account-id",
            "authorization",
            "openai-beta",
            "originator",
            "accept-encoding",
        } or lowered.startswith("x-codex-"):
            value = request.headers.get(key, "")
            if lowered == "authorization":
                value = "<redacted>"
            trace_headers[key] = value
    input_items = forwarded.get("input")
    input_count = len(input_items) if isinstance(input_items, list) else 1 if input_items else 0
    print(
        "[chatgpt-trace] "
        f"phase={phase} "
        f"previous_response_id={forwarded.get('previous_response_id')!r} "
        f"input_items={input_count} "
        f"client_headers={json.dumps(trace_headers, separators=(',', ':'))} "
        f"upstream_header_keys={sorted(upstream_headers)}",
        flush=True,
    )


def _summarize_compaction_input_items(input_items: Any) -> list[str]:
    if not isinstance(input_items, list):
        return []
    summary: list[str] = []
    for item in input_items:
        if not isinstance(item, dict):
            summary.append("?")
            continue
        item_type = str(item.get("type") or item.get("role") or "?")
        extra = ""
        if item_type == "function_call":
            extra = f" name={item.get('name', '?')!r}"
        elif item_type == "function_call_output":
            extra = f" call_id={str(item.get('call_id', ''))[:24]!r}"
        elif item_type == "message":
            role = item.get("role")
            if role:
                extra = f" role={role!r}"
        summary.append(f"{item_type}{extra}")
    return summary


def _log_compaction_sanitization_warnings(warnings: list[str]) -> None:
    for warning in warnings:
        if warning:
            print(f"[warn] compaction: {warning}", flush=True)


def _log_compaction_upstream_trace(
    *,
    phase: str,
    url: str,
    forwarded: dict[str, Any],
    status: int | None = None,
    response_text: str | None = None,
) -> None:
    _log_upstream_io_detail(
        surface="compaction",
        phase=phase,
        url=url,
        forwarded=forwarded,
        status=status,
        response_text=response_text,
    )
    if status is not None:
        _, message = parse_upstream_error(response_text or "", status)
        _log_upstream_status("compaction", url, status, message=message)


def _log_incoming_request(endpoint: str, body: dict[str, Any]) -> None:
    _log_client_request(endpoint, body)


def _request_disconnected(request: web.Request | None) -> bool:
    if request is None:
        return False
    transport = request.transport
    if transport is not None and transport.is_closing():
        return True
    protocol = getattr(request, "protocol", None)
    if protocol is not None:
        proto_transport = getattr(protocol, "transport", None)
        if proto_transport is not None and proto_transport.is_closing():
            return True
    return False


async def _close_upstream(upstream) -> None:
    if upstream is None:
        return
    try:
        upstream.close()
    except Exception:
        pass
    try:
        upstream.release()
    except Exception:
        pass


async def _iter_upstream_chunks(content, request: web.Request | None = None):
    """Read upstream bytes until EOF or client disconnect.

    With ``handler_cancellation=True`` on the aiohttp runner, client STOP
    cancels the handler task and ``readany()`` raises ``CancelledError``.
    """
    del request  # reserved for future transport-level hooks
    try:
        while True:
            chunk = await content.readany()
            if not chunk:
                break
            yield chunk
    except asyncio.CancelledError:
        raise


async def _sse_lines(upstream, request: web.Request | None = None) -> Any:
    buffer = b""
    content = upstream.content
    try:
        async for chunk in _iter_upstream_chunks(content, request):
            buffer += chunk
            while b"\n" in buffer:
                raw, buffer = buffer.split(b"\n", 1)
                line = raw.decode("utf-8", errors="replace").strip()
                if line.startswith("data:"):
                    yield line[5:].strip()
    except asyncio.CancelledError:
        await _close_upstream(upstream)
        raise
    tail = buffer.decode("utf-8", errors="replace").strip()
    if tail.startswith("data:"):
        yield tail[5:].strip()


def _anthropic_stream_to_chat_chunk(event: dict[str, Any], model: str) -> dict[str, Any]:
    content = ""
    if event.get("type") == "content_block_delta":
        delta = event.get("delta") or {}
        if delta.get("type") == "text_delta":
            content = delta.get("text", "")
    return {"object": "chat.completion.chunk", "model": model, "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}]}


def _compact_request_body(body: dict[str, Any], upstream_model: str) -> dict[str, Any]:
    instructions = body.get("instructions") or _default_compact_instructions()
    return {
        "model": upstream_model,
        "instructions": instructions,
        "input": body.get("input") or [],
        "max_output_tokens": body.get("max_output_tokens") or body.get("max_tokens") or 4096,
        "stream": False,
    }


def _chatgpt_compact_request_body(stripped_input: list[Any], upstream_model: str) -> dict[str, Any]:
    return {
        "model": upstream_model,
        "instructions": _default_compact_instructions(),
        "input": stripped_input,
    }


def _default_compact_instructions() -> str:
    return (
        "Compact the conversation into a concise state handoff for the next Codex turn. "
        "Preserve the active task, user requirements, important file paths, commands already run, "
        "tool results, decisions, blockers, and the latest state. Omit filler and repeated text."
    )


async def _as_compact_response(response: web.StreamResponse, model: str) -> web.Response:
    if not isinstance(response, web.Response) or response.status >= 400:
        return response
    try:
        payload = json.loads(response.text or "{}")
    except json.JSONDecodeError:
        return response
    output = payload.get("output") if isinstance(payload, dict) else None
    summary = compaction_summary_from_output(output)
    compacted = compact_response_payload(model, summary, payload.get("usage") if isinstance(payload, dict) else None)
    return web.json_response(compacted)


def _compaction_orchestrator_error_response(
    slug: str,
    exc: CompactionOrchestratorError,
) -> web.Response:
    status = 502
    code = "compaction_failed"
    if exc.error_response is not None:
        status = getattr(exc.error_response, "status", 502)
        text = getattr(exc.error_response, "text", "") or ""
        parsed_code, _ = parse_upstream_error(text, status)
        if parsed_code:
            code = parsed_code
    return web.json_response(
        _responses_error_payload(slug, code, str(exc)),
        status=status,
    )


def parse_upstream_error(body: str, http_status: int) -> tuple[str, str]:
    text = (body or "").strip()
    code = f"upstream_http_{http_status}"
    message = text or f"Upstream returned HTTP {http_status}"
    if not text:
        return code, message
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return code, text[:2000]
    if not isinstance(payload, dict):
        return code, text[:2000]

    err = payload.get("error")
    if isinstance(err, dict):
        nested_message = err.get("message")
        if isinstance(nested_message, str) and nested_message.strip():
            message = nested_message.strip()
        nested_code = err.get("type") or err.get("code")
        if isinstance(nested_code, str) and nested_code.strip():
            code = nested_code.strip()
    elif isinstance(err, str) and err.strip():
        message = err.strip()

    top_message = payload.get("message")
    if isinstance(top_message, str) and top_message.strip():
        message = top_message.strip()
    top_code = payload.get("code")
    if isinstance(top_code, str) and top_code.strip():
        code = top_code.strip()

    detail = payload.get("detail")
    if isinstance(detail, str) and detail.strip():
        message = detail.strip()
    elif isinstance(detail, list):
        parts = [str(item).strip() for item in detail if str(item).strip()]
        if parts:
            message = "; ".join(parts)

    return code, message


def _responses_error_payload(model: str, code: str, message: str) -> dict[str, Any]:
    return {
        "id": f"resp_{int(time.time() * 1000)}",
        "object": "response",
        "created_at": int(time.time()),
        "status": "failed",
        "model": model,
        "output": [],
        "error": {"code": code, "message": message},
    }


async def _stream_responses_upstream_error(
    request: web.Request,
    client_slug: str,
    upstream,
    *,
    slug: str | None = None,
) -> web.StreamResponse:
    text = await upstream.text()
    status = upstream.status
    upstream.release()
    return await _stream_responses_error_from_body(request, client_slug, status, text, slug=slug)


async def _stream_responses_error_from_http_response(
    request: web.Request,
    client_slug: str,
    error_response: web.Response,
    *,
    slug: str | None = None,
) -> web.StreamResponse:
    return await _stream_responses_error_from_body(
        request,
        client_slug,
        error_response.status,
        error_response.text or "",
        slug=slug,
    )


async def _stream_responses_error_from_body(
    request: web.Request,
    client_slug: str,
    status: int,
    text: str,
    *,
    slug: str | None = None,
) -> web.StreamResponse:
    code, message = parse_upstream_error(text, status)
    if slug:
        print(f"[err] upstream {slug} returned {status}: {message[:500]}", flush=True)
    response = _sse_response()
    await response.prepare(request)
    state = ResponsesStreamState(client_slug)
    await state.start(response)
    await state.fail(response, message, code=code)
    try:
        await response.write_eof()
    except Exception:
        pass
    return response


async def _error_response(
    upstream,
    *,
    slug: str | None = None,
    url: str | None = None,
    request_body: dict[str, Any] | None = None,
) -> web.Response:
    observe_upstream_response(f"upstream-error:{slug or 'unknown'}", upstream)
    text = await upstream.text()
    status = upstream.status
    content_type = upstream.content_type or "text/plain"
    code, message = parse_upstream_error(text, status)
    if slug:
        print(f"[err] upstream {slug} returned {status}: {message[:500]}", flush=True)
    log_upstream_response(
        slug or "upstream-error",
        url or slug or "unknown",
        status,
        text,
        request_body=request_body,
    )
    upstream_response_headers = upstream_headers_from_response(upstream)
    upstream.release()
    return _upstream_text_response(status, text, content_type=content_type, upstream_headers=upstream_response_headers)


def _upstream_text_response(
    status: int,
    text: str,
    *,
    content_type: str,
    upstream_headers: Mapping[str, str] | None = None,
    client_surface: str | None = None,
) -> web.Response:
    if status >= 400:
        code, message = parse_upstream_error(text, status)
        _log_client_response(client_surface or "downstream", status, detail=message, code=code)
    response = web.Response(status=status, text=text, content_type=content_type)
    if upstream_headers:
        apply_upstream_headers_to_response(response, upstream_headers)
    return response


async def _anthropic_error_response(upstream) -> web.Response:
    observe_upstream_response("upstream-error:anthropic", upstream)
    text = await upstream.text()
    message = text
    error_type = "api_error"
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict):
        err = payload.get("error")
        if isinstance(err, dict):
            message = str(err.get("message") or message)
            error_type = str(err.get("type") or error_type)
        elif payload.get("message"):
            message = str(payload["message"])
    status_type = {
        400: "invalid_request_error",
        401: "authentication_error",
        403: "permission_error",
        404: "not_found_error",
        413: "request_too_large",
        429: "rate_limit_error",
    }.get(upstream.status)
    if status_type:
        error_type = status_type
    body = {
        "type": "error",
        "error": {"type": error_type, "message": message},
    }
    request_id = upstream.headers.get("request-id") or upstream.headers.get("x-request-id")
    if request_id:
        body["request_id"] = request_id
    upstream_response_headers = upstream_headers_from_response(upstream)
    upstream.release()
    response = web.json_response(body, status=upstream.status)
    apply_upstream_headers_to_response(response, upstream_response_headers)
    return response


def _missing_api_key_message(route: ShimModel) -> str:
    env_name = route.raw.get("api_key_env") or route.raw.get("apiKeyEnv")
    if env_name:
        return f"Model {route.slug} has no API key. Set {env_name} or add api_key/apiKey for this model."
    return f"Model {route.slug} has no API key. Add api_key/apiKey or api_key_env/apiKeyEnv for this model."


def _normalize_roles(messages: list[dict]) -> list[dict]:
    result = []
    for message in messages:
        if isinstance(message, dict):
            message = dict(message)
            if message.get("role") == "developer":
                message["role"] = "system"
        result.append(message)
    return result


def _dump_debug_request(slug: str, url: str, body: dict[str, Any]) -> None:
    """Best-effort dump of the last forwarded request body for debugging.

    Writes ``.codex-shim/last_request.json`` next to the rest of the runtime
    state (catalog, pid, log). Failures are silently swallowed — this is a
    debug aid, not a code path the request should depend on.
    """
    try:
        dump_path = DEBUG_DIR / "last_request.json"
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"slug": slug, "url": url, "body": body}
        full = json.dumps(payload, indent=2, default=str)
        if len(full) > 2_000_000:
            messages = body.get("messages") or []
            summary = {
                "slug": slug,
                "url": url,
                "_truncated": True,
                "_full_size": len(full),
                "message_count": len(messages),
                "tool_count": len(body.get("tools") or []),
                "last_3_messages": messages[-3:],
            }
            dump_path.write_text(json.dumps(summary, indent=2, default=str))
        else:
            dump_path.write_text(full)
    except OSError as exc:
        print(f"[debug] dump failed: {exc}", flush=True)


def _current_managed_model() -> str | None:
    """Return the first ``model = "..."`` value from ~/.codex/config.toml."""
    if not CODEX_CONFIG_PATH.exists():
        return None
    try:
        text = CODEX_CONFIG_PATH.read_text()
    except OSError:
        return None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("model = "):
            return stripped.split("=", 1)[1].strip().strip('"')
    return None


_MODEL_LINE_RE = re.compile(r'(?m)^(\s*model\s*=\s*")[^"]*(")')


def _set_active_model(slug: str, display_name: str | None = None) -> None:
    """Rewrite the active model in ~/.codex/config.toml."""
    if not CODEX_CONFIG_PATH.exists():
        return
    try:
        text = CODEX_CONFIG_PATH.read_text()
    except OSError:
        return
    text = _MODEL_LINE_RE.sub(rf'\g<1>{slug}\g<2>', text, count=1)
    try:
        CODEX_CONFIG_PATH.write_text(text)
    except OSError as exc:
        print(f"[switch] failed to write {CODEX_CONFIG_PATH}: {exc}", flush=True)
        return
    label = display_name or slug
    print(f"[switch] set active model to {slug} ({label})", flush=True)


def _restart_codex_app() -> None:
    """Quit and relaunch Codex Desktop in a background thread (non-blocking).

    Cross-platform: ``taskkill`` + ``Codex.exe`` on Windows, ``osascript`` +
    ``open -a Codex`` on macOS. Linux has no Codex Desktop build today, so
    the branch is a no-op there.
    """
    import os as _os
    import subprocess as _subprocess
    import threading as _threading
    import time as _time

    def _do_restart() -> None:
        try:
            if _os.name == "nt":
                _subprocess.run(
                    ["taskkill", "/IM", "Codex.exe", "/F"],
                    check=False,
                    stdout=_subprocess.DEVNULL,
                    stderr=_subprocess.DEVNULL,
                )
                _time.sleep(1.5)
                local_appdata = _os.environ.get("LOCALAPPDATA", "")
                codex_exe = Path(local_appdata) / "Programs" / "Codex" / "Codex.exe"
                if codex_exe.exists():
                    _subprocess.Popen([str(codex_exe)])
                else:
                    _subprocess.Popen(["Codex.exe"], shell=True)
            elif sys.platform == "darwin":
                quit_script = 'tell application "Codex" to if it is running then quit'
                _subprocess.run(
                    ["osascript", "-e", quit_script],
                    check=False,
                    stdout=_subprocess.DEVNULL,
                    stderr=_subprocess.DEVNULL,
                )
                _time.sleep(1.5)
                _subprocess.Popen(["open", "-a", "Codex"])
        except OSError:
            pass

    _threading.Thread(target=_do_restart, daemon=True).start()


def _picker_html(picker_token: str) -> str:
    token_json = json.dumps(picker_token).replace("<", "\\u003c")
    html = '''<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Codex Shim - Model Picker</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: #0d1117; color: #c9d1d9;
    display: flex; justify-content: center; align-items: center;
    min-height: 100vh; padding: 20px;
  }
  .container { max-width: 500px; width: 100%; }
  h1 { font-size: 24px; margin-bottom: 8px; color: #f0f6fc; }
  .subtitle { color: #8b949e; margin-bottom: 24px; font-size: 14px; }
  .model-card {
    background: #161b22; border: 1px solid #30363d; border-radius: 8px;
    padding: 16px; margin-bottom: 12px; cursor: pointer;
    transition: all 0.15s ease; display: flex; align-items: center;
    justify-content: space-between;
  }
  .model-card:hover { border-color: #58a6ff; background: #1c2333; }
  .model-card.active { border-color: #3fb950; background: #1a2e1a; }
  .model-info { flex: 1; }
  .model-name { font-size: 16px; font-weight: 600; color: #f0f6fc; }
  .model-provider { font-size: 12px; color: #8b949e; margin-top: 4px; }
  .model-badge {
    font-size: 11px; padding: 2px 8px; border-radius: 12px;
    font-weight: 600; text-transform: uppercase;
  }
  .badge-active { background: #1a4d2e; color: #3fb950; }
  .badge-switch { background: #1c2333; color: #58a6ff; }
  .status { text-align: center; margin-top: 16px; font-size: 14px; min-height: 20px; }
  .status.ok { color: #3fb950; }
  .status.err { color: #f85149; }
  .status.loading { color: #d29922; }
  .restart-note { color: #8b949e; font-size: 12px; text-align: center; margin-top: 8px; }
  .opt { display: flex; align-items: center; justify-content: center; gap: 8px; margin-top: 12px; }
  .opt label { font-size: 13px; color: #8b949e; cursor: pointer; }
  .opt input { cursor: pointer; }
</style>
</head>
<body>
<div class="container">
  <h1>Model Picker</h1>
  <p class="subtitle">Choose the active model for Codex Desktop</p>
  <div id="models"><div class="status loading">Loading models...</div></div>
  <div class="opt">
    <input type="checkbox" id="autoRestart" checked>
    <label for="autoRestart">Auto-restart Codex after switching</label>
  </div>
  <div id="status" class="status"></div>
  <p class="restart-note">Codex needs to restart to use the new model</p>
</div>
<script>
const PICKER_TOKEN = @@TOKEN_JSON@@;
async function loadModels() {
  const res = await fetch('/api/models');
  const models = await res.json();
  const container = document.getElementById('models');
  container.innerHTML = '';
  models.forEach(m => {
    const card = document.createElement('div');
    card.className = 'model-card' + (m.active ? ' active' : '');
    const info = document.createElement('div');
    info.className = 'model-info';
    const name = document.createElement('div');
    name.className = 'model-name';
    name.textContent = m.display_name;
    const prov = document.createElement('div');
    prov.className = 'model-provider';
    prov.textContent = m.provider + ' \u00b7 ' + m.slug;
    info.appendChild(name);
    info.appendChild(prov);
    const badge = document.createElement('span');
    badge.className = 'model-badge ' + (m.active ? 'badge-active' : 'badge-switch');
    badge.textContent = m.active ? 'Active' : 'Switch';
    card.appendChild(info);
    card.appendChild(badge);
    if (!m.active) {
      card.onclick = () => switchModel(m.slug);
    }
    container.appendChild(card);
  });
}
async function switchModel(slug) {
  const status = document.getElementById('status');
  const restart = document.getElementById('autoRestart').checked;
  status.className = 'status loading';
  status.textContent = 'Switching to ' + slug + '...';
  try {
    const res = await fetch('/api/switch', {
      method: 'POST',
      headers: {'Content-Type': 'application/json', '@@PICKER_HEADER@@': PICKER_TOKEN},
      body: JSON.stringify({slug, restart_codex: restart})
    });
    const data = await res.json();
    if (data.ok) {
      status.className = 'status ok';
      status.textContent = 'Switched to ' + slug + (restart ? ' \u2014 Codex restarting...' : '');
      setTimeout(loadModels, 1000);
    } else {
      status.className = 'status err';
      status.textContent = data.error || 'Failed';
    }
  } catch(e) {
    status.className = 'status err';
    status.textContent = 'Error: ' + e.message;
  }
}
loadModels();
</script>
</body>
</html>'''
    return (
        html.replace("@@TOKEN_JSON@@", token_json, 1)
        .replace("@@PICKER_HEADER@@", PICKER_TOKEN_HEADER, 1)
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings", type=Path, default=DEFAULT_SETTINGS)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args(argv)

    shim = ShimServer(args.settings, host=args.host)
    web.run_app(
        shim.app(),
        host=args.host,
        port=args.port,
        handle_signals=True,
        handler_cancellation=True,
        shutdown_timeout=5.0,
    )


if __name__ == "__main__":
    main()

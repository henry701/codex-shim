from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

from aiohttp import ClientSession, ClientTimeout, web

from .cursor_passthrough import (
    CURSOR_MODEL_SLUG,
    build_cursor_prompt,
    cursor_passthrough_available,
    cursor_passthrough_display_names,
    cursor_upstream_model,
    is_cursor_passthrough_slug,
    iter_cursor_agent_events,
)
from . import router as router_module
from . import mcp_search
from .hostguard import build_allowed_hosts, host_guard_middleware
from .settings import (
    CHATGPT_MODEL_SLUG,
    DEFAULT_CODEX_AUTH,
    DEFAULT_SETTINGS,
    DEFAULT_HOST,
    DEFAULT_PORT,
    PROVIDER_NAME,
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
    anthropic_to_chat_response,
    anthropic_to_response,
    chat_completion_to_response,
    chat_to_anthropic,
    normalize_responses_usage,
    responses_to_anthropic,
    responses_to_chat,
)

DEBUG_DIR = Path(__file__).resolve().parents[1] / ".codex-shim"
CODEX_CONFIG_PATH = Path.home() / ".codex" / "config.toml"


class ShimServer:
    def __init__(self, settings_path: Path = DEFAULT_SETTINGS, host: str = DEFAULT_HOST):
        self.settings = ModelSettings(settings_path)
        self.host = host
        self.timeout = ClientTimeout(total=None, sock_connect=120, sock_read=None)

    def app(self) -> web.Application:
        allowed_hosts = build_allowed_hosts(self.host)
        app = web.Application(
            client_max_size=64 * 1024 * 1024,
            middlewares=[host_guard_middleware(allowed_hosts)],
        )
        app.router.add_get("/health", self.health)
        app.router.add_get("/v1/models", self.models)
        app.router.add_post("/v1/chat/completions", self.chat_completions)
        app.router.add_post("/v1/responses", self.responses)
        app.router.add_post("/v1/responses/compact", self.responses_compact)
        app.router.add_get("/picker", self.picker_page)
        app.router.add_get("/api/models", self.api_models)
        app.router.add_post("/api/switch", self.switch_model)
        return app

    async def picker_page(self, _request: web.Request) -> web.Response:
        return web.Response(text=_picker_html(), content_type="text/html")

    async def api_models(self, _request: web.Request) -> web.Response:
        current = _current_managed_model()
        data: list[dict[str, Any]] = []
        router_config = self._active_router()
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
        for m in usable_byok_models(self.settings.load()):
            data.append(
                {
                    "slug": m.slug,
                    "display_name": m.display_name,
                    "provider": m.provider,
                    "active": current == m.slug,
                }
            )
        return web.json_response(data)

    async def switch_model(self, request: web.Request) -> web.Response:
        try:
            body = await request.json()
        except json.JSONDecodeError:
            return web.json_response({"error": "invalid JSON body"}, status=400)
        slug = str(body.get("slug") or "").strip()
        if not slug:
            return web.json_response({"error": "slug is required"}, status=400)
        models = usable_byok_models(self.settings.load())
        valid = {m.slug for m in models}
        display_for: dict[str, str] = {m.slug: m.display_name for m in models}
        router_config = self._active_router()
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

    async def health(self, _request: web.Request) -> web.Response:
        models = usable_byok_models(self.settings.load())
        chatgpt_ok = chatgpt_passthrough_available()
        cursor_ok = cursor_passthrough_available()
        passthrough_count = len(chatgpt_passthrough_slugs()) if chatgpt_ok else 0
        if cursor_ok:
            passthrough_count += len(cursor_passthrough_display_names())
        count = len(models) + passthrough_count
        return web.json_response(
            {
                "ok": True,
                "models": count,
                "chatgpt_passthrough": chatgpt_ok,
                "cursor_passthrough": cursor_ok,
                "auto_router": self._active_router() is not None,
            }
        )

    async def models(self, _request: web.Request) -> web.Response:
        now = int(time.time())
        data: list[dict[str, Any]] = []
        router_config = self._active_router()
        if router_config is not None:
            data.append(router_module.router_models_entry(router_config, now))
        if chatgpt_passthrough_available():
            data.extend(
                {"id": slug, "object": "model", "created": now, "owned_by": "chatgpt"}
                for slug in sorted(chatgpt_passthrough_slugs())
            )
        if cursor_passthrough_available():
            data.extend(
                {
                    "id": slug,
                    "object": "model",
                    "created": now,
                    "owned_by": "cursor",
                }
                for slug in sorted(cursor_passthrough_display_names())
            )
        data.extend({"id": model.slug, "object": "model", "created": now, "owned_by": "codex-shim"} for model in usable_byok_models(self.settings.load()))
        return web.json_response({"object": "list", "data": data})

    async def chat_completions(self, request: web.Request) -> web.StreamResponse:
        body = await request.json()
        body = await self._maybe_apply_auto_router(body)
        route = self._route(body)
        if route.is_openai_chat:
            forwarded = dict(body)
            forwarded["model"] = route.model
            if "messages" in forwarded:
                forwarded["messages"] = _normalize_roles(forwarded["messages"])
            discovered = await _pre_discover_if_mcp(forwarded)
            _inject_tool_search_if_mcp(forwarded, discovered)
            return await self._post_openai_chat(request, route, forwarded, as_responses=False)
        if route.is_anthropic:
            forwarded = chat_to_anthropic(body, route.model, route.max_output_tokens)
            return await self._post_anthropic(request, route, forwarded, as_responses=False)
        raise web.HTTPBadGateway(text=f"Unsupported model provider: {route.provider}")

    async def responses(self, request: web.Request) -> web.StreamResponse:
        body = await request.json()
        _log_incoming_request("/v1/responses", body)
        body = await self._maybe_apply_auto_router(body)
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
        route = self._route(body)
        if route.is_openai_chat:
            discovered = await _pre_discover_if_mcp(body)
            forwarded = responses_to_chat(body, route.model, discovered_mcp_tools=discovered)
            return await self._post_openai_chat(request, route, forwarded, as_responses=True)
        if route.is_anthropic:
            forwarded = responses_to_anthropic(body, route.model, route.max_output_tokens)
            return await self._post_anthropic(request, route, forwarded, as_responses=True)
        raise web.HTTPBadGateway(text=f"Unsupported model provider: {route.provider}")

    async def responses_compact(self, request: web.Request) -> web.StreamResponse:
        body = await request.json()
        _log_incoming_request("/v1/responses/compact", body)
        body = await self._maybe_apply_auto_router(body)
        model = str(body.get("model") or "")
        if is_chatgpt_passthrough_slug(model):
            upstream = chatgpt_upstream_model(model)
            return await self._chatgpt_compact_passthrough(request, body, upstream_model=upstream)
        if is_cursor_passthrough_slug(model):
            compact_body = dict(body)
            compact_body["input"] = body.get("input") or []
            compact_body["instructions"] = (
                f"{body.get('instructions') or ''}\n\nSummarize the conversation above into a compact "
                "context window suitable for continuing the task."
            ).strip()
            return await self._cursor_passthrough(
                request,
                compact_body,
                response_model_override=model,
                upstream_model=cursor_upstream_model(model),
                force_non_stream=True,
            )
        route = self._route(body)
        compact_body = _compact_request_body(body, route.model)
        if route.is_openai_chat:
            discovered = await _pre_discover_if_mcp(body)
            forwarded = responses_to_chat(compact_body, route.model, discovered_mcp_tools=discovered)
            forwarded["stream"] = False
            response = await self._post_openai_chat(request, route, forwarded, as_responses=True)
            return await _as_compact_response(response, route.slug)
        if route.is_anthropic:
            forwarded = responses_to_anthropic(compact_body, route.model, route.max_output_tokens)
            forwarded["stream"] = False
            response = await self._post_openai_chat(request, route, forwarded, as_responses=True)
            return await _as_compact_response(response, route.slug)
        raise web.HTTPBadGateway(text=f"Unsupported model provider: {route.provider}")

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

    async def _chatgpt_passthrough(
        self,
        request: web.Request,
        body: dict[str, Any],
        response_model_override: str | None = None,
        upstream_model: str | None = None,
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
        forwarded = _sanitize_chatgpt_passthrough_body(body)
        forwarded["model"] = upstream_model or CHATGPT_MODEL_SLUG
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream" if forwarded.get("stream") else "application/json",
            "OpenAI-Beta": "responses=2026-02-06",
            "originator": "codex_cli_rs",
            "chatgpt-account-id": account_id,
            "session_id": request.headers.get("session_id", ""),
        }
        url = "https://chatgpt.com/backend-api/codex/responses"
        async with ClientSession(timeout=self.timeout) as session:
            upstream = await session.post(url, json=forwarded, headers=headers)
            if upstream.status >= 400:
                text = await upstream.text()
                status = upstream.status
                content_type = upstream.content_type or "text/plain"
                upstream.release()
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
                return web.Response(status=status, text=text, content_type=content_type)
            if not forwarded.get("stream"):
                payload = await upstream.json(content_type=None)
                _rewrite_response_model(payload, response_model_override)
                return web.json_response(payload)
            response = _sse_response()
            await response.prepare(request)
            try:
                if response_model_override:
                    async for line in _sse_lines(upstream):
                        if line == "[DONE]":
                            await _safe_write(response, b"data: [DONE]\n\n")
                            break
                        try:
                            payload = json.loads(line)
                        except json.JSONDecodeError:
                            await _safe_write(response, f"data: {line}\n\n".encode())
                            continue
                        _rewrite_response_model(payload, response_model_override)
                        await _write_sse(response, payload)
                else:
                    async for chunk in upstream.content.iter_chunked(4096):
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
        forwarded = _sanitize_chatgpt_passthrough_body(body)
        original_model = str(forwarded.get("model") or "")
        forwarded["model"] = upstream_model or CHATGPT_MODEL_SLUG
        forwarded.pop("stream", None)
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "OpenAI-Beta": "responses=2026-02-06",
            "originator": "codex_cli_rs",
            "chatgpt-account-id": account_id,
            "session_id": request.headers.get("session_id", ""),
        }
        url = "https://chatgpt.com/backend-api/codex/responses/compact"
        async with ClientSession(timeout=self.timeout) as session:
            upstream = await session.post(url, json=forwarded, headers=headers)
            if upstream.status >= 400:
                text = await upstream.text()
                status = upstream.status
                content_type = upstream.content_type or "text/plain"
                upstream.release()
                fallback = await self._maybe_passthrough_byok_fallback(
                    request,
                    body,
                    requested=requested,
                    response_slug=original_model or requested,
                    status=status,
                    detail=text,
                    compact=True,
                )
                if fallback is not None:
                    return fallback
                return web.Response(status=status, text=text, content_type=content_type)
            payload = await upstream.json(content_type=None)
        _rewrite_response_model(payload, original_model or None)
        return web.json_response(payload)

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
        stream = bool(body.get("stream")) and not force_non_stream

        if not stream:
            text = ""
            usage: dict[str, Any] | None = None
            async for event in iter_cursor_agent_events(prompt, upstream):
                if event["type"] == "completed":
                    text = str(event.get("text") or text)
                elif event["type"] == "usage":
                    usage = event.get("usage") if isinstance(event.get("usage"), dict) else None
                elif event["type"] == "error":
                    raise web.HTTPBadGateway(text=str(event.get("message") or "cursor-agent failed"))
            payload: dict[str, Any] = {
                "id": f"resp_{int(time.time() * 1000)}",
                "object": "response",
                "model": slug,
                "status": "completed",
                "output": [
                    {
                        "id": "msg_0",
                        "type": "message",
                        "status": "completed",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": text, "annotations": []}],
                    }
                ],
            }
            normalized_usage = normalize_responses_usage(usage)
            if normalized_usage:
                payload["usage"] = normalized_usage
            return web.json_response(payload)

        response = _sse_response()
        await response.prepare(request)
        state = ResponsesStreamState(slug)
        try:
            await state.start(response)
            async for event in iter_cursor_agent_events(prompt, upstream):
                if event["type"] == "text_delta":
                    await state.write_chat_delta(
                        response,
                        {"choices": [{"delta": {"content": event["delta"]}}]},
                    )
                elif event["type"] == "usage":
                    normalized_usage = normalize_responses_usage(event.get("usage"))
                    if normalized_usage:
                        state.usage = normalized_usage
                elif event["type"] == "error":
                    message = str(event.get("message") or "cursor-agent failed")
                    await state.write_chat_delta(
                        response,
                        {"choices": [{"delta": {"content": message}}]},
                    )
                    break
            await state.finish(response)
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

    # ------------------------------------------------------------------
    # Auto Router
    # ------------------------------------------------------------------
    def _active_router(self):
        """Return the RouterConfig only when enabled and at least one candidate
        backend is usable, so discovery never advertises a dead Auto entry."""
        config = self.settings.load_router()
        if config and router_module.router_is_active(config, available_model_slugs(self.settings.load())):
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
        models = self.settings.load()
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
        return resolved or router_module.fallback_slug(config, candidates)

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
                    upstream = await session.post(url, json=payload, headers=_anthropic_headers(model))
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
                upstream = await session.post(url, json=payload, headers=_openai_headers(model))
                upstream.raise_for_status()
                data = await upstream.json(content_type=None)
                message = (data.get("choices") or [{}])[0].get("message") or {}
                return str(message.get("content") or "")

        return classify

    def _route(self, body: dict[str, Any]) -> ShimModel:
        requested = str(body.get("model") or "")
        route = self.settings.by_slug_or_model(requested)
        if route is None:
            raise web.HTTPNotFound(text=f"Unknown model slug/model: {requested}")
        if not byok_model_has_credentials(route):
            raise web.HTTPUnauthorized(
                text=(
                    f"Model {route.slug} has no API key. "
                    "Set CURSOR_API_KEY or create ~/.codex-shim/cursor-api-key."
                )
            )
        return route

    def _passthrough_fallback_slug(self, requested: str) -> str | None:
        return self.settings.passthrough_error_fallback().get(requested)

    async def _dispatch_byok_responses(
        self,
        request: web.Request,
        body: dict[str, Any],
        *,
        response_slug: str | None = None,
    ) -> web.StreamResponse:
        route = self._route(body)
        client_slug = response_slug or route.slug
        if route.is_openai_chat:
            discovered = await _pre_discover_if_mcp(body)
            forwarded = responses_to_chat(body, route.model, discovered_mcp_tools=discovered)
            return await self._post_openai_chat(
                request, route, forwarded, as_responses=True, response_slug=client_slug
            )
        if route.is_anthropic:
            forwarded = responses_to_anthropic(body, route.model, route.max_output_tokens)
            return await self._post_anthropic(
                request, route, forwarded, as_responses=True, response_slug=client_slug
            )
        raise web.HTTPBadGateway(text=f"Unsupported model provider: {route.provider}")

    async def _dispatch_byok_compact_responses(
        self,
        request: web.Request,
        body: dict[str, Any],
        *,
        response_slug: str | None = None,
    ) -> web.StreamResponse:
        route = self._route(body)
        client_slug = response_slug or route.slug
        compact_body = _compact_request_body(body, route.model)
        if route.is_openai_chat:
            discovered = await _pre_discover_if_mcp(body)
            forwarded = responses_to_chat(compact_body, route.model, discovered_mcp_tools=discovered)
            forwarded["stream"] = False
            response = await self._post_openai_chat(
                request, route, forwarded, as_responses=True, response_slug=client_slug
            )
            return await _as_compact_response(response, client_slug)
        if route.is_anthropic:
            forwarded = responses_to_anthropic(compact_body, route.model, route.max_output_tokens)
            forwarded["stream"] = False
            response = await self._post_anthropic(
                request, route, forwarded, as_responses=True, response_slug=client_slug
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

    async def _post_openai_chat(
        self, request: web.Request, route: ShimModel, body: dict[str, Any], as_responses: bool,
        *, response_slug: str | None = None,
    ) -> web.StreamResponse:
        client_slug = response_slug or route.slug
        url = _join_url(route.base_url, "/chat/completions")
        _dump_debug_request(route.slug, url, body)
        if body.get("stream"):
            return await self._stream_chat_loop(
                request, route, body, as_responses, response_slug=client_slug
            )
        async with ClientSession(timeout=self.timeout) as session:
            upstream = await session.post(url, json=body, headers=_openai_headers(route))
            if upstream.status >= 400:
                return await _error_response(upstream, slug=route.slug)
            payload = await upstream.json(content_type=None)
        if as_responses:
            response_payload = chat_completion_to_response(payload, client_slug)
            response_payload = await mcp_search.augment_response_with_tool_search(response_payload)
            return web.json_response(response_payload)
        return web.json_response(payload)

    async def _stream_chat_loop(
        self,
        request: web.Request,
        route: ShimModel,
        body: dict[str, Any],
        as_responses: bool,
        *,
        response_slug: str | None = None,
        max_turns: int = 6,
    ) -> web.StreamResponse:
        client_slug = response_slug or route.slug
        response = _sse_response()
        await response.prepare(request)
        state = ResponsesStreamState(client_slug) if as_responses else None
        if as_responses and state is not None:
            await state.start(response)
        messages = list(body.get("messages", []))
        chat_body = {k: v for k, v in body.items() if k != "stream"}
        chat_body["stream"] = True
        url = _join_url(route.base_url, "/chat/completions")
        headers = _openai_headers(route)
        try:
            async with ClientSession(timeout=self.timeout) as session:
                for _ in range(max_turns):
                    turn_body = {**chat_body, "messages": messages}
                    upstream = await session.post(url, json=turn_body, headers=headers)
                    if upstream.status >= 400:
                        text = await upstream.text()
                        await _safe_write(
                            response,
                            json.dumps({"error": f"upstream {upstream.status}: {text[:200]}"}).encode() + b"\n",
                        )
                        upstream.release()
                        break
                    async for line in _sse_lines(upstream):
                        if line == "[DONE]":
                            break
                        try:
                            event = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if as_responses and state is not None:
                            await state.write_chat_delta(
                                response, event, hide_tool_calls=True
                            )
                        else:
                            await _write_sse(response, event)
                    upstream.release()
                    mcp_calls = self._collect_state_mcp_calls(state)
                    if not mcp_calls:
                        break
                    if as_responses and state is not None:
                        await state.close_turn_items(response)
                        state.snapshot_turn()
                    messages = await self._build_followup_messages_from_state(
                        messages, state, mcp_calls
                    )
                    if as_responses and state is not None:
                        state.reset_for_next_turn()
            if as_responses and state is not None:
                await state.finish(response)
            else:
                await _safe_write(response, b"data: [DONE]\n\n")
        except ClientDisconnected:
            pass
        try:
            await response.write_eof()
        except Exception:
            pass
        return response

    @staticmethod
    def _collect_state_mcp_calls(state: ResponsesStreamState | None) -> list[tuple[Any, dict[str, Any]]]:
        if state is None:
            return []
        out: list[tuple[Any, dict[str, Any]]] = []
        for key, tc in state.all_tool_calls().items():
            name = tc.get("name") or ""
            server = mcp_search.is_mcp_tool_call(name)
            if server:
                out.append((key, tc))
        return out

    @classmethod
    async def _build_followup_messages_from_state(
        cls,
        messages: list[dict[str, Any]],
        state: ResponsesStreamState | None,
        mcp_calls: list[tuple[Any, dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        all_calls: list[dict[str, Any]] = []
        if state is not None:
            for tc in state.all_tool_calls().values():
                all_calls.append(
                    {
                        "id": tc.get("call_id"),
                        "type": "function",
                        "function": {
                            "name": tc.get("name"),
                            "arguments": tc.get("arguments"),
                        },
                    }
                )
        mcp_ids = {tc.get("call_id") for _, tc in mcp_calls}
        assistant_content = state.message_text if state is not None else None
        assistant_msg: dict[str, Any] = {
            "role": "assistant",
            "content": assistant_content or None,
            "tool_calls": all_calls,
        }
        new_messages = list(messages) + [assistant_msg]
        for _, tc in mcp_calls:
            name = tc.get("name") or ""
            server = mcp_search.is_mcp_tool_call(name) or ""
            result = await cls._dispatch_state_mcp_call(tc, server)
            new_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.get("call_id"),
                    "content": result,
                }
            )
        for call in all_calls:
            if call.get("id") in mcp_ids:
                continue
            name = (call.get("function") or {}).get("name") or ""
            new_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.get("id"),
                    "content": json.dumps({"error": f"unsupported call: {name}"}),
                }
            )
        return new_messages

    @staticmethod
    async def _dispatch_state_mcp_call(tc: dict[str, Any], server: str) -> str:
        name = tc.get("name") or ""
        args_str = tc.get("arguments") or "{}"
        try:
            args = json.loads(args_str) if isinstance(args_str, str) and args_str.strip() else {}
            if not isinstance(args, dict):
                args = {"_value": args}
        except json.JSONDecodeError:
            args = {"_raw": args_str}
        if not server:
            return json.dumps({"error": f"Could not parse MCP server from tool name '{name}'"})
        tool = name[len(server) + 2:]
        url = mcp_search.resolve_mcp_url(server)
        if not url:
            return json.dumps({"error": f"Unknown MCP server '{server}'"})
        return await mcp_search.call_mcp_tool(url, tool, args)

    async def _post_anthropic(
        self, request: web.Request, route: ShimModel, body: dict[str, Any], as_responses: bool,
        *, response_slug: str | None = None,
    ) -> web.StreamResponse:
        client_slug = response_slug or route.slug
        url = _join_url(route.base_url, "/messages")
        headers = _anthropic_headers(route)
        async with ClientSession(timeout=self.timeout) as session:
            upstream = await session.post(url, json=body, headers=headers)
            if upstream.status >= 400:
                return await _error_response(upstream)
            if body.get("stream"):
                return await self._stream_anthropic(
                    request, upstream, route, as_responses, response_slug=client_slug
                )
            payload = await upstream.json(content_type=None)
        if as_responses:
            return web.json_response(anthropic_to_response(payload, client_slug))
        return web.json_response(anthropic_to_chat_response(payload, client_slug))

    async def _stream_anthropic(
        self,
        request: web.Request,
        upstream,
        route: ShimModel,
        as_responses: bool,
        *,
        response_slug: str | None = None,
    ) -> web.StreamResponse:
        client_slug = response_slug or route.slug
        response = _sse_response()
        await response.prepare(request)
        if as_responses:
            state = ResponsesStreamState(client_slug)
        try:
            if as_responses:
                await state.start(response)
            async for line in _sse_lines(upstream):
                if line == "[DONE]":
                    break
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if as_responses:
                    await state.write_anthropic_delta(response, event)
                else:
                    await _write_sse(response, _anthropic_stream_to_chat_chunk(event, client_slug))
            if as_responses:
                await state.finish(response)
            else:
                await _safe_write(response, b"data: [DONE]\n\n")
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


def _sanitize_chatgpt_passthrough_body(body: dict[str, Any]) -> dict[str, Any]:
    sanitized = _sanitize_chatgpt_passthrough_value(body)
    return sanitized if isinstance(sanitized, dict) else {}


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
        output = {}
        for key, item in value.items():
            if key == "encrypted_content" and isinstance(item, str) and item.startswith(SHIM_ENCRYPTED_CONTENT_PREFIX):
                continue
            sanitized = _sanitize_chatgpt_passthrough_value(item)
            if sanitized is not _DROP_ITEM:
                output[key] = sanitized
        return output
    return value


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


class ResponsesStreamState:
    """Translates upstream chat-completions / anthropic stream events into the
    Codex Desktop Responses-API event sequence. Keeps the message item and
    each tool call as separate output items with stable indices, and emits
    proper .added / .delta / .done / .completed events plus a final
    `response.completed` with the full reconciled `output` array."""

    def __init__(self, model: str):
        self.response_id = f"resp_{int(time.time() * 1000)}"
        self.message_item_id = f"msg_{int(time.time() * 1000)}"
        self.model = model
        self.message_index: int | None = None  # output_index for the assistant message
        self.message_text = ""
        self.message_opened = False
        self.message_closed = False
        self.usage: dict[str, Any] | None = None
        self.tool_calls: dict[int, dict[str, Any]] = {}
        self.hidden_tool_calls: dict[int, dict[str, Any]] = {}
        # Synthesized function_call_output items for shim-handled tool calls
        # (currently only tool_search_call), keyed by call_id.
        self.tool_outputs: dict[str, dict[str, Any]] = {}
        # Reasoning (extended thinking) blocks, keyed by upstream index.
        self.reasoning_blocks: dict[Any, dict[str, Any]] = {}
        self.next_output_index = 0
        self.completed_turns: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    async def start(self, response: web.StreamResponse) -> None:
        await _write_sse(response, {"type": "response.created", "response": self._response("in_progress")})

    async def finish(self, response: web.StreamResponse) -> None:
        await self.close_turn_items(response)
        await _write_sse(response, {"type": "response.completed", "response": self._response("completed", final=True)})
        await response.write(b"data: [DONE]\n\n")

    async def close_turn_items(self, response: web.StreamResponse) -> None:
        for state in sorted(self.reasoning_blocks.values(), key=lambda s: s["output_index"]):
            if not state.get("closed"):
                await self._close_reasoning(response, state)
        if self.message_opened and not self.message_closed:
            await self._close_message(response)
        for state in sorted(self.tool_calls.values(), key=lambda s: s["output_index"]):
            if not state.get("closed"):
                await self._close_tool(response, state)
        await self._handle_tool_search(response)

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
        self.hidden_tool_calls = {}
        self.reasoning_blocks = {}
        self.tool_outputs = {}

    def _current_turn_dict(self) -> dict[str, Any]:
        return {
            "message_item_id": self.message_item_id,
            "message_index": self.message_index,
            "message_text": self.message_text,
            "message_opened": self.message_opened,
            "message_closed": self.message_closed,
            "tool_calls": dict(self.tool_calls),
            "reasoning_blocks": dict(self.reasoning_blocks),
            "tool_outputs": dict(self.tool_outputs),
        }

    async def _handle_tool_search(self, response: web.StreamResponse) -> None:
        """Resolve any tool_search_call items by running the MCP tools/list
        lookup on the shim side and emitting paired function_call_output
        events so Codex Desktop does not need to know about the virtual tool."""
        for state in sorted(self.tool_calls.values(), key=lambda s: s["output_index"]):
            if state.get("name") != mcp_search.MCP_TOOL_SEARCH_NAME:
                continue
            call_id = state.get("call_id") or state.get("id")
            if not call_id or call_id in self.tool_outputs:
                continue
            raw_args = state.get("arguments", "")
            if isinstance(raw_args, str):
                try:
                    args = json.loads(raw_args) if raw_args.strip() else {}
                except json.JSONDecodeError:
                    args = {}
            elif isinstance(raw_args, dict):
                args = raw_args
            else:
                args = {}
            query = args.get("query", "") if isinstance(args, dict) else ""
            if not isinstance(query, str) or not query.strip():
                result = mcp_search.format_tool_search_error("", "tool_search_call requires a non-empty string 'query' argument")
            else:
                result = await mcp_search.execute_tool_search(query)
            output_index = self.next_output_index
            self.next_output_index += 1
            output_item = {
                "id": f"out_{int(time.time() * 1000)}_{call_id}",
                "type": "function_call_output",
                "status": "completed",
                "call_id": call_id,
                "output": result,
            }
            self.tool_outputs[call_id] = {
                "output_index": output_index,
                "item": output_item,
            }
            await _write_sse(
                response,
                {
                    "type": "response.output_item.added",
                    "output_index": output_index,
                    "item": {
                        "id": output_item["id"],
                        "type": "function_call_output",
                        "status": "in_progress",
                        "call_id": call_id,
                        "output": "",
                    },
                },
            )
            await _write_sse(
                response,
                {
                    "type": "response.output_item.done",
                    "output_index": output_index,
                    "item": output_item,
                },
            )

    # ------------------------------------------------------------------
    # Chat-completions (OpenAI-style) deltas
    # ------------------------------------------------------------------
    async def write_chat_delta(
        self,
        response: web.StreamResponse,
        chunk: dict[str, Any],
        *,
        hide_tool_calls: bool = False,
    ) -> None:
        usage = chunk.get("usage")
        if isinstance(usage, dict):
            self.usage = normalize_responses_usage(usage)
        choice = (chunk.get("choices") or [{}])[0]
        delta = choice.get("delta") or {}
        reasoning = delta.get("reasoning_content") or delta.get("reasoning")
        if reasoning:
            await self._chat_reasoning_delta(response, reasoning)
        content = delta.get("content")
        if content:
            for state in list(self.reasoning_blocks.values()):
                if not state.get("closed"):
                    await self._close_reasoning(response, state)
            await self._text_delta(response, content)
        for call in delta.get("tool_calls") or []:
            await self._chat_tool_delta(response, call, hidden=hide_tool_calls)

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
        *,
        hidden: bool = False,
    ) -> None:
        index = int(call.get("index", 0))
        fn = call.get("function") or {}
        bucket = self.hidden_tool_calls if hidden else self.tool_calls
        state = bucket.get(index)
        if state is None:
            call_id = call.get("id") or f"call_{index}"
            if hidden:
                state = self._open_tool_internal(key=index, call_id=call_id, name=fn.get("name") or "", hidden=True)
            else:
                state = await self._open_tool(response, key=index, call_id=call_id, name=fn.get("name") or "")
        else:
            if fn.get("name"):
                state["name"] += fn["name"]
        arg_delta = fn.get("arguments") or ""
        if arg_delta:
            state["arguments"] += arg_delta
            if not hidden:
                await _write_sse(
                    response,
                    {
                        "type": "response.function_call_arguments.delta",
                        "item_id": state["id"],
                        "output_index": state["output_index"],
                        "delta": arg_delta,
                    },
                )

    def all_tool_calls(self) -> dict[int, dict[str, Any]]:
        return {**self.tool_calls, **self.hidden_tool_calls}

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
                await self._open_tool(
                    response,
                    key=("anthropic", idx),
                    call_id=block.get("id") or f"call_{idx}",
                    name=block.get("name") or "",
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

    async def _text_delta(self, response: web.StreamResponse, text: str) -> None:
        if not text:
            return
        if not self.message_opened:
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

    async def _open_tool(self, response: web.StreamResponse, *, key: Any, call_id: str, name: str) -> dict[str, Any]:
        if self.message_opened and not self.message_closed:
            await self._close_message(response)
        output_index = self.next_output_index
        self.next_output_index += 1
        state: dict[str, Any] = {
            "id": call_id,
            "call_id": call_id,
            "name": name,
            "arguments": "",
            "output_index": output_index,
            "closed": False,
        }
        self.tool_calls[key] = state
        await _write_sse(
            response,
            {
                "type": "response.output_item.added",
                "output_index": output_index,
                "item": {
                    "id": call_id,
                    "type": "function_call",
                    "status": "in_progress",
                    "call_id": call_id,
                    "name": name,
                    "arguments": "",
                },
            },
        )
        return state

    def _open_tool_internal(
        self,
        *,
        key: Any,
        call_id: str,
        name: str,
        hidden: bool = False,
    ) -> dict[str, Any]:
        output_index = self.next_output_index
        self.next_output_index += 1
        state: dict[str, Any] = {
            "id": call_id,
            "call_id": call_id,
            "name": name,
            "arguments": "",
            "output_index": output_index,
            "closed": False,
            "hidden": hidden,
        }
        target = self.hidden_tool_calls if hidden else self.tool_calls
        target[key] = state
        return state

    async def _close_tool(self, response: web.StreamResponse, state: dict[str, Any]) -> None:
        if state.get("hidden"):
            state["closed"] = True
            return
        state["closed"] = True
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

    def _tool_item(self, state: dict[str, Any], status: str) -> dict[str, Any]:
        return {
            "id": state["id"],
            "type": "function_call",
            "status": status,
            "call_id": state["call_id"],
            "name": state["name"],
            "arguments": state["arguments"],
        }

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
                ):
                    collected.append((turn["message_index"], self._message_item_for_turn(turn, "completed")))
                for state in turn["tool_calls"].values():
                    collected.append((state["output_index"], self._tool_item(state, "completed")))
                for entry in turn["tool_outputs"].values():
                    collected.append((entry["output_index"], entry["item"]))
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


def _join_url(base_url: str, endpoint: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        return base + endpoint
    if endpoint == "/messages":
        return base + "/v1/messages"
    return urljoin(base + "/", "v1" + endpoint)


def _openai_headers(route: ShimModel) -> dict[str, str]:
    headers = {"Content-Type": "application/json", **route.extra_headers}
    if route.api_key:
        headers.setdefault("Authorization", f"Bearer {route.api_key}")
    return headers


def _anthropic_headers(route: ShimModel) -> dict[str, str]:
    headers = {
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01",
        **route.extra_headers,
    }
    if route.api_key:
        headers.setdefault("x-api-key", route.api_key)
    return headers


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


async def _write_sse(response: web.StreamResponse, payload: dict[str, Any]) -> None:
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


class ClientDisconnected(Exception):
    """Raised when the downstream Codex client closes the SSE connection."""


def _log_incoming_request(endpoint: str, body: dict[str, Any]) -> None:
    try:
        tools = body.get("tools") or []
        names = []
        for t in tools[:80]:
            if isinstance(t, dict):
                name = t.get("name") or (t.get("function") or {}).get("name") or t.get("type")
                if name:
                    names.append(str(name))
        input_items = body.get("input") or []
        input_summary = []
        if isinstance(input_items, list):
            for item in input_items[-6:]:
                if isinstance(item, dict):
                    t = item.get("type") or item.get("role") or "?"
                    extra = ""
                    if t == "function_call":
                        extra = f"({item.get('name', '?')})"
                    elif t == "function_call_output":
                        extra = f"(call_id={str(item.get('call_id', ''))[:24]})"
                    input_summary.append(f"{t}{extra}")
        print(
            f"[req] {endpoint} model={body.get('model')!r} stream={body.get('stream')!r} "
            f"tools={len(tools)} ({names[:8]}) "
            f"input={len(input_items)} ({input_summary})",
            flush=True,
        )
    except Exception as exc:
        print(f"[req] failed to log: {exc}", flush=True)


async def _sse_lines(upstream) -> Any:
    buffer = b""
    async for chunk in upstream.content.iter_chunked(4096):
        buffer += chunk
        while b"\n" in buffer:
            raw, buffer = buffer.split(b"\n", 1)
            line = raw.decode("utf-8", errors="replace").strip()
            if line.startswith("data:"):
                yield line[5:].strip()
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
    summary = _compact_summary_from_output(output)
    compacted = _compact_response_payload(model, summary, payload.get("usage") if isinstance(payload, dict) else None)
    return web.json_response(compacted)


def _compact_summary_from_output(output: Any) -> str:
    parts: list[str] = []
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "message":
                content = item.get("content") or []
                if isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("text"):
                            parts.append(str(part["text"]))
            elif item.get("type") == "output_text" and item.get("text"):
                parts.append(str(item["text"]))
    return "\n".join(part for part in parts if part).strip()


def _compact_response_payload(model: str, summary: str, usage: Any = None) -> dict[str, Any]:
    now = int(time.time())
    response_id = f"resp_compact_{now}"
    text = summary or "No prior conversation state was available to compact."
    payload = {
        "id": response_id,
        "object": "response",
        "created_at": now,
        "status": "completed",
        "model": model,
        "output": [
            {
                "id": f"msg_compact_{now}",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
    }
    if usage is not None:
        payload["usage"] = usage
    return payload


async def _error_response(upstream, *, slug: str | None = None) -> web.Response:
    text = await upstream.text()
    if slug:
        print(
            f"[err] upstream {slug} returned {upstream.status}: {text[:500]}",
            flush=True,
        )
    return web.Response(status=upstream.status, text=text, content_type=upstream.content_type or "text/plain")


def _normalize_roles(messages: list[dict]) -> list[dict]:
    result = []
    for message in messages:
        if isinstance(message, dict):
            message = dict(message)
            if message.get("role") == "developer":
                message["role"] = "system"
        result.append(message)
    return result


def _chat_tools_have_mcp(tools: Any) -> bool:
    if not isinstance(tools, list):
        return False
    for t in tools:
        if not isinstance(t, dict):
            continue
        fn = t.get("function")
        name = t.get("name") if isinstance(t.get("name"), str) else None
        if not name and isinstance(fn, dict):
            name = fn.get("name")
        if isinstance(name, str) and name.startswith("mcp__"):
            return True
    return False


async def _pre_discover_if_mcp(body: dict[str, Any]) -> list[dict[str, Any]]:
    if not _chat_tools_have_mcp(body.get("tools")):
        return []
    return await mcp_search.pre_discover_mcp_tools()


def _inject_tool_search_if_mcp(
    body: dict[str, Any],
    discovered: list[dict[str, Any]] | None = None,
) -> None:
    if not _chat_tools_have_mcp(body.get("tools")):
        return
    tools = body.get("tools")
    if not isinstance(tools, list):
        return
    already = any(
        isinstance(t, dict)
        and (t.get("name") == mcp_search.MCP_TOOL_SEARCH_NAME
             or (isinstance(t.get("function"), dict)
                 and t["function"].get("name") == mcp_search.MCP_TOOL_SEARCH_NAME))
        for t in tools
    )
    filtered: list[dict[str, Any]] = []
    for t in tools:
        if not isinstance(t, dict):
            filtered.append(t)
            continue
        fn = t.get("function")
        name = t.get("name") if isinstance(t.get("name"), str) else None
        if not name and isinstance(fn, dict):
            name = fn.get("name")
        if isinstance(name, str) and name.startswith("mcp__") and name.count("__") == 1:
            continue
        filtered.append(t)
    prefix: list[dict[str, Any]] = []
    if discovered:
        existing = {
            (t.get("function") or {}).get("name") or t.get("name")
            for t in filtered
            if isinstance(t, dict)
        }
        for t in discovered:
            if not isinstance(t, dict):
                continue
            fn = t.get("function") or {}
            name = fn.get("name") or t.get("name")
            if not isinstance(name, str) or not name or name in existing:
                continue
            existing.add(name)
            prefix.append(t)
    body["tools"] = [*prefix, mcp_search.MCP_TOOL_SEARCH_DEFINITION, *filtered]


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
_PROVIDER_NAME_RE = re.compile(
    r'(\[model_providers\.' + re.escape(PROVIDER_NAME) + r'\][^\[]*?\n\s*name\s*=\s*")[^"]*(")',
    re.DOTALL,
)


def _set_active_model(slug: str, display_name: str | None = None) -> None:
    """Rewrite the active model + provider label in ~/.codex/config.toml."""
    if not CODEX_CONFIG_PATH.exists():
        return
    try:
        text = CODEX_CONFIG_PATH.read_text()
    except OSError:
        return
    text = _MODEL_LINE_RE.sub(rf'\g<1>{slug}\g<2>', text, count=1)
    if display_name:
        text = _PROVIDER_NAME_RE.sub(rf'\g<1>{display_name}\g<2>', text, count=1)
    try:
        CODEX_CONFIG_PATH.write_text(text)
    except OSError as exc:
        print(f"[switch] failed to write {CODEX_CONFIG_PATH}: {exc}", flush=True)
        return
    print(f"[switch] set active model to {slug} ({display_name})", flush=True)


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


def _picker_html() -> str:
    return '''<!DOCTYPE html>
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
      headers: {'Content-Type': 'application/json'},
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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings", type=Path, default=DEFAULT_SETTINGS)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args(argv)

    shim = ShimServer(args.settings, host=args.host)
    web.run_app(shim.app(), host=args.host, port=args.port, handle_signals=True)


if __name__ == "__main__":
    main()

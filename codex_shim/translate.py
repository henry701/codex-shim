from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

from . import mcp_search
from .compaction import decode_shim_compaction_summary
from .responses_input_pipeline import UNKNOWN_FUNCTION_TOOL_NAME
from .tool_translate import mcp_namespace, responses_function_call_ids


THINK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)


def namespaced_tool_chat_name(namespace: str, name: str) -> str:
    if namespace:
        return f"{namespace}.{name}"
    return name


def upstream_chat_tool_name(namespace: str, name: str) -> str:
    """OpenAI-compatible chat tool id (strict upstreams reject dots in names)."""
    return _sanitize_tool_name(namespaced_tool_chat_name(namespace, name))


def responses_tool_resolve_map(tools: Any) -> dict[str, tuple[str | None, str]]:
    if not isinstance(tools, list):
        return {}
    resolved: dict[str, tuple[str | None, str]] = {}
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        if tool.get("type") == "namespace":
            namespace = str(tool.get("name") or "")
            for sub_tool in tool.get("tools") or []:
                if not isinstance(sub_tool, dict) or sub_tool.get("type") != "function":
                    continue
                sub_name = str(sub_tool.get("name") or "")
                if not namespace or not sub_name:
                    continue
                resolved[upstream_chat_tool_name(namespace, sub_name)] = (namespace, sub_name)
            continue
        name = _responses_tool_function_name(tool)
        if name:
            resolved[name] = (None, name)
    return resolved


def resolve_namespaced_tool_name(
    raw_name: str,
    tool_resolve: dict[str, tuple[str | None, str]] | None = None,
) -> tuple[str | None, str]:
    if tool_resolve and raw_name in tool_resolve:
        return tool_resolve[raw_name]
    return split_namespaced_tool_chat_name(raw_name)


def split_namespaced_tool_chat_name(raw_name: str) -> tuple[str | None, str]:
    if mcp_search.parse_mcp_tool_reference(raw_name):
        return None, raw_name
    parts = raw_name.split(".", 1)
    if len(parts) == 2 and parts[0] and parts[1]:
        return parts[0], parts[1]
    return None, raw_name


def function_call_item_from_chat_tool(
    call: dict[str, Any],
    tool_types: dict[str, str] | None = None,
    tool_resolve: dict[str, tuple[str | None, str]] | None = None,
) -> dict[str, Any]:
    fn = call.get("function") or {}
    raw_name = fn.get("name", "")
    call_id = call.get("id", "call_0")
    original_type = original_responses_tool_type(raw_name, tool_types)
    if original_type == "apply_patch":
        return {
            "id": call_id,
            "type": "custom_tool_call",
            "status": "completed",
            "call_id": call_id,
            "name": "apply_patch",
            "input": fn.get("arguments", ""),
        }
    if original_type.startswith("web_search"):
        return _web_search_call_item(call_id, fn.get("arguments", ""))
    item_id, call_id = responses_function_call_ids(call_id)
    item: dict[str, Any] = {
        "id": item_id,
        "type": "function_call",
        "status": "completed",
        "call_id": call_id,
        "arguments": fn.get("arguments", ""),
    }
    mcp_parsed = mcp_search.parse_mcp_tool_reference(raw_name)
    if mcp_parsed:
        server, tool = mcp_parsed
        item["namespace"] = mcp_namespace(server)
        item["name"] = tool
        return item
    namespace, tool_name = resolve_namespaced_tool_name(raw_name, tool_resolve)
    if namespace is not None:
        item["namespace"] = namespace
        item["name"] = tool_name
    else:
        item["name"] = raw_name
    return item


def responses_tool_type_map(tools: Any) -> dict[str, str]:
    if not isinstance(tools, list):
        return {}
    mapped: dict[str, str] = {}
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        if tool.get("type") == "namespace":
            namespace = str(tool.get("name") or "")
            for sub_tool in tool.get("tools") or []:
                if not isinstance(sub_tool, dict) or sub_tool.get("type") != "function":
                    continue
                sub_name = str(sub_tool.get("name") or "")
                if namespace and sub_name:
                    mapped[upstream_chat_tool_name(namespace, sub_name)] = "function"
            continue
        tool_type = str(tool.get("type") or "").strip().lower()
        name = _responses_tool_function_name(tool)
        if name and tool_type:
            mapped[_sanitize_tool_name(name)] = tool_type
    return mapped


def original_responses_tool_type(name: str, tool_types: dict[str, str] | None = None) -> str:
    clean = _sanitize_tool_name(str(name or ""))
    if tool_types and clean in tool_types:
        return str(tool_types[clean] or "").strip().lower()
    if clean == "apply_patch":
        return "apply_patch"
    if clean in {"web_search", "web_search_preview"}:
        return "web_search"
    return ""


def _web_search_call_item(call_id: str, raw_arguments: Any, status: str = "completed") -> dict[str, Any]:
    args: dict[str, Any] = {}
    if isinstance(raw_arguments, str):
        try:
            parsed = json.loads(raw_arguments) if raw_arguments.strip() else {}
        except json.JSONDecodeError:
            parsed = {"query": raw_arguments}
        if isinstance(parsed, dict):
            args = parsed
    elif isinstance(raw_arguments, dict):
        args = raw_arguments
    query = str(args.get("query") or args.get("q") or args.get("search_query") or "")
    return {
        "id": call_id,
        "type": "web_search_call",
        "status": status,
        "call_id": call_id,
        "action": {"type": "search", "query": query},
    }


HOSTED_WEB_SEARCH_UNAVAILABLE_MESSAGE = (
    "The Codex hosted web_search tool returned no results and is unavailable at this time. "
    "Other search tools may still be available (for example MCP web_search_exa). "
    "Do not retry web_search; use an alternative search tool instead."
)


def _is_hosted_web_search_name(name: str) -> bool:
    clean = str(name or "").strip().lower()
    return clean in {"web_search", "web_search_preview"}


def _function_call_name_map(input_items: list[Any]) -> dict[str, str]:
    names: dict[str, str] = {}
    for item in input_items:
        if not isinstance(item, dict) or item.get("type") != "function_call":
            continue
        call_id = str(item.get("call_id") or item.get("id") or "")
        if call_id:
            names[call_id] = str(item.get("name") or "")
    return names


def _format_web_search_result_entries(entries: Any) -> list[str]:
    parts: list[str] = []
    if not isinstance(entries, list):
        return parts
    for entry in entries:
        if isinstance(entry, str):
            text = entry.strip()
            if text:
                parts.append(text)
            continue
        if not isinstance(entry, dict):
            continue
        snippet = str(
            entry.get("snippet")
            or entry.get("text")
            or entry.get("content")
            or entry.get("description")
            or ""
        ).strip()
        title = str(entry.get("title") or entry.get("name") or "").strip()
        url = str(entry.get("url") or "").strip()
        if snippet and title:
            parts.append(f"{title}\n{snippet}")
        elif snippet:
            parts.append(snippet)
        elif title and url:
            parts.append(f"{title}\n{url}")
        elif title:
            parts.append(title)
        elif url:
            parts.append(url)
    return parts


def _web_search_call_result_text(item: dict[str, Any]) -> str:
    parts: list[str] = []
    action = item.get("action")
    if isinstance(action, dict):
        parts.extend(_format_web_search_result_entries(action.get("sources")))
        parts.extend(_format_web_search_result_entries(action.get("results")))
    parts.extend(_format_web_search_result_entries(item.get("results")))
    output = item.get("output")
    if output not in (None, "", [], {}):
        text = _content_to_text(output).strip()
        if text:
            parts.append(text)
    return "\n\n".join(parts)


def _is_absolutely_empty_web_search_output(output: Any) -> bool:
    if output in (None, "", [], {}):
        return True
    if isinstance(output, list):
        return not any(not _is_absolutely_empty_web_search_output(entry) for entry in output)
    if isinstance(output, dict):
        if output.get("type") in {"input_text", "output_text", "text"}:
            return not str(output.get("text") or "").strip()
        meaningful = {
            key: value
            for key, value in output.items()
            if key not in {"type", "status", "call_id", "id"}
        }
        if not meaningful:
            return True
        return not any(not _is_absolutely_empty_web_search_output(value) for value in meaningful.values())
    return not str(output).strip()

SHIM_ENCRYPTED_CONTENT_PREFIX = "anthropic-thinking-v1:"
_THINKING_MAGIC = SHIM_ENCRYPTED_CONTENT_PREFIX

_CODEX_CLIENT_HEADERS = (
    "x-codex-installation-id",
    "x-codex-window-id",
    "x-codex-turn-state",
    "x-codex-turn-metadata",
    "x-codex-parent-thread-id",
)


def is_codex_client_headers(headers: Mapping[str, str]) -> bool:
    """True when the request looks like Codex Desktop / codex-cli, not a generic OpenAI client."""
    lowered = {str(k).lower(): str(v) for k, v in headers.items()}
    if any(lowered.get(name) for name in _CODEX_CLIENT_HEADERS):
        return True
    ua = lowered.get("user-agent", "")
    return "codex" in ua.lower()


_HOSTED_CODEX_TOOL_NAMES = frozenset(
    {
        "web_search",
        "web_search_preview",
        "image_generation",
        "computer_use",
        "computer_use_preview",
    }
)


def is_hosted_codex_tool(tool: Any) -> bool:
    """Codex-hosted tools that BYOK local models cannot execute."""
    if not isinstance(tool, dict):
        return False
    tool_type = str(tool.get("type") or "").strip().lower()
    if tool_type.startswith("web_search"):
        return True
    if tool_type.startswith("computer_use"):
        return True
    if tool_type == "image_generation":
        return True
    fn = tool.get("function")
    if isinstance(fn, dict):
        name = str(fn.get("name") or "").strip().lower()
        if name in _HOSTED_CODEX_TOOL_NAMES:
            return True
    name = str(tool.get("name") or "").strip().lower()
    return name in _HOSTED_CODEX_TOOL_NAMES


def omit_hosted_codex_tools(tools: Any) -> list[Any]:
    if not isinstance(tools, list):
        return []
    return [tool for tool in tools if not is_hosted_codex_tool(tool)]


def _tool_choice_targets_hosted_codex_tool(tool_choice: Any) -> bool:
    if not isinstance(tool_choice, dict):
        return False
    choice_type = str(tool_choice.get("type") or "").strip().lower()
    if choice_type.startswith("web_search"):
        return True
    if choice_type.startswith("computer_use"):
        return True
    if choice_type == "image_generation":
        return True
    name = str(tool_choice.get("name") or "").strip().lower()
    return name in _HOSTED_CODEX_TOOL_NAMES


def prepare_codex_byok_responses_body(
    body: dict[str, Any],
    headers: Mapping[str, str],
) -> dict[str, Any]:
    """Drop Codex-hosted tools (web search, image gen, computer use) for BYOK upstreams."""
    if not is_codex_client_headers(headers):
        return body
    tools = body.get("tools")
    if not isinstance(tools, list) or not any(is_hosted_codex_tool(tool) for tool in tools):
        return body
    prepared = dict(body)
    prepared["tools"] = omit_hosted_codex_tools(tools)
    if _tool_choice_targets_hosted_codex_tool(prepared.get("tool_choice")):
        prepared.pop("tool_choice", None)
    return prepared


def _decode_thinking_blob(encoded: Any) -> dict[str, Any] | None:
    import base64

    if not isinstance(encoded, str) or not encoded.startswith(_THINKING_MAGIC):
        return None
    blob = encoded[len(_THINKING_MAGIC) :]
    try:
        raw = base64.urlsafe_b64decode(blob.encode("ascii"))
        data = json.loads(raw.decode("utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    return data


def responses_to_chat(
    body: dict[str, Any],
    upstream_model: str,
) -> dict[str, Any]:
    messages = []
    instructions_text = _content_to_text(body.get("instructions")) if body.get("instructions") else ""
    if instructions_text:
        messages.append({"role": "system", "content": instructions_text})
    pending_reasoning: str | None = None
    for m in _responses_input_to_messages(body.get("input")):
        if m.get("_reasoning_only"):
            summary = m.get("summary") or []
            text = " ".join(item.get("text", "") for item in summary if isinstance(item, dict))
            if text:
                pending_reasoning = text
            continue
        if pending_reasoning and m.get("role") == "assistant":
            m["reasoning_content"] = pending_reasoning
            pending_reasoning = None
        messages.append(m)
    messages = _sanitize_chat_messages(_merge_consecutive_messages(_normalize_chat_roles(messages)))

    chat: dict[str, Any] = {
        "model": upstream_model,
        "messages": messages or [{"role": "user", "content": ""}],
        "stream": bool(body.get("stream", False)),
    }
    _copy_if_present(body, chat, "temperature")
    _copy_if_present(body, chat, "top_p")
    _copy_if_present(body, chat, "max_output_tokens", "max_tokens")
    _copy_if_present(body, chat, "max_tokens")
    _copy_if_present(body, chat, "parallel_tool_calls")
    _copy_if_present(body, chat, "reasoning_effort")

    tools = _responses_tools_to_chat_tools(body.get("tools"))
    if tools:
        chat["tools"] = tools
        tool_choice = _responses_tool_choice_to_chat(body.get("tool_choice"), body.get("tools"))
        if tool_choice is not None:
            chat["tool_choice"] = tool_choice
    return chat


def responses_to_anthropic(body: dict[str, Any], upstream_model: str, max_tokens: int | None) -> dict[str, Any]:
    system_parts: list[str] = []
    instructions = body.get("instructions")
    if instructions:
        system_parts.append(_content_to_text(instructions))

    messages: list[dict[str, Any]] = []

    def append(role: str, content: Any) -> None:
        if messages and messages[-1]["role"] == role and isinstance(messages[-1]["content"], list) and isinstance(content, list):
            messages[-1]["content"].extend(content)
        else:
            messages.append({"role": role, "content": content})

    pending_thinking: list[dict[str, Any]] = []
    for chat_msg in _responses_input_to_messages(body.get("input")):
        role = chat_msg.get("role", "user")
        if chat_msg.get("_reasoning_only"):
            decoded = _decode_thinking_blob(chat_msg.get("encrypted_content"))
            if decoded is not None:
                pending_thinking.append(decoded)
            else:
                # Summary-only fallback: emit a plain `thinking` block (no
                # signature). Anthropic requires `signature` on the original
                # session; if we lack it, skip rather than upsetting strict
                # APIs.
                for summary in chat_msg.get("summary") or []:
                    text = summary.get("text") if isinstance(summary, dict) else None
                    if text:
                        pending_thinking.append({"type": "thinking", "thinking": text, "signature": ""})
            continue
        if role in {"system", "developer"}:
            system_parts.append(_content_to_text(chat_msg.get("content", "")))
            continue
        if role == "assistant":
            blocks: list[dict[str, Any]] = []
            blocks.extend(pending_thinking)
            pending_thinking = []
            content = chat_msg.get("content")
            if content:
                blocks.extend(_chat_content_to_anthropic_blocks(content))
            for call in chat_msg.get("tool_calls") or []:
                fn = call.get("function") or {}
                args_raw = fn.get("arguments") or ""
                try:
                    args_obj = json.loads(args_raw) if args_raw else {}
                except json.JSONDecodeError:
                    args_obj = {"_raw": args_raw}
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": call.get("id") or "call_0",
                        "name": fn.get("name") or "",
                        "input": args_obj,
                    }
                )
            if blocks:
                append("assistant", blocks)
            continue
        if role == "tool":
            # Reasoning items only attach to assistant turns; drop any pending
            # thinking when a tool result interrupts (shouldn't happen in
            # normal Codex flows but defensive).
            pending_thinking = []
            append(
                "user",
                [
                    {
                        "type": "tool_result",
                        "tool_use_id": chat_msg.get("tool_call_id") or "call_0",
                        "content": _content_to_text(chat_msg.get("content", "")),
                    }
                ],
            )
            continue
        # user / anything else
        pending_thinking = []
        append(role, _chat_content_to_anthropic_content(chat_msg.get("content", "")))

    # If reasoning items appeared without a following assistant turn (e.g. the
    # final pending think after a tool_use round-trip), emit an assistant
    # message containing them so Anthropic's API accepts the followup.
    if pending_thinking:
        append("assistant", pending_thinking)

    anthropic: dict[str, Any] = {
        "model": upstream_model,
        "messages": messages or [{"role": "user", "content": ""}],
        "max_tokens": int(body.get("max_output_tokens") or body.get("max_tokens") or max_tokens or 4096),
        "stream": bool(body.get("stream", False)),
    }
    if system_parts:
        anthropic["system"] = "\n\n".join(system_parts)
    _copy_if_present(body, anthropic, "temperature")
    _copy_if_present(body, anthropic, "top_p")

    tools = _responses_tools_to_anthropic_tools(body.get("tools"))
    if tools:
        anthropic["tools"] = tools
    return anthropic


def chat_to_responses_request(body: dict[str, Any], upstream_model: str, max_tokens: int | None = None) -> dict[str, Any]:
    converted = {
        "model": upstream_model,
        "input": body.get("messages", []),
        "stream": bool(body.get("stream", False)),
    }
    for src, dst in [("temperature", "temperature"), ("top_p", "top_p"), ("max_tokens", "max_output_tokens")]:
        if src in body:
            converted[dst] = body[src]
    if max_tokens and "max_output_tokens" not in converted:
        converted["max_output_tokens"] = max_tokens
    if "tools" in body:
        converted["tools"] = body["tools"]
    return converted


def chat_to_anthropic(body: dict[str, Any], upstream_model: str, max_tokens: int | None) -> dict[str, Any]:
    pseudo_responses = chat_to_responses_request(body, upstream_model, max_tokens=max_tokens)
    return responses_to_anthropic(pseudo_responses, upstream_model, max_tokens)


def anthropic_messages_to_chat(body: dict[str, Any], upstream_model: str, max_tokens: int | None = None) -> dict[str, Any]:
    messages: list[dict[str, Any]] = []
    system = body.get("system")
    if system:
        messages.append({"role": "system", "content": _anthropic_content_to_text(system)})

    for raw_msg in body.get("messages") or []:
        if not isinstance(raw_msg, dict):
            continue
        role = str(raw_msg.get("role") or "user")
        content = raw_msg.get("content", "")
        if role == "assistant":
            messages.append(_anthropic_assistant_message_to_chat(content))
        elif role == "user":
            messages.extend(_anthropic_user_message_to_chat(content))
        else:
            messages.append({"role": role, "content": _anthropic_content_to_chat_content(content)})

    messages = _sanitize_chat_messages(_merge_consecutive_messages(_normalize_chat_roles(messages)))
    chat: dict[str, Any] = {
        "model": upstream_model,
        "messages": messages or [{"role": "user", "content": ""}],
        "stream": bool(body.get("stream", False)),
    }
    _copy_if_present(body, chat, "temperature")
    _copy_if_present(body, chat, "top_p")
    _copy_if_present(body, chat, "max_tokens")
    if max_tokens and "max_tokens" not in chat:
        chat["max_tokens"] = max_tokens
    _copy_if_present(body, chat, "stop_sequences", "stop")

    thinking = body.get("thinking")
    if isinstance(thinking, dict) and thinking.get("effort"):
        chat["reasoning_effort"] = thinking["effort"]
    output_config = body.get("output_config")
    if isinstance(output_config, dict) and output_config.get("effort"):
        chat["reasoning_effort"] = output_config["effort"]

    tools = _anthropic_tools_to_chat_tools(body.get("tools"))
    if tools:
        chat["tools"] = tools
        tool_choice = _anthropic_tool_choice_to_chat(body.get("tool_choice"))
        if tool_choice is not None:
            chat["tool_choice"] = tool_choice
    if chat["stream"]:
        stream_options = dict(body.get("stream_options") or {})
        stream_options["include_usage"] = True
        chat["stream_options"] = stream_options
    return chat


def chat_completion_to_anthropic_message(payload: dict[str, Any], requested_model: str) -> dict[str, Any]:
    choice = (payload.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content: list[dict[str, Any]] = []
    reasoning = message.get("reasoning_content") or message.get("reasoning")
    if reasoning:
        content.append({"type": "thinking", "thinking": str(reasoning)})
    text = strip_think(message.get("content") or "")
    if text:
        content.append({"type": "text", "text": text})
    for call in message.get("tool_calls") or []:
        if not isinstance(call, dict):
            continue
        fn = call.get("function") or {}
        args_raw = fn.get("arguments") or ""
        try:
            args_obj = json.loads(args_raw) if args_raw else {}
        except json.JSONDecodeError:
            args_obj = {"_raw": args_raw}
        content.append(
            {
                "type": "tool_use",
                "id": call.get("id") or "call_0",
                "name": fn.get("name") or "",
                "input": args_obj,
            }
        )
    if not content:
        content.append({"type": "text", "text": ""})

    response: dict[str, Any] = {
        "id": payload.get("id") or "msg_chat",
        "type": "message",
        "role": "assistant",
        "model": requested_model,
        "content": content,
        "stop_reason": _chat_finish_to_anthropic_stop(choice.get("finish_reason")),
        "stop_sequence": None,
    }
    usage = _responses_usage_to_anthropic_usage(normalize_responses_usage(payload.get("usage")))
    if usage is not None:
        response["usage"] = usage
    return response


def anthropic_to_chat_response(payload: dict[str, Any], requested_model: str) -> dict[str, Any]:
    content = ""
    tool_calls = []
    for block in payload.get("content", []):
        if block.get("type") == "text":
            content += block.get("text", "")
        elif block.get("type") == "tool_use":
            tool_calls.append(
                {
                    "id": block.get("id"),
                    "type": "function",
                    "function": {
                        "name": block.get("name", ""),
                        "arguments": _jsonish(block.get("input", {})),
                    },
                }
            )
    message: dict[str, Any] = {"role": "assistant", "content": strip_think(content)}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "id": payload.get("id", "chatcmpl-anthropic"),
        "object": "chat.completion",
        "created": 0,
        "model": requested_model,
        "choices": [{"index": 0, "message": message, "finish_reason": _anthropic_stop(payload.get("stop_reason"))}],
    }


def chat_completion_to_response(
    payload: dict[str, Any],
    requested_model: str,
    tool_types: dict[str, str] | None = None,
    tool_resolve: dict[str, tuple[str | None, str]] | None = None,
) -> dict[str, Any]:
    choice = (payload.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    output: list[dict[str, Any]] = []
    reasoning = message.get("reasoning_content")
    if reasoning:
        output.append(
            {
                "id": "reasoning_0",
                "type": "reasoning",
                "status": "completed",
                "summary": [{"type": "summary_text", "text": reasoning}],
            }
        )
    text = strip_think(message.get("content") or "")
    if text:
        output.append(
            {
                "id": "msg_0",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        )
    tool_types = tool_types or {}
    for call in message.get("tool_calls") or []:
        output.append(function_call_item_from_chat_tool(call, tool_types, tool_resolve))
    return {
        "id": payload.get("id", "resp_chat"),
        "object": "response",
        "created_at": payload.get("created", 0),
        "status": "completed",
        "model": requested_model,
        "output": output,
        "usage": normalize_responses_usage(payload.get("usage")),
    }


def anthropic_to_response(
    payload: dict[str, Any],
    requested_model: str,
    tool_types: dict[str, str] | None = None,
    tool_resolve: dict[str, tuple[str | None, str]] | None = None,
) -> dict[str, Any]:
    response = chat_completion_to_response(
        anthropic_to_chat_response(payload, requested_model),
        requested_model,
        tool_types,
        tool_resolve,
    )
    response["usage"] = normalize_responses_usage(payload.get("usage"))
    return response


def normalize_responses_usage(usage: Any) -> dict[str, Any] | None:
    if not isinstance(usage, dict):
        return None

    input_tokens = _int_token(usage.get("input_tokens"))
    if input_tokens is None:
        input_tokens = _int_token(usage.get("prompt_tokens"))

    output_tokens = _int_token(usage.get("output_tokens"))
    if output_tokens is None:
        output_tokens = _int_token(usage.get("completion_tokens"))

    total_tokens = _int_token(usage.get("total_tokens"))
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens

    if input_tokens is None:
        input_tokens = max(total_tokens - output_tokens, 0) if total_tokens is not None and output_tokens is not None else 0
    if output_tokens is None:
        output_tokens = max(total_tokens - input_tokens, 0) if total_tokens is not None else 0
    if total_tokens is None:
        total_tokens = input_tokens + output_tokens

    normalized: dict[str, Any] = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }

    input_details: dict[str, Any] = {}
    if isinstance(usage.get("input_tokens_details"), dict):
        input_details.update(usage["input_tokens_details"])
    if isinstance(usage.get("prompt_tokens_details"), dict):
        input_details.update(usage["prompt_tokens_details"])

    cache_read = _int_token(usage.get("cache_read_input_tokens"))
    if cache_read is not None:
        input_details.setdefault("cached_tokens", cache_read)
        input_details.setdefault("cache_read_input_tokens", cache_read)
    cache_created = _int_token(usage.get("cache_creation_input_tokens"))
    if cache_created is not None:
        input_details.setdefault("cache_creation_input_tokens", cache_created)

    if input_details:
        normalized["input_tokens_details"] = input_details

    output_details: dict[str, Any] = {}
    if isinstance(usage.get("output_tokens_details"), dict):
        output_details.update(usage["output_tokens_details"])
    if isinstance(usage.get("completion_tokens_details"), dict):
        output_details.update(usage["completion_tokens_details"])
    if output_details:
        normalized["output_tokens_details"] = output_details

    return normalized


def _int_token(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def strip_think(text: str) -> str:
    return THINK_RE.sub("", text or "")


def _responses_input_to_messages(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, str):
        return [{"role": "user", "content": value}]
    if not isinstance(value, list):
        return [{"role": "user", "content": _responses_content_to_chat_content(value)}]
    messages: list[dict[str, Any]] = []
    pending_tool_calls: list[dict[str, Any]] = []
    deferred_tool_outputs: dict[str, list[dict[str, Any]]] = {}
    emitted_tool_call_ids: set[str] = set()
    function_call_names = _function_call_name_map(value) if isinstance(value, list) else {}

    def flush_pending_assistant_tool_calls():
        if pending_tool_calls:
            messages.append({"role": "assistant", "content": None, "tool_calls": list(pending_tool_calls)})
            for tool_call in pending_tool_calls:
                call_id = str(tool_call.get("id") or "")
                if call_id:
                    emitted_tool_call_ids.add(call_id)
                    for deferred in deferred_tool_outputs.pop(call_id, []):
                        messages.append(deferred)
            pending_tool_calls.clear()

    for item in value:
        if isinstance(item, str):
            flush_pending_assistant_tool_calls()
            messages.append({"role": "user", "content": item})
            continue
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type in {"message", None} and "role" in item:
            flush_pending_assistant_tool_calls()
            role = item.get("role", "user")
            if role == "developer":
                role = "system"
            messages.append({"role": role, "content": _responses_content_to_chat_content(item.get("content", ""))})
        elif item_type in {"input_text", "text", "input_image"}:
            flush_pending_assistant_tool_calls()
            messages.append({"role": "user", "content": _responses_content_to_chat_content(item)})
        elif item_type == "computer_call_output":
            flush_pending_assistant_tool_calls()
            messages.append({"role": "user", "content": _computer_output_to_chat_content(item)})
        elif item_type == "function_call":
            # Coalesce consecutive function_call items into a single assistant
            # message with multiple tool_calls so chat-completions upstreams
            # accept the subsequent tool messages.
            call_id = item.get("call_id") or item.get("id") or "call_0"
            fn_name = item.get("name") or ""
            namespace = item.get("namespace") or ""
            if namespace and fn_name:
                fn_name = upstream_chat_tool_name(namespace, fn_name)
            pending_tool_calls.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": fn_name,
                        "arguments": item.get("arguments") or "",
                    },
                }
            )
        elif item_type == "function_call_output":
            output = item.get("output", "")
            call_id = str(item.get("call_id") or "")
            tool_name = function_call_names.get(call_id, "")
            if _is_hosted_web_search_name(tool_name) and _is_absolutely_empty_web_search_output(output):
                content = HOSTED_WEB_SEARCH_UNAVAILABLE_MESSAGE
            else:
                content = _content_to_text(output)
            tool_messages = [
                {"role": "tool", "tool_call_id": call_id or item.get("call_id"), "content": content}
            ]
            if _has_visual_content(output):
                tool_messages.append(
                    {"role": "user", "content": _visual_feedback_chat_content(output, item.get("call_id"))}
                )
            pending_ids = {str(tool_call.get("id") or "") for tool_call in pending_tool_calls}
            if call_id and call_id in pending_ids:
                flush_pending_assistant_tool_calls()
                messages.extend(tool_messages)
            elif call_id and call_id in emitted_tool_call_ids:
                messages.extend(tool_messages)
            elif call_id:
                deferred_tool_outputs.setdefault(call_id, []).extend(tool_messages)
            else:
                flush_pending_assistant_tool_calls()
                messages.extend(tool_messages)
        elif item_type == "web_search_call":
            flush_pending_assistant_tool_calls()
            call_id = str(item.get("call_id") or item.get("id") or "call_0")
            action = item.get("action") if isinstance(item.get("action"), dict) else {}
            query = str(action.get("query") or "")
            arguments = json.dumps({"query": query}) if query else "{}"
            pending_tool_calls.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "arguments": arguments,
                    },
                }
            )
            flush_pending_assistant_tool_calls()
            result_text = _web_search_call_result_text(item)
            if not result_text.strip():
                result_text = HOSTED_WEB_SEARCH_UNAVAILABLE_MESSAGE
            messages.append({"role": "tool", "tool_call_id": call_id, "content": result_text})
        elif item_type == "tool_search_call":
            flush_pending_assistant_tool_calls()
            call_id = item.get("call_id") or item.get("id") or "call_0"
            args = item.get("arguments") or {}
            pending_tool_calls.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": mcp_search.CODEX_TOOL_SEARCH_NAME,
                        "arguments": json.dumps(args) if isinstance(args, dict) else str(args),
                    },
                }
            )
        elif item_type == "tool_search_output":
            flush_pending_assistant_tool_calls()
            tools = item.get("tools")
            if tools is None:
                content = json.dumps(item, indent=2)
            else:
                content = json.dumps(
                    {"tools": mcp_search.flatten_tool_search_tools(tools)},
                    indent=2,
                )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": item.get("call_id"),
                    "content": content,
                }
            )
        elif item_type == "mcp_tool_call":
            flush_pending_assistant_tool_calls()
            result = item.get("result")
            if not result:
                continue
            content = _content_to_text(result)
            if not content:
                continue
            server = item.get("server") or "mcp"
            tool = item.get("tool") or "tool"
            messages.append(
                {
                    "role": "user",
                    "content": f"MCP tool {server}/{tool} result:\n{content}",
                }
            )
        elif item_type == "compaction":
            flush_pending_assistant_tool_calls()
            summary = decode_shim_compaction_summary(item.get("encrypted_content"))
            if summary:
                messages.append(
                    {
                        "role": "user",
                        "content": f"Compacted conversation state:\n{summary}",
                    }
                )
        elif item_type == "agent_message":
            # Codex collaboration delivers parent->child task text as `agent_message`.
            # Dropping it leaves a spawned sub-agent with no task at all.
            flush_pending_assistant_tool_calls()
            agent_message = _agent_message_to_chat_message(item)
            if agent_message:
                messages.append(agent_message)
        elif item_type == "compaction_trigger":
            continue
        elif item_type == "reasoning":
            # For Chat-Completions upstreams reasoning is informational only.
            # We keep it as a marker so the Anthropic translator can reattach
            # encrypted_content as a `thinking` block on the assistant turn.
            flush_pending_assistant_tool_calls()
            messages.append(
                {
                    "role": "assistant",
                    "_reasoning_only": True,
                    "encrypted_content": item.get("encrypted_content"),
                    "summary": item.get("summary") or [],
                    "content": None,
                }
            )
    flush_pending_assistant_tool_calls()
    for call_id, deferred in deferred_tool_outputs.items():
        messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": UNKNOWN_FUNCTION_TOOL_NAME,
                            "arguments": "{}",
                        },
                    }
                ],
            }
        )
        messages.extend(deferred)
    return messages


def _agent_message_looks_like_plaintext(value: str) -> bool:
    """Inter-agent payloads ride in `encrypted_content` parts that are usually plain text.

    Opaque reasoning blobs use the same field name, so only accept values that read
    like prose rather than a single long token.
    """
    text = value.strip()
    if not text:
        return False
    if len(text) > 64 and not re.search(r"\s", text):
        return False
    printable = sum(1 for ch in text if ch.isprintable() or ch in "\n\r\t")
    return printable / len(text) > 0.9


def _agent_message_text(item: dict[str, Any]) -> str:
    parts: list[str] = []
    content = item.get("content")
    if isinstance(content, str):
        parts.append(content)
    elif isinstance(content, list):
        for part in content:
            if isinstance(part, str):
                parts.append(part)
                continue
            if not isinstance(part, dict):
                continue
            part_type = part.get("type")
            if part_type in {"text", "input_text", "output_text"}:
                parts.append(str(part.get("text") or ""))
            elif part_type == "encrypted_content":
                blob = str(part.get("encrypted_content") or "")
                if _agent_message_looks_like_plaintext(blob):
                    parts.append(blob)
    direct = item.get("message")
    if isinstance(direct, str):
        parts.append(direct)
    return "\n".join(part for part in parts if part and part.strip()).strip()


def _agent_message_to_chat_message(item: dict[str, Any]) -> dict[str, Any] | None:
    text = _agent_message_text(item)
    if not text:
        return None
    author = str(item.get("author") or "").strip()
    recipient = str(item.get("recipient") or "").strip()
    if not author and not recipient:
        # No routing metadata: this is the agent's own prior message.
        return {"role": "assistant", "content": text}
    header = f"[agent message from {author or 'unknown'}"
    if recipient:
        header += f" to {recipient}"
    header += "]"
    return {"role": "user", "content": f"{header}\n{text}"}


def _responses_content_to_chat_content(content: Any) -> str | list[dict[str, Any]]:
    parts = _chat_parts_from_content(content)
    if not parts:
        return ""
    if any(part.get("type") == "image_url" for part in parts):
        return parts
    return "\n".join(str(part.get("text", "")) for part in parts if part.get("type") == "text")


def _computer_output_to_chat_content(item: dict[str, Any]) -> str | list[dict[str, Any]]:
    call_id = item.get("call_id") or item.get("id")
    prefix = f"Computer output for {call_id}." if call_id else "Computer output."
    parts = _chat_parts_from_content(item.get("output", ""))
    if any(part.get("type") == "image_url" for part in parts):
        return [{"type": "text", "text": prefix}, *parts]
    text = "\n".join(str(part.get("text", "")) for part in parts if part.get("type") == "text")
    return f"{prefix}\n{text}" if text else prefix


def _visual_feedback_chat_content(output: Any, call_id: Any) -> list[dict[str, Any]]:
    prefix = f"Visual tool output for {call_id}." if call_id else "Visual tool output."
    parts = [part for part in _chat_parts_from_content(output) if part.get("type") == "image_url"]
    return [{"type": "text", "text": prefix}, *parts]


def _chat_parts_from_content(content: Any) -> list[dict[str, Any]]:
    if content is None:
        return []
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content else []
    if isinstance(content, list):
        parts: list[dict[str, Any]] = []
        for part in content:
            parts.extend(_chat_parts_from_content(part))
        return parts
    if isinstance(content, dict):
        content_type = str(content.get("type") or "")
        if content_type in {"input_text", "output_text", "text"}:
            text = str(content.get("text", ""))
            return [{"type": "text", "text": text}] if text else []
        if content_type in {"input_image", "image_url"} or "image_url" in content:
            image = _chat_image_part(content)
            return [image] if image else []
        if content_type == "computer_call_output":
            return _chat_parts_from_content(content.get("output"))
        if "output" in content and _has_visual_content(content.get("output")):
            return _chat_parts_from_content(content.get("output"))
        if "content" in content:
            return _chat_parts_from_content(content["content"])
        if "text" in content:
            text = str(content.get("text", ""))
            return [{"type": "text", "text": text}] if text else []
    return []


def _chat_image_part(part: dict[str, Any]) -> dict[str, Any] | None:
    url = _image_url_from_part(part)
    if not url:
        return None
    image_url: dict[str, Any] = {"url": url}
    detail = part.get("detail") or part.get("image_detail")
    if detail and detail not in ("low", "auto", "high", "xhigh"):
        # Codex Desktop sends "original" which is not a standard OpenAI Chat
        # Completions value — providers like Kimi K2.6 reject it (400).
        # Map it to "high" (the closest standard equivalent). Any unknown
        # detail value falls back to "auto".
        detail = "high" if detail == "original" else "auto"
    if detail:
        image_url["detail"] = _normalize_chat_image_detail(str(detail))
    return {"type": "image_url", "image_url": image_url}


def _normalize_chat_image_detail(detail: str) -> str:
    normalized = detail.strip().lower()
    if normalized in {"low", "auto", "high"}:
        return normalized
    if normalized == "original":
        return "high"
    return "auto"


def _image_url_from_part(part: dict[str, Any]) -> str:
    image_url = part.get("image_url")
    if isinstance(image_url, str):
        return image_url
    if isinstance(image_url, dict) and isinstance(image_url.get("url"), str):
        return image_url["url"]
    for key in ("url", "file_url"):
        value = part.get(key)
        if isinstance(value, str):
            return value
    return ""


def _has_visual_content(content: Any) -> bool:
    return any(part.get("type") == "image_url" for part in _chat_parts_from_content(content))


def _chat_content_to_anthropic_content(content: Any) -> str | list[dict[str, Any]]:
    blocks = _chat_content_to_anthropic_blocks(content)
    if not any(block.get("type") == "image" for block in blocks):
        return "\n".join(block.get("text", "") for block in blocks if block.get("type") == "text")
    return blocks


def _chat_content_to_anthropic_blocks(content: Any) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for part in _chat_parts_from_content(content):
        if part.get("type") == "text":
            text = str(part.get("text", ""))
            if text:
                blocks.append({"type": "text", "text": text})
        elif part.get("type") == "image_url":
            block = _chat_image_part_to_anthropic(part)
            if block:
                blocks.append(block)
    return blocks or [{"type": "text", "text": ""}]


def _chat_image_part_to_anthropic(part: dict[str, Any]) -> dict[str, Any] | None:
    image_url = part.get("image_url")
    url = ""
    if isinstance(image_url, dict):
        url = str(image_url.get("url") or "")
    elif isinstance(image_url, str):
        url = image_url
    if not url:
        return None
    if url.startswith("data:"):
        match = re.match(r"data:([^;,]+);base64,(.*)", url, re.DOTALL)
        if not match:
            return None
        return {"type": "image", "source": {"type": "base64", "media_type": match.group(1), "data": match.group(2)}}
    return {"type": "image", "source": {"type": "url", "url": url}}


def _anthropic_content_to_chat_content(content: Any) -> str | list[dict[str, Any]]:
    parts = _anthropic_content_to_chat_parts(content)
    if not parts:
        return ""
    if any(part.get("type") == "image_url" for part in parts):
        return parts
    return "\n".join(str(part.get("text", "")) for part in parts if part.get("type") == "text")


def _anthropic_content_to_text(content: Any) -> str:
    chat_content = _anthropic_content_to_chat_content(content)
    return _content_to_text(chat_content)


def _anthropic_content_to_chat_parts(content: Any) -> list[dict[str, Any]]:
    if content is None:
        return []
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content else []
    if isinstance(content, list):
        parts: list[dict[str, Any]] = []
        for block in content:
            if not isinstance(block, dict):
                parts.extend(_anthropic_content_to_chat_parts(block))
                continue
            block_type = block.get("type")
            if block_type == "text":
                text = str(block.get("text", ""))
                if text:
                    parts.append({"type": "text", "text": text})
            elif block_type == "image":
                image = _anthropic_image_block_to_chat_part(block)
                if image:
                    parts.append(image)
            elif block_type == "image_url" or "image_url" in block:
                image = _chat_image_part(block)
                if image:
                    parts.append(image)
            elif "content" in block:
                parts.extend(_anthropic_content_to_chat_parts(block.get("content")))
        return parts
    if isinstance(content, dict):
        return _anthropic_content_to_chat_parts([content])
    return [{"type": "text", "text": str(content)}]


def _anthropic_image_block_to_chat_part(block: dict[str, Any]) -> dict[str, Any] | None:
    source = block.get("source")
    if not isinstance(source, dict):
        return None
    url = ""
    if source.get("type") == "base64":
        media_type = str(source.get("media_type") or "image/png")
        data = str(source.get("data") or "")
        if data:
            url = f"data:{media_type};base64,{data}"
    elif source.get("type") == "url":
        url = str(source.get("url") or "")
    if not url:
        return None
    return {"type": "image_url", "image_url": {"url": url}}


def _anthropic_assistant_message_to_chat(content: Any) -> dict[str, Any]:
    text_parts: list[Any] = []
    tool_calls: list[dict[str, Any]] = []
    reasoning_parts: list[str] = []
    blocks = content if isinstance(content, list) else [{"type": "text", "text": content}]
    for block in blocks:
        if not isinstance(block, dict):
            text_parts.append(block)
            continue
        block_type = block.get("type")
        if block_type == "tool_use":
            tool_calls.append(
                {
                    "id": block.get("id") or f"call_{len(tool_calls)}",
                    "type": "function",
                    "function": {
                        "name": block.get("name") or "",
                        "arguments": _jsonish(block.get("input", {})),
                    },
                }
            )
        elif block_type in {"thinking", "redacted_thinking"}:
            thinking = block.get("thinking") or block.get("data") or ""
            if thinking:
                reasoning_parts.append(str(thinking))
        else:
            text_parts.append(block)
    message: dict[str, Any] = {"role": "assistant", "content": _anthropic_content_to_chat_content(text_parts) if text_parts else ""}
    if tool_calls:
        message["tool_calls"] = tool_calls
    if reasoning_parts:
        message["reasoning_content"] = "\n".join(reasoning_parts)
    return message


def _anthropic_user_message_to_chat(content: Any) -> list[dict[str, Any]]:
    if not isinstance(content, list):
        return [{"role": "user", "content": _anthropic_content_to_chat_content(content)}]
    tool_messages: list[dict[str, Any]] = []
    user_parts: list[Any] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "tool_result":
            tool_content = block.get("content", "")
            tool_use_id = block.get("tool_use_id") or "call_0"
            tool_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_use_id,
                    "content": _anthropic_content_to_text(tool_content),
                }
            )
            if _anthropic_has_visual_content(tool_content):
                user_parts.extend(_anthropic_visual_tool_result_chat_parts(tool_content, tool_use_id))
        else:
            user_parts.append(block)
    messages = list(tool_messages)
    if user_parts or not messages:
        messages.append({"role": "user", "content": _anthropic_content_to_chat_content(user_parts)})
    return messages


def _anthropic_has_visual_content(content: Any) -> bool:
    return any(part.get("type") == "image_url" for part in _anthropic_content_to_chat_parts(content))


def _anthropic_visual_tool_result_chat_parts(content: Any, tool_use_id: Any) -> list[dict[str, Any]]:
    prefix = f"Visual tool result for {tool_use_id}." if tool_use_id else "Visual tool result."
    images = [part for part in _anthropic_content_to_chat_parts(content) if part.get("type") == "image_url"]
    return [{"type": "text", "text": prefix}, *images]


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                if part.get("type") in {"input_text", "output_text", "text"}:
                    parts.append(str(part.get("text", "")))
                elif part.get("type") in {"input_image", "image_url"} or "image_url" in part:
                    parts.append("[image]")
                elif "content" in part:
                    parts.append(_content_to_text(part["content"]))
        return "\n".join(p for p in parts if p)
    if isinstance(content, dict):
        if content.get("type") in {"input_image", "image_url"} or "image_url" in content:
            return "[image]"
        if "output" in content:
            return _content_to_text(content.get("output"))
        if "text" in content:
            return str(content.get("text", ""))
        return str(content)
    return str(content)


def _responses_tools_to_chat_tools(tools: Any) -> list[dict[str, Any]]:
    if not isinstance(tools, list):
        return []
    converted = []
    for tool in tools:
        if isinstance(tool, dict) and tool.get("type") == "namespace":
            namespace = str(tool.get("name") or "")
            desc = str(tool.get("description") or f"Tools in the {namespace} namespace.")
            for sub_tool in tool.get("tools") or []:
                if not isinstance(sub_tool, dict):
                    continue
                if sub_tool.get("type") != "function":
                    continue
                sub_name = str(sub_tool.get("name") or "")
                if not sub_name:
                    continue
                converted.append(
                    {
                        "type": "function",
                        "function": {
                            "name": upstream_chat_tool_name(namespace, sub_name),
                            "description": sub_tool.get("description") or desc,
                            "parameters": sub_tool.get("parameters")
                            or {"type": "object", "properties": {}},
                        },
                    }
                )
            continue
        function_tool = _responses_tool_to_chat_function(tool)
        if function_tool is None:
            continue
        fn = function_tool.get("function") or {}
        name = fn.get("name") or ""
        if mcp_search.is_deferred_mcp_server_stub(name):
            continue
        converted.append(function_tool)
    return converted


def _responses_tool_to_chat_function(tool: Any) -> dict[str, Any] | None:
    if not isinstance(tool, dict):
        return None
    if tool.get("type") == "function" and "function" in tool:
        return tool
    name = _responses_tool_function_name(tool)
    if not name:
        return None
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": tool.get("description") or _native_tool_description(tool),
            "parameters": tool.get("parameters") or _native_tool_parameters(tool),
        },
    }


def _responses_tool_function_name(tool: dict[str, Any]) -> str:
    fn = tool.get("function")
    if isinstance(fn, dict) and fn.get("name"):
        return _sanitize_tool_name(str(fn["name"]))
    if tool.get("name"):
        return _sanitize_tool_name(str(tool["name"]))
    tool_type = str(tool.get("type") or "").strip().lower()
    if tool_type == mcp_search.CODEX_TOOL_SEARCH_TYPE:
        return mcp_search.CODEX_TOOL_SEARCH_NAME
    aliases = {
        "web_search": "web_search",
        "web_search_preview": "web_search",
        "computer_use": "computer_use",
        "computer_use_preview": "computer_use",
        "apply_patch": "apply_patch",
        "local_shell": "local_shell",
        "shell": "local_shell",
    }
    if tool_type in aliases:
        return aliases[tool_type]
    if tool_type.startswith("mcp"):
        name = tool.get("name")
        if isinstance(name, str) and name:
            return _sanitize_tool_name(name)
        return _sanitize_tool_name(tool_type)
    return ""


def _sanitize_tool_name(name: str) -> str:
    clean = re.sub(r"[^a-zA-Z0-9_-]+", "_", name.strip())[:64]
    return clean.strip("_") or "tool"


def _native_tool_description(tool: dict[str, Any]) -> str:
    tool_type = str(tool.get("type") or "tool")
    if tool_type.startswith("web_search"):
        return "Search the web using Codex's web-search tool fallback."
    if tool_type.startswith("computer_use"):
        return "Request a Codex computer-use action."
    if tool_type == "apply_patch":
        return "Apply a unified diff patch to the working tree."
    if tool_type in {"local_shell", "shell"}:
        return "Run a local shell command through Codex."
    if tool_type.startswith("mcp"):
        return "Interact with Codex MCP resources."
    return f"Codex tool fallback for Responses tool type {tool_type}."


def _native_tool_parameters(tool: dict[str, Any]) -> dict[str, Any]:
    tool_type = str(tool.get("type") or "").strip().lower()
    if tool_type.startswith("web_search"):
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
            },
            "required": ["query"],
            "additionalProperties": True,
        }
    if tool_type.startswith("computer_use"):
        return {
            "type": "object",
            "properties": {
                "action": {"type": "string", "description": "Computer action to perform"},
                "x": {"type": "number", "description": "Screen x coordinate, when relevant"},
                "y": {"type": "number", "description": "Screen y coordinate, when relevant"},
                "text": {"type": "string", "description": "Text to type, when relevant"},
            },
            "required": ["action"],
            "additionalProperties": True,
        }
    if tool_type == "apply_patch":
        return {
            "type": "object",
            "properties": {"patch": {"type": "string", "description": "Unified diff patch"}},
            "required": ["patch"],
            "additionalProperties": True,
        }
    if tool_type in {"local_shell", "shell"}:
        return {
            "type": "object",
            "properties": {"command": {"type": "string", "description": "Shell command to run"}},
            "required": ["command"],
            "additionalProperties": True,
        }
    return {"type": "object", "properties": {"input": {"type": "string"}}, "additionalProperties": True}


def _responses_tool_choice_to_chat(tool_choice: Any, tools: Any) -> Any:
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        if tool_choice in {"auto", "none", "required"}:
            return tool_choice
        name = _tool_choice_name(tool_choice, tools)
        return {"type": "function", "function": {"name": name}} if name else tool_choice
    if isinstance(tool_choice, dict):
        if tool_choice.get("type") == "function" and "function" in tool_choice:
            return tool_choice
        name = _tool_choice_name(str(tool_choice.get("name") or tool_choice.get("type") or ""), tools)
        return {"type": "function", "function": {"name": name}} if name else tool_choice
    return tool_choice


def _tool_choice_name(choice: str, tools: Any) -> str:
    choice = choice.lower().strip()
    if isinstance(tools, list):
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            names = {
                str(tool.get("type") or "").lower(),
                str(tool.get("name") or "").lower(),
            }
            fn = tool.get("function")
            if isinstance(fn, dict):
                names.add(str(fn.get("name") or "").lower())
            if choice in names:
                return _responses_tool_function_name(tool)
    return _sanitize_tool_name(choice)


def _responses_tools_to_anthropic_tools(tools: Any) -> list[dict[str, Any]]:
    chat_tools = _responses_tools_to_chat_tools(tools)
    converted = []
    for tool in chat_tools:
        fn = tool.get("function") or {}
        converted.append(
            {
                "name": fn.get("name"),
                "description": fn.get("description", ""),
                "input_schema": fn.get("parameters") or {"type": "object", "properties": {}},
            }
        )
    return [tool for tool in converted if tool.get("name")]


def _anthropic_tools_to_chat_tools(tools: Any) -> list[dict[str, Any]]:
    if not isinstance(tools, list):
        return []
    converted: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        name = tool.get("name")
        if not name:
            continue
        converted.append(
            {
                "type": "function",
                "function": {
                    "name": str(name),
                    "description": tool.get("description") or "",
                    "parameters": tool.get("input_schema") or {"type": "object", "properties": {}},
                },
            }
        )
    return converted


def _anthropic_tool_choice_to_chat(tool_choice: Any) -> Any:
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        if tool_choice in {"auto", "none"}:
            return tool_choice
        if tool_choice == "any":
            return "required"
        return {"type": "function", "function": {"name": tool_choice}}
    if isinstance(tool_choice, dict):
        choice_type = tool_choice.get("type")
        if choice_type in {"auto", "none"}:
            return choice_type
        if choice_type == "any":
            return "required"
        if choice_type == "tool" and tool_choice.get("name"):
            return {"type": "function", "function": {"name": str(tool_choice["name"])}}
    return tool_choice


def _chat_finish_to_anthropic_stop(reason: Any) -> str:
    if reason in {"tool_calls", "function_call"}:
        return "tool_use"
    if reason == "length":
        return "max_tokens"
    if reason == "content_filter":
        return "refusal"
    return "end_turn"


def _responses_usage_to_anthropic_usage(usage: dict[str, Any] | None) -> dict[str, Any] | None:
    if usage is None:
        return None
    result = {
        "input_tokens": int(usage.get("input_tokens") or 0),
        "output_tokens": int(usage.get("output_tokens") or 0),
    }
    input_details = usage.get("input_tokens_details")
    if isinstance(input_details, dict):
        cache_read = input_details.get("cache_read_input_tokens", input_details.get("cached_tokens"))
        if isinstance(cache_read, int) and not isinstance(cache_read, bool):
            result["cache_read_input_tokens"] = cache_read
        cache_created = input_details.get("cache_creation_input_tokens")
        if isinstance(cache_created, int) and not isinstance(cache_created, bool):
            result["cache_creation_input_tokens"] = cache_created
    return result


def _copy_if_present(src: dict[str, Any], dst: dict[str, Any], src_key: str, dst_key: str | None = None) -> None:
    if src_key in src and src[src_key] is not None:
        dst[dst_key or src_key] = src[src_key]


def _anthropic_stop(reason: Any) -> str:
    return "tool_calls" if reason == "tool_use" else "stop"


def _jsonish(value: Any) -> str:
    import json

    if isinstance(value, str):
        return value
    return json.dumps(value, separators=(",", ":"))


def _sanitize_string(value: str) -> str:
    value = value.replace("\x00", "")
    return "".join(char for char in value if char in "\n\r\t" or ord(char) >= 0x20)


def _sanitize_chat_content_parts(parts: list[Any]) -> list[dict[str, Any]]:
    cleaned = []
    for part in parts:
        if isinstance(part, str):
            cleaned.append({"type": "text", "text": _sanitize_string(part)})
            continue
        if not isinstance(part, dict):
            continue
        current = dict(part)
        if current.get("type") == "text":
            current["text"] = _sanitize_string(str(current.get("text", "")))
        elif current.get("type") == "image_url":
            image_url = current.get("image_url")
            if isinstance(image_url, dict):
                current["image_url"] = {k: _sanitize_string(str(v)) for k, v in image_url.items() if v is not None}
            elif isinstance(image_url, str):
                current["image_url"] = {"url": _sanitize_string(image_url)}
        cleaned.append(current)
    return cleaned


def _sanitize_chat_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cleaned = []
    for message in messages:
        current = dict(message)
        current.pop("_reasoning_only", None)
        current.pop("encrypted_content", None)
        current.pop("summary", None)
        role = current.get("role", "user")
        content = current.get("content")
        if content is None:
            current["content"] = None if role == "assistant" else ""
        elif isinstance(content, list):
            current["content"] = _sanitize_chat_content_parts(content)
        elif isinstance(content, str):
            current["content"] = _sanitize_string(content)
        else:
            current["content"] = _sanitize_string(_content_to_text(content))

        if isinstance(current.get("reasoning_content"), str):
            current["reasoning_content"] = _sanitize_string(current["reasoning_content"])
        tool_calls = current.get("tool_calls")
        if tool_calls:
            copied_calls = []
            for call in tool_calls:
                if not isinstance(call, dict):
                    continue
                copied_call = dict(call)
                if isinstance(copied_call.get("id"), str):
                    copied_call["id"] = _sanitize_string(copied_call["id"])
                function = copied_call.get("function")
                if isinstance(function, dict):
                    function = dict(function)
                    arguments = function.get("arguments")
                    if isinstance(arguments, str):
                        function["arguments"] = _sanitize_string(arguments)
                    copied_call["function"] = function
                copied_calls.append(copied_call)
            current["tool_calls"] = copied_calls
        tool_call_id = current.get("tool_call_id")
        if isinstance(tool_call_id, str):
            current["tool_call_id"] = _sanitize_string(tool_call_id)
        cleaned.append(current)
    return cleaned


def _normalize_chat_roles(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for message in messages:
        current = dict(message)
        if current.get("role") == "developer":
            current["role"] = "system"
        normalized.append(current)
    return normalized


def _merge_consecutive_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for message in messages:
        current = dict(message)
        role = current.get("role")
        if merged and role == merged[-1].get("role") and role in {"system", "user", "assistant"}:
            previous = merged[-1]
            previous["content"] = _merge_chat_content(previous.get("content"), current.get("content"))
            if role == "assistant":
                if current.get("reasoning_content") and not previous.get("reasoning_content"):
                    previous["reasoning_content"] = current["reasoning_content"]
                tool_calls = list(previous.get("tool_calls") or []) + list(current.get("tool_calls") or [])
                if tool_calls:
                    previous["tool_calls"] = tool_calls
            continue
        merged.append(current)
    return merged


def _merge_chat_content(left: Any, right: Any) -> Any:
    if not left:
        return right or ""
    if not right:
        return left
    if isinstance(left, list) or isinstance(right, list):
        merged: list[Any] = []
        merged.extend(left if isinstance(left, list) else [{"type": "text", "text": str(left)}])
        if merged and right:
            merged.append({"type": "text", "text": ""})
        merged.extend(right if isinstance(right, list) else [{"type": "text", "text": str(right)}])
        return merged
    return f"{left}\n\n{right}"

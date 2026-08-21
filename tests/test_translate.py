from __future__ import annotations

from codex_shim.translate import (
    anthropic_messages_to_chat,
    anthropic_to_response,
    chat_completion_to_anthropic_message,
    chat_completion_to_response,
    responses_to_anthropic,
    responses_to_chat,
    responses_tool_resolve_map,
    upstream_chat_tool_name,
)


def test_responses_to_chat_text_input():
    body = {"model": "slug", "instructions": "System", "input": "Hello", "stream": True, "max_output_tokens": 99}
    out = responses_to_chat(body, "real-model")
    assert out["model"] == "real-model"
    assert out["stream"] is True
    assert out["max_tokens"] == 99
    assert out["messages"] == [{"role": "system", "content": "System"}, {"role": "user", "content": "Hello"}]


def test_anthropic_messages_to_chat_preserves_tools_images_and_tool_results():
    body = {
        "model": "slug",
        "system": [{"type": "text", "text": "System"}],
        "messages": [
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {"q": "repo"}}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": [
                            {"type": "text", "text": "result"},
                            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "BBB"}},
                        ],
                    },
                    {"type": "text", "text": "Next"},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAA"}},
                ],
            },
        ],
        "max_tokens": 99,
        "stop_sequences": ["END"],
        "stream": True,
        "tools": [{"name": "lookup", "description": "Lookup", "input_schema": {"type": "object"}}],
        "tool_choice": {"type": "tool", "name": "lookup"},
    }

    out = anthropic_messages_to_chat(body, "real-model")

    assert out["model"] == "real-model"
    assert out["max_tokens"] == 99
    assert out["stop"] == ["END"]
    assert out["stream_options"] == {"include_usage": True}
    assert out["tool_choice"] == {"type": "function", "function": {"name": "lookup"}}
    assert out["tools"] == [
        {
            "type": "function",
            "function": {"name": "lookup", "description": "Lookup", "parameters": {"type": "object"}},
        }
    ]
    assert out["messages"] == [
        {"role": "system", "content": "System"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "toolu_1", "type": "function", "function": {"name": "lookup", "arguments": "{\"q\":\"repo\"}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "toolu_1", "content": "result\n[image]"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Visual tool result for toolu_1."},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,BBB"}},
                {"type": "text", "text": "Next"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
            ],
        },
    ]


def test_chat_completion_to_anthropic_message_preserves_text_tools_and_usage():
    payload = {
        "id": "chatcmpl_fake",
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": "hello",
                    "tool_calls": [
                        {"id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": "{\"q\":\"repo\"}"}},
                        {"id": "call_2", "type": "function", "function": {"name": "broken", "arguments": "{"}},
                    ],
                },
            }
        ],
        "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
    }

    out = chat_completion_to_anthropic_message(payload, "shim-model")

    assert out["id"] == "chatcmpl_fake"
    assert out["type"] == "message"
    assert out["role"] == "assistant"
    assert out["model"] == "shim-model"
    assert out["stop_reason"] == "tool_use"
    assert out["usage"] == {"input_tokens": 4, "output_tokens": 2}
    assert out["content"] == [
        {"type": "text", "text": "hello"},
        {"type": "tool_use", "id": "call_1", "name": "lookup", "input": {"q": "repo"}},
        {"type": "tool_use", "id": "call_2", "name": "broken", "input": {"_raw": "{"}},
    ]



def test_chat_completion_to_anthropic_message_tool_only_uses_empty_content():
    payload = {
        "id": "chatcmpl_tool_only",
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": "{\"q\":\"repo\"}"}},
                    ],
                },
            }
        ],
        "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
    }

    out = chat_completion_to_anthropic_message(payload, "shim-model")

    assert out["content"] == [
        {"type": "tool_use", "id": "call_1", "name": "lookup", "input": {"q": "repo"}},
    ]
    assert out["stop_reason"] == "tool_use"


def test_chat_completion_to_anthropic_message_includes_reasoning():
    payload = {
        "id": "chatcmpl_reason",
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": "answer",
                    "reasoning_content": "thinking hard",
                },
            }
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    }

    out = chat_completion_to_anthropic_message(payload, "shim-model")

    assert out["content"] == [
        {"type": "thinking", "thinking": "thinking hard"},
        {"type": "text", "text": "answer"},
    ]

def test_responses_to_chat_preserves_reasoning_and_effort_for_deepseek():
    body = {
        "model": "slug",
        "reasoning_effort": "high",
        "input": [
            {"type": "reasoning", "summary": [{"type": "summary_text", "text": "prior thought"}]},
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "prior answer"}]},
            {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "rules"}]},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "next"}]},
        ],
    }

    out = responses_to_chat(body, "deepseek-reasoner")

    assert out["reasoning_effort"] == "high"
    assert out["messages"] == [
        {"role": "assistant", "content": "prior answer", "reasoning_content": "prior thought"},
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "next"},
    ]


def test_responses_to_chat_sanitizes_and_merges_strict_provider_messages():
    body = {
        "model": "slug",
        "instructions": "System\x00one",
        "input": [
            {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "rules\x00two"}]},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi\x00"}]},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "again\x01"}]},
            {"type": "function_call", "call_id": "call\x000", "name": "tool", "arguments": "{\"x\":\"y\x00\"}"},
        ],
    }

    out = responses_to_chat(body, "kimi-k2")

    assert out["messages"] == [
        {"role": "system", "content": "Systemone\n\nrulestwo"},
        {"role": "user", "content": "hi\n\nagain"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "call0", "type": "function", "function": {"name": "tool", "arguments": "{\"x\":\"y\"}"}}
            ],
        },
    ]


def test_responses_to_chat_synthesizes_unknown_tool_call_for_orphan_output():
    body = {
        "model": "slug",
        "input": [
            {
                "type": "function_call_output",
                "call_id": "call_orphan",
                "output": "result without preceding call",
            },
        ],
    }
    out = responses_to_chat(body, "real-model")
    assert out["messages"] == [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_orphan",
                    "type": "function",
                    "function": {"name": "unknown_tool", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_orphan", "content": "result without preceding call"},
    ]


def test_responses_function_tools_convert_to_chat_shape():
    body = {
        "model": "slug",
        "input": "Hi",
        "tools": [{"type": "function", "name": "do_work", "description": "Do work", "parameters": {"type": "object"}}],
    }
    out = responses_to_chat(body, "real-model")
    assert out["tools"] == [
        {
            "type": "function",
            "function": {"name": "do_work", "description": "Do work", "parameters": {"type": "object"}},
        }
    ]


def test_native_responses_tools_get_function_fallbacks_for_byok_chat():
    body = {
        "model": "slug",
        "input": "Use the browser",
        "tool_choice": {"type": "computer_use_preview"},
        "tools": [
            {"type": "computer_use_preview"},
            {"type": "web_search_preview"},
            {"type": "apply_patch"},
            {"type": "function", "name": "list_mcp_resources", "parameters": {"type": "object"}},
        ],
    }

    out = responses_to_chat(body, "real-model")

    functions = [tool["function"] for tool in out["tools"]]
    assert [fn["name"] for fn in functions] == ["computer_use", "web_search", "apply_patch", "list_mcp_resources"]
    assert functions[0]["parameters"]["required"] == ["action"]
    assert functions[1]["parameters"]["required"] == ["query"]
    assert functions[2]["parameters"]["required"] == ["patch"]
    assert out["tool_choice"] == {"type": "function", "function": {"name": "computer_use"}}


def test_native_responses_tools_get_anthropic_fallbacks():
    body = {
        "model": "slug",
        "input": "Search",
        "tools": [{"type": "web_search_preview"}, {"type": "computer_use_preview"}],
    }

    out = responses_to_anthropic(body, "claude-real", 123)

    assert [tool["name"] for tool in out["tools"]] == ["web_search", "computer_use"]
    assert out["tools"][0]["input_schema"]["required"] == ["query"]
    assert out["tools"][1]["input_schema"]["required"] == ["action"]


def test_responses_to_anthropic_messages():
    body = {"model": "slug", "input": [{"role": "user", "content": [{"type": "input_text", "text": "Hi"}]}]}
    out = responses_to_anthropic(body, "claude-real", 123)
    assert out["model"] == "claude-real"
    assert out["max_tokens"] == 123
    assert out["messages"] == [{"role": "user", "content": "Hi"}]


def test_responses_to_chat_preserves_input_images_for_vision_models():
    body = {
        "model": "slug",
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "What is visible?"},
                    {"type": "input_image", "image_url": "data:image/png;base64,AAA", "detail": "high"},
                ],
            }
        ],
    }

    out = responses_to_chat(body, "vision-model")

    assert out["messages"] == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is visible?"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA", "detail": "high"}},
            ],
        }
    ]


def test_responses_to_chat_normalises_original_image_detail():
    """Codex Desktop sends `detail: "original"` on input_image items, but
    "original" is not a valid OpenAI Chat Completions value. Providers like
    Kimi K2.6 (via Ark) reject it with:

    The parameter `messages.content.image_url.detail` specified in the
    request are not valid: invalid value: `original`, supported values
    are: `low`, `high`, `xhigh`, and `auto`.

    The shim must translate ``original`` to ``high`` (the closest standard
    OpenAI value — "full resolution") rather than passing it through verbatim.
    """
    body = {
        "model": "slug",
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Describe the screenshot"},
                    {"type": "input_image", "image_url": "data:image/png;base64,ZZZ", "detail": "original"},
                ],
            }
        ],
    }

    out = responses_to_chat(body, "vision-model")

    assert out["messages"] == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe the screenshot"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,ZZZ", "detail": "high"}},
            ],
        }
    ]


def test_computer_call_output_screenshot_reaches_openai_chat_vision():
    body = {
        "model": "slug",
        "input": [
            {"type": "computer_call_output", "call_id": "cu_1", "output": {"type": "input_image", "image_url": "data:image/png;base64,BBB"}}
        ],
    }

    out = responses_to_chat(body, "vision-model")

    assert out["messages"] == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Computer output for cu_1."},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,BBB"}},
            ],
        }
    ]


def test_input_image_detail_original_normalizes_to_openai_chat_high():
    body = {
        "model": "slug",
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Inspect."},
                    {"type": "input_image", "image_url": "data:image/png;base64,AAA", "detail": "original"},
                    {"type": "input_image", "image_url": "data:image/png;base64,BBB", "detail": "weird"},
                ],
            }
        ],
    }

    out = responses_to_chat(body, "vision-model")

    images = [
        part["image_url"]
        for part in out["messages"][0]["content"]
        if part.get("type") == "image_url"
    ]
    assert images == [
        {"url": "data:image/png;base64,AAA", "detail": "high"},
        {"url": "data:image/png;base64,BBB", "detail": "auto"},
    ]


def test_function_call_output_visual_feedback_adds_followup_image_message():
    body = {
        "model": "slug",
        "input": [
            {"type": "function_call", "call_id": "call_1", "name": "computer_use", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "call_1", "output": [{"type": "input_image", "image_url": "data:image/png;base64,CCC"}]},
        ],
    }

    out = responses_to_chat(body, "vision-model")

    assert out["messages"][1] == {"role": "tool", "tool_call_id": "call_1", "content": "[image]"}
    assert out["messages"][2] == {
        "role": "user",
        "content": [
            {"type": "text", "text": "Visual tool output for call_1."},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,CCC"}},
        ],
    }


def test_responses_to_anthropic_preserves_visual_feedback_as_image_blocks():
    body = {
        "model": "slug",
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Inspect this."},
                    {"type": "input_image", "image_url": "data:image/png;base64,DDD"},
                ],
            },
            {"type": "computer_call_output", "call_id": "cu_2", "output": {"type": "input_image", "image_url": "https://example.invalid/screen.png"}},
        ],
    }

    out = responses_to_anthropic(body, "claude-real", 123)

    assert out["messages"] == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Inspect this."},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "DDD"}},
                {"type": "text", "text": "Computer output for cu_2."},
                {"type": "image", "source": {"type": "url", "url": "https://example.invalid/screen.png"}},
            ],
        }
    ]


def test_chat_completion_to_response_strips_think():
    payload = {
        "id": "chatcmpl_1",
        "choices": [{"message": {"role": "assistant", "content": "<think>secret</think>Hello"}}],
    }
    out = chat_completion_to_response(payload, "slug")
    assert out["model"] == "slug"
    assert out["output"][0]["content"][0]["text"] == "Hello"


def test_chat_completion_to_response_normalizes_cached_usage():
    payload = {
        "id": "chatcmpl_1",
        "choices": [{"message": {"role": "assistant", "content": "Hello"}}],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "total_tokens": 12,
            "prompt_tokens_details": {"cached_tokens": 8},
            "completion_tokens_details": {"reasoning_tokens": 1},
        },
    }

    out = chat_completion_to_response(payload, "slug")

    assert out["usage"] == {
        "input_tokens": 10,
        "output_tokens": 2,
        "total_tokens": 12,
        "input_tokens_details": {"cached_tokens": 8},
        "output_tokens_details": {"reasoning_tokens": 1},
    }


def test_anthropic_to_response_normalizes_cache_usage():
    payload = {
        "id": "msg_1",
        "content": [{"type": "text", "text": "Hello"}],
        "usage": {
            "input_tokens": 10,
            "cache_read_input_tokens": 8,
            "cache_creation_input_tokens": 2,
            "output_tokens": 3,
        },
    }

    out = anthropic_to_response(payload, "slug")

    assert out["usage"] == {
        "input_tokens": 10,
        "output_tokens": 3,
        "total_tokens": 13,
        "input_tokens_details": {
            "cached_tokens": 8,
            "cache_read_input_tokens": 8,
            "cache_creation_input_tokens": 2,
        },
    }


def test_prepare_codex_byok_responses_body_strips_hosted_codex_tools_for_codex_client():
    from codex_shim.translate import prepare_codex_byok_responses_body

    body = {
        "model": "local-llama",
        "parallel_tool_calls": True,
        "tool_choice": {"type": "web_search_preview"},
        "tools": [
            {"type": "function", "name": "tool_search", "parameters": {"type": "object"}},
            {"type": "web_search_preview"},
            {"type": "image_generation"},
            {"type": "computer_use_preview"},
            {"type": "function", "name": "exec_command", "parameters": {"type": "object"}},
        ],
    }
    headers = {"x-codex-installation-id": "install-123"}

    out = prepare_codex_byok_responses_body(body, headers)

    assert out["parallel_tool_calls"] is True
    assert "tool_choice" not in out
    assert [tool.get("name") or tool.get("type") for tool in out["tools"]] == [
        "tool_search",
        "exec_command",
    ]


def test_prepare_codex_byok_responses_body_clears_tool_choice_for_image_and_computer_use():
    from codex_shim.translate import prepare_codex_byok_responses_body

    headers = {"User-Agent": "codex-cli/0.135.0"}
    for tool_choice in (
        {"type": "image_generation"},
        {"type": "computer_use_preview"},
    ):
        body = {
            "tools": [{"type": tool_choice["type"]}, {"type": "function", "name": "exec_command"}],
            "tool_choice": tool_choice,
        }
        out = prepare_codex_byok_responses_body(body, headers)
        assert "tool_choice" not in out
        assert len(out["tools"]) == 1
        assert out["tools"][0]["name"] == "exec_command"


def test_prepare_codex_byok_responses_body_keeps_web_search_for_non_codex_client():
    from codex_shim.translate import prepare_codex_byok_responses_body

    body = {
        "tools": [
            {"type": "web_search_preview"},
            {"type": "function", "name": "exec_command", "parameters": {"type": "object"}},
        ],
    }

    out = prepare_codex_byok_responses_body(body, {"User-Agent": "openai-python/1.0"})

    assert len(out["tools"]) == 2


def test_responses_to_chat_after_prepare_omits_web_search_for_byok():
    from codex_shim.translate import prepare_codex_byok_responses_body

    body = {
        "input": "Search the web",
        "parallel_tool_calls": True,
        "tools": [
            {"type": "web_search_preview"},
            {"type": "function", "name": "tool_search", "parameters": {"type": "object"}},
        ],
    }
    prepared = prepare_codex_byok_responses_body(body, {"User-Agent": "codex-cli/0.135.0"})
    out = responses_to_chat(prepared, "gemma-4")

    assert out["parallel_tool_calls"] is True
    assert [tool["function"]["name"] for tool in out["tools"]] == ["tool_search"]


def test_upstream_chat_tool_name_sanitizes_namespace_dots():
    assert upstream_chat_tool_name("codex_app", "load_workspace_dependencies") == (
        "codex_app_load_workspace_dependencies"
    )


def test_responses_to_chat_expands_namespace_tools_with_sanitized_names():
    body = {
        "model": "slug",
        "input": [{"role": "user", "content": "hi"}],
        "tools": [
            {
                "type": "namespace",
                "name": "multi_agent_v1",
                "description": "Multi-agent tools",
                "tools": [
                    {
                        "type": "function",
                        "name": "spawn_agent",
                        "description": "Spawn a sub-agent",
                        "parameters": {
                            "type": "object",
                            "properties": {"task": {"type": "string"}},
                        },
                    }
                ],
            }
        ],
    }
    out = responses_to_chat(body, "upstream")
    names = [t["function"]["name"] for t in out["tools"]]
    assert names == ["multi_agent_v1_spawn_agent"]


def test_function_call_round_trip_preserves_namespace_with_sanitized_names():
    from codex_shim.translate import function_call_item_from_chat_tool

    namespace_tools = [
        {
            "type": "namespace",
            "name": "multi_agent_v1",
            "description": "Multi-agent tools",
            "tools": [
                {
                    "type": "function",
                    "name": "spawn_agent",
                    "description": "Spawn a sub-agent",
                    "parameters": {"type": "object", "properties": {"task": {"type": "string"}}},
                }
            ],
        }
    ]
    body = {
        "tools": namespace_tools,
        "input": [
            {
                "type": "function_call",
                "call_id": "call_1",
                "namespace": "multi_agent_v1",
                "name": "spawn_agent",
                "arguments": '{"task":"review"}',
            }
        ],
    }
    chat = responses_to_chat(body, "upstream")
    assert chat["messages"][0]["tool_calls"][0]["function"]["name"] == "multi_agent_v1_spawn_agent"

    tool_resolve = responses_tool_resolve_map(namespace_tools)
    completion = {
        "id": "chatcmpl-1",
        "choices": [
            {
                "message": {
                    "tool_calls": chat["messages"][0]["tool_calls"],
                }
            }
        ],
    }
    response = chat_completion_to_response(completion, "slug", tool_resolve=tool_resolve)
    item = response["output"][0]
    assert item["namespace"] == "multi_agent_v1"
    assert item["name"] == "spawn_agent"

    direct = function_call_item_from_chat_tool(
        {
            "id": "call_2",
            "function": {
                "name": "mcp__exa.web_search_exa",
                "arguments": '{"query":"hi"}',
            },
        }
    )
    assert direct["namespace"] == "mcp__exa"
    assert direct["name"] == "web_search_exa"
    assert direct["id"] == "fc_2"
    assert direct["call_id"] == "call_2"


def test_web_search_call_empty_results_emits_unavailable_tool_message():
    body = {
        "model": "slug",
        "input": [
            {
                "type": "web_search_call",
                "id": "ws_empty",
                "call_id": "ws_empty",
                "status": "completed",
                "action": {"type": "search", "query": "linux caffeinate"},
            }
        ],
    }
    out = responses_to_chat(body, "upstream")
    assert out["messages"] == [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "ws_empty",
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "arguments": '{"query": "linux caffeinate"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "ws_empty",
            "content": (
                "The Codex hosted web_search tool returned no results and is unavailable at this time. "
                "Other search tools may still be available (for example MCP web_search_exa). "
                "Do not retry web_search; use an alternative search tool instead."
            ),
        },
    ]


def test_web_search_call_with_sources_preserves_results():
    body = {
        "model": "slug",
        "input": [
            {
                "type": "web_search_call",
                "id": "ws_hit",
                "call_id": "ws_hit",
                "status": "completed",
                "action": {
                    "type": "search",
                    "query": "linux caffeinate",
                    "sources": [
                        {"type": "url", "url": "https://example.com/caffeine"},
                    ],
                },
            }
        ],
    }
    out = responses_to_chat(body, "upstream")
    tool_message = out["messages"][-1]
    assert tool_message["role"] == "tool"
    assert "https://example.com/caffeine" in tool_message["content"]
    assert "unavailable" not in tool_message["content"].lower()


def test_web_search_function_call_output_empty_substitutes_unavailable_message():
    body = {
        "model": "slug",
        "input": [
            {
                "type": "function_call",
                "name": "web_search",
                "call_id": "call_ws",
                "arguments": '{"query":"linux caffeinate"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_ws",
                "output": "",
            },
        ],
    }
    out = responses_to_chat(body, "upstream")
    tool_message = out["messages"][-1]
    assert tool_message["role"] == "tool"
    assert tool_message["tool_call_id"] == "call_ws"
    assert "unavailable at this time" in tool_message["content"]


def test_empty_function_call_output_for_mcp_search_is_not_substituted():
    body = {
        "model": "slug",
        "input": [
            {
                "type": "function_call",
                "name": "web_search_exa",
                "namespace": "mcp__exa",
                "call_id": "call_exa",
                "arguments": '{"query":"linux caffeinate"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_exa",
                "output": "",
            },
        ],
    }
    out = responses_to_chat(body, "upstream")
    tool_message = out["messages"][-1]
    assert tool_message["content"] == ""


def _spawn_agent_message() -> dict:
    """Real shape captured from a Codex rollout when the parent spawns a sub-agent."""
    return {
        "type": "agent_message",
        "author": "/root",
        "recipient": "/root/smoke_timeline_fix",
        "content": [
            {
                "type": "input_text",
                "text": "Message Type: NEW_TASK\nTask name: /root/smoke_timeline_fix\nSender: /root\nPayload:\n",
            },
            {
                "type": "encrypted_content",
                "encrypted_content": "Fix App smoke session-timeline failures.\nRepo: /home/henry/x\nWrite results to smoke-fix-v2.md",
            },
        ],
        "internal_chat_message_metadata_passthrough": {"turn_id": "t1"},
    }


def test_agent_message_delivers_sub_agent_task_payload():
    """Sub-agents got no task at all while `agent_message` was dropped in translation."""
    body = {
        "model": "slug",
        "input": [
            {"role": "developer", "content": "You are a sub-agent."},
            _spawn_agent_message(),
        ],
    }
    out = responses_to_chat(body, "real-model")
    joined = "\n".join(str(message.get("content") or "") for message in out["messages"])
    assert "Fix App smoke session-timeline failures." in joined
    assert "Write results to smoke-fix-v2.md" in joined
    assert "/root/smoke_timeline_fix" in joined
    incoming = [m for m in out["messages"] if "agent message from /root" in str(m.get("content") or "")]
    assert incoming and incoming[0]["role"] == "user"


def test_agent_message_without_routing_is_the_agents_own_message():
    body = {
        "model": "slug",
        "input": [
            {"role": "user", "content": "go"},
            {
                "type": "agent_message",
                "message": "Batching bridge invokes, then waiting for results.",
                "phase": None,
                "memory_citation": None,
            },
        ],
    }
    out = responses_to_chat(body, "real-model")
    assert any(
        m.get("role") == "assistant" and "Batching bridge invokes" in str(m.get("content") or "")
        for m in out["messages"]
    )


def test_agent_message_skips_opaque_encrypted_blobs():
    body = {
        "model": "slug",
        "input": [
            {
                "type": "agent_message",
                "author": "/root",
                "recipient": "/root/child",
                "content": [
                    {"type": "input_text", "text": "Message Type: NEW_TASK"},
                    {"type": "encrypted_content", "encrypted_content": "A" * 400},
                ],
            }
        ],
    }
    out = responses_to_chat(body, "real-model")
    joined = "\n".join(str(message.get("content") or "") for message in out["messages"])
    assert "Message Type: NEW_TASK" in joined
    assert "A" * 400 not in joined

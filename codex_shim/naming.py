from __future__ import annotations

import re

_TOKEN_OVERRIDES = {
    "ai": "AI",
    "api": "API",
    "codex": "Codex",
    "gpt": "GPT",
    "llm": "LLM",
    "mcp": "MCP",
    "mimo": "MiMo",
    "minimax": "MiniMax",
    "nemotron": "Nemotron",
    "kimi": "Kimi",
    "deepseek": "DeepSeek",
    "qwen": "Qwen",
    "llama": "Llama",
    "gemma": "Gemma",
    "gemini": "Gemini",
    "claude": "Claude",
    "sonnet": "Sonnet",
    "opus": "Opus",
    "haiku": "Haiku",
    "composer": "Composer",
    "nvidia": "NVIDIA",
    "openrouter": "OpenRouter",
    "opencode": "OpenCode",
    "zen": "Zen",
    "oc-free": "OpenCode Zen (free)",
    "or": "OpenRouter",
    "pro": "Pro",
    "max": "Max",
    "mini": "Mini",
    "nano": "Nano",
    "flash": "Flash",
    "free": "(free)",
    "ultra": "Ultra",
    "super": "Super",
    "high": "High",
    "low": "Low",
    "fast": "Fast",
    "thinking": "Thinking",
    "xhigh": "Extra High",
}

_VERSION_RE = re.compile(r"^v(\d+(?:\.\d+)*)$", re.IGNORECASE)
_SIZE_RE = re.compile(r"^(\d+)(b|k|m|g)$", re.IGNORECASE)


def _format_token(token: str) -> str:
    lower = token.lower()
    if lower in _TOKEN_OVERRIDES:
        return _TOKEN_OVERRIDES[lower]
    if re.fullmatch(r"[a-z]\d+(?:\.\d+)?", token, flags=re.IGNORECASE):
        return token[0].upper() + token[1:]
    version = _VERSION_RE.match(token)
    if version:
        return f"V{version.group(1)}"
    size = _SIZE_RE.match(token)
    if size:
        suffix = size.group(2).upper()
        return f"{size.group(1)}{suffix}"
    if token.isupper() and len(token) <= 4:
        return token
    if token.isdigit():
        return token
    if "." in token and any(ch.isdigit() for ch in token):
        return token
    return token[:1].upper() + token[1:] if token else token


def _normalize_slug_body(body: str) -> str:
    normalized = body
    normalized = re.sub(r"([a-z])(\d+)-(\d+)", r"\1\2.\3", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"(\d)-(\d)", r"\1.\2", normalized)
    return normalized


_CATALOG_ROUTE_PREFIXES = ("cursor-", "codex-", "oc-free-")


def display_name_from_slug(slug: str, *, label_prefix: str | None = None) -> str:
    """Turn a catalog slug into a human-readable display name."""
    raw = slug.strip()
    if not raw:
        return "Model"
    body = raw
    if label_prefix and body.startswith(f"{label_prefix}-"):
        body = body[len(label_prefix) + 1 :]
    elif not label_prefix:
        for prefix in _CATALOG_ROUTE_PREFIXES:
            if body.startswith(prefix):
                body = body[len(prefix) :]
                break
    body = _normalize_slug_body(body)
    parts = re.split(r"[-_/]+", body)
    rendered = " ".join(_format_token(part) for part in parts if part)
    if label_prefix:
        prefix_display = _TOKEN_OVERRIDES.get(label_prefix.lower(), label_prefix.title())
        return f"{prefix_display} — {rendered}"
    return rendered


def description_for_route(display_name: str, route: str) -> str:
    return f"{display_name} {route}."

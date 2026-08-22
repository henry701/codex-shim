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
    "nous": "Nous",
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

_ROUTE_SLUG_PREFIXES = ("oc-free", "zen", "or", "nvidia", "nous", "local", "ocgo", "cursor", "codex")

ROUTE_LABEL_PREFIXES = {
    "zen": "OpenCode Zen",
    "oc-free": "OpenCode Zen (free)",
    "or": "OpenRouter",
    "nvidia": "NVIDIA",
    "nous": "Nous Portal",
    "local": "Local",
    "ocgo": "OpenCode Go",
}

CURSOR_DISPLAY_PREFIX = "Cursor - "
OPENCODE_ZEN_BASE_URL = "https://opencode.ai/zen/v1"


def _strip_route_slug_prefix(slug_body: str) -> str:
    for prefix in _ROUTE_SLUG_PREFIXES:
        if slug_body.startswith(f"{prefix}-"):
            return slug_body[len(prefix) + 1 :]
    return slug_body


def format_cursor_display_name(name: str) -> str:
    normalized = name.strip()
    if not normalized:
        return CURSOR_DISPLAY_PREFIX.strip()
    for prefix in (CURSOR_DISPLAY_PREFIX, "Cursor — ", "Cursor - "):
        if normalized.startswith(prefix):
            return normalized
    return f"{CURSOR_DISPLAY_PREFIX}{normalized}"


def catalog_display_name(model) -> str:
    """Normalize picker labels for routed shim models."""
    base_url = str(getattr(model, "base_url", "") or "").rstrip("/")
    if base_url.endswith("opencode.ai/zen/v1"):
        from .discover import is_zen_public_model

        slug = str(getattr(model, "slug", "") or "")
        model_id = str(getattr(model, "model", "") or "")
        label_prefix = (
            "oc-free"
            if slug.startswith("oc-free-") or slug.endswith("-free") or is_zen_public_model(model_id)
            else "zen"
        )
        return display_name_from_slug(slug, label_prefix=label_prefix)
    return str(getattr(model, "display_name", "") or getattr(model, "slug", "") or "Model")


def display_name_from_slug(slug: str, *, label_prefix: str | None = None) -> str:
    """Turn a catalog slug into a human-readable display name."""
    raw = slug.strip()
    if not raw:
        return "Model"
    body = raw
    if label_prefix:
        body = _strip_route_slug_prefix(body)
    elif not label_prefix:
        for prefix in _CATALOG_ROUTE_PREFIXES:
            if body.startswith(prefix):
                body = body[len(prefix) :]
                break
    body = _normalize_slug_body(body)
    parts = re.split(r"[-_/]+", body)
    rendered = " ".join(_format_token(part) for part in parts if part)
    if label_prefix:
        prefix_display = ROUTE_LABEL_PREFIXES.get(
            label_prefix.lower(),
            _TOKEN_OVERRIDES.get(label_prefix.lower(), label_prefix.title()),
        )
        return f"{prefix_display} — {rendered}"
    return rendered


def description_for_route(display_name: str, route: str) -> str:
    return f"{display_name} {route}."

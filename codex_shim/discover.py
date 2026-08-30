from __future__ import annotations

from dataclasses import dataclass, replace
import ipaddress
import json
import logging
import os
import re
import shutil
import subprocess
import threading
import time
from typing import Any
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import urlopen
from pathlib import Path

from .header_passthrough import KEYLESS_API_KEY
from .naming import ROUTE_LABEL_PREFIXES, description_for_route, display_name_from_slug
from .net.retry import RetryPolicy, request_urllib
from .nous_auth import resolve_nous_api_key
from .settings import ShimModel, slugify

logger = logging.getLogger(__name__)

ZEN_MODELS_URL = "https://opencode.ai/zen/v1/models"
NOUS_INFERENCE_MODELS_URL = "https://inference-api.nousresearch.com/v1/models"
NOUS_FALLBACK_MODEL_IDS = ("stealth/ox-alpha",)
MODELS_DEV_API_URL = "https://models.dev/api.json"
MODELS_DEV_OPENCODE_PROVIDER = "opencode"
OPENAI_MODELS_URL = "https://api.openai.com/v1/models"
CHATGPT_CODEX_MODELS_URL = "https://chatgpt.com/backend-api/codex/models"
DEFAULT_CHATGPT_CODEX_CLIENT_VERSION = "0.144.1"
CODEX_OAUTH_TOKEN_URL = "https://auth.openai.com/oauth/token"
CODEX_OAUTH_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
DISCOVER_INDEX_BASE = 10_000

_NVIDIA_SKIP_RE = re.compile(
    r"(embed|embedding|rerank|bge-|flux|dracarys|image|schnell|kontext|esm|paligemma|vision)",
    re.IGNORECASE,
)
_OPENROUTER_FREE_SUFFIX = ":free"
_OPENROUTER_FREE_ROUTER = "openrouter/free"
_CHATGPT_MODEL_RE = re.compile(r"^(gpt-|codex-|o\d)", re.IGNORECASE)
_ZEN_PUBLIC_IDS = frozenset({"big-pickle"})


@dataclass(frozen=True)
class DiscoverTemplate:
    kind: str
    base_url: str
    provider: str
    slug_prefix: str
    api_key: str
    extra_headers: dict[str, str]
    label_prefix: str | None = None


@dataclass(frozen=True)
class LocalModelRecord:
    model_id: str
    max_context_limit: int | None = None


ZEN_PUBLIC_TEMPLATE = DiscoverTemplate(
    kind="zen_public",
    base_url="https://opencode.ai/zen/v1",
    provider="generic-chat-completion-api",
    slug_prefix="oc-free",
    api_key=KEYLESS_API_KEY,
    extra_headers={
        "HTTP-Referer": "https://hermes-agent.nousresearch.com",
        "X-Title": "Hermes Agent",
    },
    label_prefix="oc-free",
)

ZEN_PAID_TEMPLATE = DiscoverTemplate(
    kind="zen",
    base_url="https://opencode.ai/zen/v1",
    provider="generic-chat-completion-api",
    slug_prefix="zen",
    api_key="${OPENCODE_API_KEY}",
    extra_headers={},
    label_prefix="zen",
)

OPENROUTER_FREE_TEMPLATE = DiscoverTemplate(
    kind="openrouter_free",
    base_url="https://openrouter.ai/api/v1",
    provider="generic-chat-completion-api",
    slug_prefix="or",
    api_key="${OPENROUTER_API_KEY}",
    extra_headers={
        "HTTP-Referer": "https://opencode.ai/",
        "X-Title": "codex-shim",
    },
    label_prefix="or",
)

NVIDIA_INTEGRATE_TEMPLATE = DiscoverTemplate(
    kind="nvidia_integrate",
    base_url="https://integrate.api.nvidia.com/v1",
    provider="generic-chat-completion-api",
    slug_prefix="nvidia",
    api_key="${NVIDIA_API_KEY}",
    extra_headers={
        "HTTP-Referer": "https://opencode.ai/",
        "X-Title": "codex-shim",
        "X-BILLING-INVOKE-ORIGIN": "OpenCode",
    },
    label_prefix="nvidia",
)

NOUS_PORTAL_TEMPLATE = DiscoverTemplate(
    kind="nous",
    base_url="https://inference-api.nousresearch.com/v1",
    provider="generic-chat-completion-api",
    slug_prefix="nous",
    api_key="${NOUS_API_KEY}",
    extra_headers={
        "HTTP-Referer": "https://hermes-agent.nousresearch.com",
        "X-Title": "Hermes Agent",
    },
    label_prefix="nous",
)

# Published Ox Alpha window is 1_048_576. Nous /v1/models often omits
# context_length, so discovery used to fall through to the 128k catalog
# default and Desktop auto-compacted around 102k (UI then shows ~122k).
OX_ALPHA_CONTEXT_TOKENS = 1_048_576
_OX_ALPHA_MODEL_MARKERS = (
    "ox-alpha",
    "ox_alpha",
    "x-preview-f-free",
)


def context_limit_for_discovered_model(model_id: str, reported: int | None = None) -> int | None:
    """Prefer a real listing window; never trust tiny stand-ins for Ox Alpha."""
    known = None
    haystack = model_id.strip().lower().replace("_", "-")
    if any(marker in haystack for marker in _OX_ALPHA_MODEL_MARKERS):
        known = OX_ALPHA_CONTEXT_TOKENS
    if known is not None:
        if reported is None or reported < 500_000:
            return known
        return max(known, reported)
    if reported is not None and reported > 0:
        return reported
    return None


_MODELS_DEV_PROVIDERS_BY_KIND = {
    "zen_public": ("opencode",),
    "zen": ("opencode",),
    "openrouter_free": ("openrouter",),
    "openrouter": ("openrouter",),
    "nvidia_integrate": ("nvidia",),
    "nvidia": ("nvidia",),
    "nous": ("openrouter",),
    "nous_portal": ("openrouter",),
}
_EFFORT_ALIASES = {
    "x-high": "xhigh",
    "x_high": "xhigh",
    "xhigh": "xhigh",
    "extra-high": "xhigh",
    "maximum": "max",
    "min": "minimal",
}
_MODELS_DEV_CATALOG_TTL_SEC = 3 * 60 * 60
_models_dev_catalog_cache: tuple[float, dict[str, Any]] | None = None
_models_dev_catalog_lock = threading.Lock()


def clear_models_dev_catalog_cache() -> None:
    global _models_dev_catalog_cache
    with _models_dev_catalog_lock:
        _models_dev_catalog_cache = None


def expire_models_dev_catalog_cache() -> None:
    """Mark the models.dev cache stale while keeping the last successful payload."""
    global _models_dev_catalog_cache
    with _models_dev_catalog_lock:
        cached = _models_dev_catalog_cache
        if cached is not None:
            _models_dev_catalog_cache = (0.0, cached[1])


def fetch_models_dev_catalog() -> dict[str, Any]:
    """Return the models.dev provider catalog, cached for a few hours."""
    global _models_dev_catalog_cache
    now = time.monotonic()
    with _models_dev_catalog_lock:
        cached = _models_dev_catalog_cache
        if cached is not None and now - cached[0] < _MODELS_DEV_CATALOG_TTL_SEC:
            return cached[1]
        stale = cached[1] if cached is not None else {}
    try:
        payload = fetch_http_json(MODELS_DEV_API_URL)
    except (OSError, URLError, json.JSONDecodeError, ValueError):
        return stale
    if not isinstance(payload, dict) or not payload:
        return stale
    with _models_dev_catalog_lock:
        _models_dev_catalog_cache = (time.monotonic(), payload)
    return payload


def _positive_int(value: Any) -> int | None:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _unique_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    items: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            items.append(value)
    return items


def _normalize_effort(value: Any) -> str:
    raw = str(value or "").strip().lower().replace("_", "-")
    if not raw:
        return ""
    if raw in _EFFORT_ALIASES:
        return _EFFORT_ALIASES[raw]
    return raw


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    items: list[str] = []
    for item in value:
        text = str(item or "").strip().lower()
        if text:
            items.append(text)
    return _unique_strings(items)


def _efforts_from_models_dev_row(row: dict[str, Any]) -> list[str]:
    efforts: list[str] = []
    options = row.get("reasoning_options")
    if isinstance(options, list):
        for option in options:
            if not isinstance(option, dict):
                continue
            kind = str(option.get("type") or "").strip().lower()
            if kind not in {"effort", "reasoning_effort", "reasoning"}:
                continue
            values = option.get("values") or option.get("options") or []
            if isinstance(values, list):
                efforts.extend(_normalize_effort(item) for item in values)
    variants = row.get("variants")
    if isinstance(variants, dict):
        for key, value in variants.items():
            if isinstance(value, dict):
                nested = value.get("effort") or value.get("reasoning_effort")
                if isinstance(nested, list):
                    efforts.extend(_normalize_effort(item) for item in nested)
                elif nested:
                    efforts.append(_normalize_effort(nested))
            elif isinstance(value, str):
                efforts.append(_normalize_effort(value))
    for key in ("reasoning_efforts", "supported_reasoning_efforts"):
        raw = row.get(key)
        if isinstance(raw, list):
            efforts.extend(_normalize_effort(item) for item in raw)
    return _unique_strings([item for item in efforts if item])


def metadata_from_models_dev_row(row: dict[str, Any] | None) -> dict[str, Any]:
    """Map a models.dev model row onto catalog-facing metadata."""
    if not isinstance(row, dict) or not row:
        return {}
    limit = row.get("limit") if isinstance(row.get("limit"), dict) else {}
    modalities = row.get("modalities") if isinstance(row.get("modalities"), dict) else {}
    interleaved = row.get("interleaved") if isinstance(row.get("interleaved"), dict) else {}
    input_modalities = _string_list(modalities.get("input"))
    efforts = _efforts_from_models_dev_row(row)
    reasoning = row.get("reasoning")
    attachment = row.get("attachment")
    name = str(row.get("name") or "").strip()
    description = str(row.get("description") or "").strip()
    no_image = False
    if input_modalities:
        no_image = "image" not in input_modalities
    elif attachment is False:
        no_image = True
    meta: dict[str, Any] = {}
    if name:
        meta["upstream_name"] = name
    if description:
        meta["upstream_description"] = description
    if efforts:
        meta["reasoning_efforts"] = efforts
    if input_modalities:
        meta["input_modalities"] = input_modalities
    reported = _positive_int(limit.get("context"))
    output = _positive_int(limit.get("output"))
    if reported:
        meta["reported_context_limit"] = reported
    if output:
        meta["output_limit"] = output
    if isinstance(reasoning, bool):
        meta["reasoning"] = reasoning
    if isinstance(attachment, bool):
        meta["attachment"] = attachment
    field = str(interleaved.get("field") or "").strip()
    if field:
        meta["interleaved_reasoning_field"] = field
    meta["no_image_support"] = no_image
    return meta


def lookup_models_dev_row(
    catalog: dict[str, Any],
    *,
    providers: tuple[str, ...],
    model_id: str,
) -> dict[str, Any] | None:
    needle = model_id.strip()
    if not needle or not isinstance(catalog, dict):
        return None
    for provider_name in providers:
        provider = catalog.get(provider_name)
        if not isinstance(provider, dict):
            continue
        models = provider.get("models")
        if not isinstance(models, dict):
            continue
        exact = models.get(needle)
        if isinstance(exact, dict):
            return exact
        for key, row in models.items():
            if not isinstance(row, dict):
                continue
            row_id = str(row.get("id") or key).strip()
            if _model_id_matches(needle, key, row_id):
                return row
    return None


def _model_id_matches(needle: str, *candidates: str) -> bool:
    lowered = needle.lower()
    aliases = {lowered, lowered.removesuffix(":free")}
    needle_tail = lowered.split("/")[-1]
    for candidate in candidates:
        value = str(candidate or "").strip().lower()
        if not value:
            continue
        if value in aliases or value.removesuffix(":free") in aliases:
            return True
        if value.split("/")[-1] == needle_tail:
            return True
        if value.endswith("/" + lowered) or lowered.endswith("/" + value):
            return True
    return False


def _metadata_for_discovered_model(template: DiscoverTemplate, model_id: str) -> dict[str, Any]:
    providers = _MODELS_DEV_PROVIDERS_BY_KIND.get(template.kind, ())
    if not providers:
        return {}
    row = lookup_models_dev_row(
        fetch_models_dev_catalog(),
        providers=providers,
        model_id=model_id,
    )
    return metadata_from_models_dev_row(row)


_BUILTIN_TEMPLATES: dict[str, DiscoverTemplate] = {
    "zen_public": ZEN_PUBLIC_TEMPLATE,
    "zen": ZEN_PAID_TEMPLATE,
    "openrouter_free": OPENROUTER_FREE_TEMPLATE,
    "nvidia_integrate": NVIDIA_INTEGRATE_TEMPLATE,
    "nous": NOUS_PORTAL_TEMPLATE,
}

_BUILTIN_TEMPLATE_ORDER = ("zen", "zen_public", "openrouter_free", "nvidia_integrate", "nous")


def discover_enabled(settings_data: dict[str, Any] | None, kind: str, *, has_template: bool) -> bool:
    if not has_template and kind not in _BUILTIN_TEMPLATES:
        return False
    if not isinstance(settings_data, dict):
        return True
    discover = settings_data.get("discover")
    if discover is None:
        return True
    if isinstance(discover, bool):
        return discover
    if isinstance(discover, dict):
        aliases = {
            "zen_public": ("zen_public", "zen"),
            "zen": ("zen", "zen_paid"),
            "openrouter_free": ("openrouter_free", "openrouter"),
            "nvidia_integrate": ("nvidia_integrate", "nvidia"),
            "nous": ("nous", "nous_portal", "nousresearch"),
        }
        keys = aliases.get(kind, (kind,))
        value = next((discover.get(key) for key in keys if key in discover), None)
        if value is None:
            return True
        return bool(value)
    return True


def is_zen_public_model(model_id: str) -> bool:
    lower = model_id.lower()
    return lower in _ZEN_PUBLIC_IDS or lower.endswith("-free")


def is_local_base_url(base_url: str) -> bool:
    host = (urlparse(base_url).hostname or "").strip().lower()
    if not host:
        return False
    if host in {"localhost", "127.0.0.1", "::1"}:
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    return ip.is_private or ip.is_loopback


def infer_discover_templates(models: list[ShimModel]) -> list[DiscoverTemplate]:
    _ = models
    return []


def discover_byok_models(
    explicit_models: list[ShimModel],
    *,
    settings_data: dict[str, Any] | None = None,
) -> list[ShimModel]:
    """Return explicit models plus auto-discovered entries from provider listings."""
    explicit_models = refresh_local_explicit_models(explicit_models, settings_data=settings_data)
    templates: list[DiscoverTemplate] = []
    for kind in _BUILTIN_TEMPLATE_ORDER:
        builtin = _BUILTIN_TEMPLATES.get(kind)
        if builtin is None:
            continue
        if discover_enabled(settings_data, kind, has_template=True):
            templates.append(_enrich_builtin_template(builtin, explicit_models))
    templates.extend(infer_discover_templates(explicit_models))

    discovered: list[ShimModel] = []
    for template in templates:
        if not discover_enabled(settings_data, template.kind, has_template=True):
            continue
        if not _template_has_credentials(template):
            continue
        rows = _discover_rows_for_template(template)
        discovered.extend(_rows_to_shim_models(rows, template))
    return merge_discovered_models(explicit_models, discovered)


def refresh_local_explicit_models(
    explicit_models: list[ShimModel],
    *,
    settings_data: dict[str, Any] | None = None,
) -> list[ShimModel]:
    refreshed: list[ShimModel] = []
    for model in explicit_models:
        if not _should_refresh_local_model(model, settings_data):
            refreshed.append(model)
            continue
        records = fetch_local_openai_models(model.base_url, model.api_key)
        if not records:
            refreshed.append(model)
            continue
        if len(records) == 1:
            record = records[0]
            refreshed.append(
                replace(
                    model,
                    model=record.model_id,
                    display_name=_local_display_name(record.model_id),
                    max_context_limit=record.max_context_limit or model.max_context_limit,
                    raw={**model.raw, "discovered_local": True},
                )
            )
            continue
        slug_prefix = model.slug or "local"
        for offset, record in enumerate(records):
            slug = slug_prefix if offset == 0 else f"{slug_prefix}-{slugify(record.model_id)}"
            refreshed.append(
                ShimModel(
                    slug=slug,
                    model=record.model_id,
                    display_name=_local_display_name(record.model_id),
                    provider=model.provider,
                    base_url=model.base_url,
                    api_key=model.api_key,
                    index=model.index + offset,
                    max_context_limit=record.max_context_limit or model.max_context_limit,
                    max_output_tokens=model.max_output_tokens,
                    no_image_support=model.no_image_support,
                    supports_reasoning_summaries=model.supports_reasoning_summaries,
                    extra_headers=dict(model.extra_headers),
                    raw={**model.raw, "discovered_local": True},
                )
            )
    return refreshed


def _should_refresh_local_model(model: ShimModel, settings_data: dict[str, Any] | None) -> bool:
    row = model.raw if isinstance(model.raw, dict) else {}
    discover_flag = row.get("discover")
    if discover_flag is False:
        return False
    if discover_flag in {True, "local", "true"}:
        return True
    if discover_enabled(settings_data, "local", has_template=True) and is_local_base_url(model.base_url):
        return True
    return False


def _local_display_name(model_id: str) -> str:
    body = model_id.removesuffix(".gguf")
    return display_name_from_slug(slugify(body), label_prefix="local")


def merge_discovered_models(explicit: list[ShimModel], discovered: list[ShimModel]) -> list[ShimModel]:
    explicit_models = {model.model for model in explicit}
    explicit_slugs = {model.slug for model in explicit}
    merged = list(explicit)
    next_index = max((model.index for model in explicit), default=-1) + 1
    for model in discovered:
        if model.model in explicit_models or model.slug in explicit_slugs:
            continue
        merged.append(
            ShimModel(
                slug=model.slug,
                model=model.model,
                display_name=model.display_name,
                provider=model.provider,
                base_url=model.base_url,
                api_key=model.api_key,
                index=next_index,
                max_context_limit=model.max_context_limit,
                max_output_tokens=model.max_output_tokens,
                no_image_support=model.no_image_support,
                supports_reasoning_summaries=model.supports_reasoning_summaries,
                extra_headers=model.extra_headers,
                raw={"discovered": True, **model.raw},
            )
        )
        explicit_models.add(model.model)
        explicit_slugs.add(model.slug)
        next_index += 1
    merged.sort(key=lambda model: model.slug.lower())
    return merged


def fetch_http_json(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    timeout: float = 20.0,
    retries: int = 1,
    backoff_base: float = 0.5,
    backoff_factor: float = 2.0,
) -> Any:
    """GET JSON with optional retries and exponential backoff (stdlib urllib)."""
    merged = {
        "User-Agent": "codex-shim/1.0 (+https://github.com/henry701/codex-shim)",
        "Accept": "application/json",
    }
    if headers:
        merged.update(headers)
    result = request_urllib(
        url,
        headers=merged,
        timeout=timeout,
        policy=RetryPolicy(
            attempts=max(1, int(retries)),
            backoff_base=backoff_base,
            backoff_factor=backoff_factor,
        ),
        urlopen_fn=urlopen,
        sleep_fn=time.sleep,
        label=f"GET {url}",
        ensure_json=True,
    )
    return json.loads(result.body.decode("utf-8"))


def _load_codex_auth_tokens(auth_path: Any | None = None) -> tuple[str, str] | None:
    from .settings import DEFAULT_CODEX_AUTH

    expanded = Path(auth_path if auth_path is not None else DEFAULT_CODEX_AUTH).expanduser()
    if not expanded.exists():
        return None
    try:
        data = json.loads(expanded.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    tokens = data.get("tokens") if isinstance(data, dict) else None
    if not isinstance(tokens, dict):
        return None
    access_token = str(tokens.get("access_token") or "").strip()
    if not access_token:
        return None
    account_id = str(tokens.get("account_id") or "").strip()
    return access_token, account_id


def refresh_codex_auth_tokens(auth_path: Any | None = None, *, timeout: float = 20.0) -> bool:
    """Refresh Codex OAuth tokens in ``~/.codex/auth.json``. Returns True on success."""
    from .settings import DEFAULT_CODEX_AUTH

    expanded = Path(auth_path if auth_path is not None else DEFAULT_CODEX_AUTH).expanduser()
    if not expanded.exists():
        return False
    try:
        data = json.loads(expanded.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(data, dict):
        return False
    tokens = data.get("tokens")
    if not isinstance(tokens, dict):
        return False
    refresh_token = str(tokens.get("refresh_token") or "").strip()
    if not refresh_token:
        logger.warning("Codex auth refresh skipped: no refresh_token in %s", expanded)
        return False
    body = urlencode(
        {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": CODEX_OAUTH_CLIENT_ID,
        }
    ).encode("utf-8")
    try:
        result = request_urllib(
            CODEX_OAUTH_TOKEN_URL,
            method="POST",
            headers={"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"},
            data=body,
            timeout=timeout,
            policy=RetryPolicy(attempts=1, retry_json_decode=False),
            urlopen_fn=urlopen,
            sleep_fn=time.sleep,
            label="codex-oauth-refresh",
        )
        payload = json.loads(result.body.decode("utf-8"))
    except (OSError, URLError, HTTPError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
        logger.warning("Codex auth refresh failed: %s", exc)
        return False
    if not isinstance(payload, dict):
        return False
    access_token = str(payload.get("access_token") or "").strip()
    if not access_token:
        logger.warning("Codex auth refresh returned no access_token")
        return False
    next_tokens = dict(tokens)
    next_tokens["access_token"] = access_token
    if payload.get("refresh_token"):
        next_tokens["refresh_token"] = str(payload["refresh_token"])
    if payload.get("id_token"):
        next_tokens["id_token"] = str(payload["id_token"])
    data["tokens"] = next_tokens
    data["last_refresh"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    try:
        expanded.write_text(json.dumps(data, indent=2) + "\n")
    except OSError as exc:
        logger.warning("Codex auth refresh write failed: %s", exc)
        return False
    logger.info("Refreshed Codex OAuth tokens in %s", expanded)
    return True


def persist_chatgpt_models_cache(
    models: list[dict[str, Any]],
    cache_path: Any | None = None,
    *,
    client_version: str | None = None,
) -> Path | None:
    """Write a successful ChatGPT Codex /models payload to ``models_cache.json``."""
    from .settings import DEFAULT_CODEX_MODELS_CACHE

    if not models:
        return None
    path = Path(cache_path if cache_path is not None else DEFAULT_CODEX_MODELS_CACHE).expanduser()
    version = (
        client_version
        or os.environ.get("CODEX_SHIM_CHATGPT_MODELS_CLIENT_VERSION")
        or DEFAULT_CHATGPT_CODEX_CLIENT_VERSION
    ).strip()
    existing: dict[str, Any] = {}
    if path.exists():
        try:
            loaded = json.loads(path.read_text())
            if isinstance(loaded, dict):
                existing = loaded
        except (OSError, json.JSONDecodeError):
            existing = {}
    payload = {
        "fetched_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "etag": existing.get("etag"),
        "client_version": version,
        "models": [dict(model) for model in models if isinstance(model, dict)],
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n")
    except OSError as exc:
        logger.warning("Failed to persist ChatGPT models cache to %s: %s", path, exc)
        return None
    return path


def _parse_chatgpt_codex_models_payload(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    rows = payload.get("models")
    if not isinstance(rows, list):
        return []
    models: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        slug = str(row.get("slug") or "").strip()
        if not slug:
            continue
        if row.get("visibility") == "hidden":
            continue
        lower = slug.lower()
        if not (lower.startswith("gpt-") or lower.startswith("codex-")):
            continue
        models.append(dict(row))
    return models


def _chatgpt_codex_models_headers(access_token: str, account_id: str, version: str) -> dict[str, str]:
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Accept": "application/json",
        "User-Agent": f"codex_cli_rs/{version}",
        "originator": "codex_cli_rs",
    }
    if account_id:
        headers["ChatGPT-Account-Id"] = account_id
    return headers


def fetch_chatgpt_codex_backend_models(
    *,
    client_version: str | None = None,
    auth_path: Any | None = None,
    timeout: float = 20.0,
    retries: int = 3,
    backoff_base: float = 0.5,
    backoff_factor: float = 2.0,
    refresh_on_unauthorized: bool = True,
) -> list[dict[str, Any]]:
    """List ChatGPT Codex models from chatgpt.com/backend-api/codex/models.

    Uses the Codex OAuth token in ``~/.codex/auth.json``. On HTTP 401, refreshes
    the OAuth token once and retries. Retries transient HTTP failures with
    exponential backoff. Returns ``[]`` when auth is missing or the request
    ultimately fails.
    """
    auth = _load_codex_auth_tokens(auth_path)
    if auth is None:
        return []
    access_token, account_id = auth
    version = (client_version or os.environ.get("CODEX_SHIM_CHATGPT_MODELS_CLIENT_VERSION") or DEFAULT_CHATGPT_CODEX_CLIENT_VERSION).strip()
    url = f"{CHATGPT_CODEX_MODELS_URL}?client_version={version}"
    headers = _chatgpt_codex_models_headers(access_token, account_id, version)
    try:
        payload = fetch_http_json(
            url,
            headers=headers,
            timeout=timeout,
            retries=retries,
            backoff_base=backoff_base,
            backoff_factor=backoff_factor,
        )
    except HTTPError as exc:
        if refresh_on_unauthorized and exc.code == 401 and refresh_codex_auth_tokens(auth_path, timeout=timeout):
            auth = _load_codex_auth_tokens(auth_path)
            if auth is None:
                return []
            access_token, account_id = auth
            headers = _chatgpt_codex_models_headers(access_token, account_id, version)
            try:
                payload = fetch_http_json(
                    url,
                    headers=headers,
                    timeout=timeout,
                    retries=retries,
                    backoff_base=backoff_base,
                    backoff_factor=backoff_factor,
                )
            except (OSError, URLError, HTTPError, TimeoutError, json.JSONDecodeError, ValueError) as retry_exc:
                logger.warning("ChatGPT Codex /models fetch failed after auth refresh: %s", retry_exc)
                return []
        else:
            logger.warning("ChatGPT Codex /models fetch failed after retries: %s", exc)
            return []
    except (OSError, URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
        logger.warning("ChatGPT Codex /models fetch failed after retries: %s", exc)
        return []
    return _parse_chatgpt_codex_models_payload(payload)


def _parse_openai_model_list(payload: Any) -> list[str]:
    if not isinstance(payload, dict):
        return []
    rows = payload.get("data")
    if not isinstance(rows, list):
        return []
    ids: list[str] = []
    for row in rows:
        if isinstance(row, dict) and row.get("id"):
            ids.append(str(row["id"]).strip())
    return ids


def fetch_zen_model_ids(*, api_key: str = "") -> list[str]:
    headers: dict[str, str] = {}
    token = api_key.strip()
    if token and token != KEYLESS_API_KEY:
        headers["Authorization"] = f"Bearer {token}"
    try:
        return _parse_openai_model_list(fetch_http_json(ZEN_MODELS_URL, headers=headers or None))
    except (OSError, URLError, json.JSONDecodeError, ValueError):
        return discover_opencode_cli_ids("opencode")


def fetch_nous_portal_model_ids(*, api_key: str = "") -> list[str]:
    token = (api_key or "").strip() or resolve_nous_api_key()
    if not token:
        return []
    ids: list[str] = []
    try:
        # Cloudflare 1010s Python-urllib's default UA; HermesAgent is the
        # portal client identity. Inference still uses the Codex/Desktop UA.
        ids = _parse_openai_model_list(
            fetch_http_json(
                NOUS_INFERENCE_MODELS_URL,
                headers={
                    "Authorization": f"Bearer {token}",
                    "User-Agent": "HermesAgent/1.0",
                    "HTTP-Referer": "https://hermes-agent.nousresearch.com",
                    "X-Title": "Hermes Agent",
                },
            )
        )
    except (OSError, URLError, json.JSONDecodeError, ValueError, HTTPError):
        ids = []
    merged: list[str] = []
    seen: set[str] = set()
    for model_id in [*ids, *NOUS_FALLBACK_MODEL_IDS]:
        clean = str(model_id or "").strip()
        if not clean or clean in seen:
            continue
        seen.add(clean)
        merged.append(clean)
    return merged


def _models_dev_model_status(row: dict[str, Any]) -> str:
    status = row.get("status")
    if status in (None, ""):
        return "active"
    return str(status).strip().lower()


def _models_dev_model_cost_input(row: dict[str, Any]) -> float | None:
    cost = row.get("cost")
    if not isinstance(cost, dict):
        return None
    value = cost.get("input")
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_models_dev_opencode_free_ids(payload: Any) -> list[str]:
    if not isinstance(payload, dict):
        return []
    provider = payload.get(MODELS_DEV_OPENCODE_PROVIDER)
    if not isinstance(provider, dict):
        return []
    models = provider.get("models")
    if not isinstance(models, dict):
        return []
    ids: list[str] = []
    for model_key, row in models.items():
        if not isinstance(row, dict):
            continue
        if _models_dev_model_status(row) == "deprecated":
            continue
        if _models_dev_model_cost_input(row) != 0:
            continue
        model_id = str(row.get("id") or model_key).strip()
        if model_id:
            ids.append(model_id)
    return sorted(set(ids))


def fetch_models_dev_opencode_free_model_ids() -> list[str]:
    return _parse_models_dev_opencode_free_ids(fetch_models_dev_catalog())


def _parse_models_dev_opencode_paid_ids(payload: Any) -> list[str]:
    if not isinstance(payload, dict):
        return []
    provider = payload.get(MODELS_DEV_OPENCODE_PROVIDER)
    if not isinstance(provider, dict):
        return []
    models = provider.get("models")
    if not isinstance(models, dict):
        return []
    ids: list[str] = []
    for model_key, row in models.items():
        if not isinstance(row, dict):
            continue
        if _models_dev_model_status(row) == "deprecated":
            continue
        cost_input = _models_dev_model_cost_input(row)
        if cost_input in (None, 0):
            continue
        model_id = str(row.get("id") or model_key).strip()
        if model_id and not is_zen_public_model(model_id):
            ids.append(model_id)
    return sorted(set(ids))


def fetch_models_dev_opencode_paid_model_ids() -> list[str]:
    return _parse_models_dev_opencode_paid_ids(fetch_models_dev_catalog())


def fetch_zen_paid_model_ids(*, api_key: str = "") -> list[str]:
    token = api_key.strip()
    if token and token != KEYLESS_API_KEY:
        from_api = [
            model_id
            for model_id in fetch_zen_model_ids(api_key=token)
            if model_id and not is_zen_public_model(model_id)
        ]
        if from_api:
            return sorted(set(from_api))
    from_models_dev = fetch_models_dev_opencode_paid_model_ids()
    if from_models_dev:
        return from_models_dev
    return sorted(
        {
            model_id
            for model_id in discover_opencode_cli_ids("opencode")
            if model_id and not is_zen_public_model(model_id)
        }
    )


def fetch_zen_public_model_ids() -> list[str]:
    from_models_dev = fetch_models_dev_opencode_free_model_ids()
    if from_models_dev:
        return [model_id for model_id in from_models_dev if is_zen_public_model(model_id)]
    return sorted(
        model_id
        for model_id in discover_opencode_cli_ids("opencode")
        if is_zen_public_model(model_id)
    )


def is_openrouter_free_model(model_id: str) -> bool:
    lower = model_id.lower()
    return lower == _OPENROUTER_FREE_ROUTER or lower.endswith(_OPENROUTER_FREE_SUFFIX)


def fetch_openrouter_free_model_ids() -> list[str]:
    ids = discover_opencode_cli_ids("openrouter")
    free = [model_id for model_id in ids if is_openrouter_free_model(model_id)]
    if free:
        return sorted(set(free))
    return [_OPENROUTER_FREE_ROUTER]


def fetch_nvidia_integrate_model_ids() -> list[str]:
    ids = discover_opencode_cli_ids("nvidia")
    return sorted(
        {model_id for model_id in ids if model_id and not _NVIDIA_SKIP_RE.search(model_id)}
    )


def fetch_local_openai_models(base_url: str, api_key: str) -> list[LocalModelRecord]:
    url = f"{base_url.rstrip('/')}/models"
    headers: dict[str, str] = {}
    token = api_key.strip()
    if token and token not in {"local", "ollama"}:
        headers["Authorization"] = f"Bearer {token}"
    try:
        payload = fetch_http_json(url, headers=headers, timeout=5.0)
    except (OSError, URLError, json.JSONDecodeError, ValueError):
        return []
    if not isinstance(payload, dict):
        return []
    rows = payload.get("data")
    if not isinstance(rows, list):
        return []
    records: list[LocalModelRecord] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        model_id = str(row.get("id") or "").strip()
        if not model_id:
            continue
        records.append(LocalModelRecord(model_id=model_id, max_context_limit=_context_from_model_row(row)))
    return records


def discover_chatgpt_model_ids_from_openai_api() -> list[str]:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        return []
    try:
        ids = _parse_openai_model_list(
            fetch_http_json(OPENAI_MODELS_URL, headers={"Authorization": f"Bearer {api_key}"})
        )
    except (OSError, URLError, json.JSONDecodeError, ValueError):
        return []
    return [model_id for model_id in ids if _CHATGPT_MODEL_RE.match(model_id)]


def discover_chatgpt_models_from_cursor() -> list[tuple[str, str]]:
    from .cursor_passthrough import _cursor_agent_bin, _parse_cursor_list_models_output, cursor_spawn_env

    if not shutil.which(_cursor_agent_bin()) and not os.environ.get("CURSOR_AGENT_BIN"):
        return []
    try:
        result = subprocess.run(
            [_cursor_agent_bin(), "--list-models"],
            capture_output=True,
            text=True,
            timeout=30,
            env=cursor_spawn_env(),
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    output = f"{result.stdout}\n{result.stderr}"
    models = _parse_cursor_list_models_output(output)
    rows: list[tuple[str, str]] = []
    seen: set[str] = set()
    for model in models.values():
        upstream = model.upstream_id
        if not _CHATGPT_MODEL_RE.match(upstream):
            continue
        if upstream in seen:
            continue
        seen.add(upstream)
        rows.append((upstream, model.display_name))
    return sorted(rows, key=lambda item: item[0])


# `opencode models` is ~2s and openrouter/nvidia each call it on discover.
# Without a TTL cache, Codex Desktop traffic that reloads settings re-spawns it
# continuously (process parent = codex-shim). Keep automatic refreshes rare.
_OPENCODE_CLI_MODELS_CACHE_TTL_SEC = 3 * 60 * 60
_opencode_cli_models_cache: tuple[float, list[str]] | None = None
_opencode_cli_models_lock = threading.Lock()


def clear_opencode_cli_models_cache() -> None:
    global _opencode_cli_models_cache
    with _opencode_cli_models_lock:
        _opencode_cli_models_cache = None


def expire_opencode_cli_models_cache() -> None:
    """Mark the OpenCode CLI model list stale while keeping the last successful payload."""
    global _opencode_cli_models_cache
    with _opencode_cli_models_lock:
        cached = _opencode_cli_models_cache
        if cached is not None:
            _opencode_cli_models_cache = (0.0, cached[1])


def list_opencode_cli_models() -> list[str]:
    global _opencode_cli_models_cache
    now = time.monotonic()
    with _opencode_cli_models_lock:
        cached = _opencode_cli_models_cache
        if cached is not None and now - cached[0] < _OPENCODE_CLI_MODELS_CACHE_TTL_SEC:
            return list(cached[1])
        stale = list(cached[1]) if cached is not None else None

    binary = shutil.which("opencode")
    if not binary:
        if stale is not None:
            return stale
        with _opencode_cli_models_lock:
            _opencode_cli_models_cache = (time.monotonic(), [])
        return []
    try:
        result = subprocess.run(
            [binary, "models"],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return stale if stale is not None else []
    output = result.stdout or ""
    if result.returncode != 0:
        output = f"{result.stdout}\n{result.stderr}"
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if not lines and stale is not None:
        return stale
    with _opencode_cli_models_lock:
        _opencode_cli_models_cache = (time.monotonic(), lines)
    return list(lines)


def discover_opencode_cli_ids(prefix: str) -> list[str]:
    needle = f"{prefix.rstrip('/')}/"
    return [line[len(needle) :] for line in list_opencode_cli_models() if line.startswith(needle)]


def _discover_rows_for_template(template: DiscoverTemplate) -> list[str]:
    if template.kind == "zen_public":
        return fetch_zen_public_model_ids()
    if template.kind == "zen":
        return fetch_zen_paid_model_ids(api_key=_resolved_api_key(template.api_key))
    if template.kind in {"openrouter", "openrouter_free"}:
        return fetch_openrouter_free_model_ids()
    if template.kind in {"nvidia", "nvidia_integrate"}:
        return fetch_nvidia_integrate_model_ids()
    if template.kind in {"nous", "nous_portal"}:
        return fetch_nous_portal_model_ids(
            api_key=_resolved_api_key(template.api_key) or resolve_nous_api_key()
        )
    return []


def _rows_to_shim_models(model_ids: list[str], template: DiscoverTemplate) -> list[ShimModel]:
    models: list[ShimModel] = []
    used_slugs: set[str] = set()
    for offset, model_id in enumerate(sorted(model_ids, key=str.lower)):
        slug = _catalog_slug_for_model(model_id, template.slug_prefix, used_slugs, offset)
        meta = _metadata_for_discovered_model(template, model_id)
        display_name = display_name_from_slug(slug, label_prefix=template.label_prefix)
        upstream_name = str(meta.get("upstream_name") or "").strip()
        if upstream_name:
            prefix = ROUTE_LABEL_PREFIXES.get((template.label_prefix or "").lower())
            display_name = f"{prefix} — {upstream_name}" if prefix else upstream_name
        models.append(
            ShimModel(
                slug=slug,
                model=model_id,
                display_name=display_name,
                provider=template.provider,
                base_url=template.base_url,
                api_key=_resolved_api_key(_api_key_for_discovered_model(model_id, template)),
                index=DISCOVER_INDEX_BASE + offset,
                extra_headers=dict(template.extra_headers),
                max_context_limit=context_limit_for_discovered_model(
                    model_id,
                    reported=meta.get("reported_context_limit"),
                ),
                max_output_tokens=meta.get("output_limit"),
                no_image_support=bool(meta.get("no_image_support")),
                supports_reasoning_summaries=bool(meta.get("reasoning")),
                raw={"discovered": True, "discover_kind": template.kind, **meta},
            )
        )
    return models


def _catalog_slug_for_model(
    model_id: str,
    slug_prefix: str,
    used_slugs: set[str],
    offset: int,
) -> str:
    if model_id == _OPENROUTER_FREE_ROUTER and slug_prefix == "or":
        slug = "or-free-router"
        if slug in used_slugs:
            slug = f"{slug}-{offset}"
        used_slugs.add(slug)
        return slug
    body = slugify(model_id.replace("/", "-").replace(":", "-"))
    slug = f"{slug_prefix}-{body}" if slug_prefix else body
    if slug in used_slugs:
        slug = f"{slug}-{offset}"
    used_slugs.add(slug)
    return slug


def _api_key_for_discovered_model(model_id: str, template: DiscoverTemplate) -> str:
    if template.kind == "zen_public" and is_zen_public_model(model_id):
        return KEYLESS_API_KEY
    if template.kind in {"nous", "nous_portal"}:
        return resolve_nous_api_key() or template.api_key
    return template.api_key


def _resolved_api_key(api_key: str) -> str:
    raw = api_key.strip()
    if raw.startswith("${") and raw.endswith("}"):
        return os.environ.get(raw[2:-1].strip(), "")
    return raw


def _enrich_builtin_template(template: DiscoverTemplate, explicit_models: list[ShimModel]) -> DiscoverTemplate:
    needle = {
        "zen": "opencode.ai/zen",
        "openrouter_free": "openrouter.ai",
        "nvidia_integrate": "integrate.api.nvidia.com",
        "nous": "inference-api.nousresearch.com",
    }.get(template.kind, "")
    if not needle:
        return template
    for model in explicit_models:
        if needle not in model.base_url.lower():
            continue
        if template.kind == "zen" and _resolved_api_key(model.api_key) in {"", KEYLESS_API_KEY}:
            continue
        return DiscoverTemplate(
            kind=template.kind,
            base_url=template.base_url,
            provider=template.provider,
            slug_prefix=template.slug_prefix,
            api_key=model.api_key or template.api_key,
            extra_headers={**template.extra_headers, **model.extra_headers},
            label_prefix=template.label_prefix,
        )
    return template


def _template_has_credentials(template: DiscoverTemplate) -> bool:
    if template.kind == "zen":
        return bool(_resolved_api_key(template.api_key))
    if template.kind in {"zen_public", "openrouter_free", "nvidia_integrate"}:
        return True
    if template.kind in {"nous", "nous_portal"}:
        return bool(_resolved_api_key(template.api_key) or resolve_nous_api_key())
    return bool(_resolved_api_key(template.api_key))


def _context_from_model_row(row: dict[str, Any]) -> int | None:
    for key in ("context_length", "max_context_length", "n_ctx"):
        value = row.get(key)
        if value not in (None, ""):
            try:
                return int(value)
            except (TypeError, ValueError):
                pass
    meta = row.get("meta")
    if isinstance(meta, dict):
        for key in ("n_ctx_train", "n_ctx", "context_length"):
            value = meta.get(key)
            if value not in (None, ""):
                try:
                    return int(value)
                except (TypeError, ValueError):
                    pass
    return None


def _infer_slug_prefix(slug: str, *, default: str) -> str:
    if "-" not in slug:
        return default
    prefix = slug.split("-", 1)[0]
    return prefix or default


def discover_summary(
    explicit_models: list[ShimModel],
    *,
    settings_data: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    merged = discover_byok_models(explicit_models, settings_data=settings_data)
    discovered = [model for model in merged if model.raw.get("discovered") or model.raw.get("discovered_local")]
    rows: list[dict[str, str]] = []
    for model in discovered:
        kind = "local" if model.raw.get("discovered_local") else str(model.raw.get("discover_kind") or "byok")
        rows.append(
            {
                "slug": model.slug,
                "model": model.model,
                "display_name": model.display_name,
                "description": description_for_route(model.display_name, "via local Codex shim"),
                "kind": kind,
            }
        )
    return rows

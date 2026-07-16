from __future__ import annotations

import re

CURSOR_PREFIX = "cursor-"
CODEX_PREFIX = "codex-"
OC_FREE_PREFIX = "oc-free-"

CHATGPT_UPSTREAM_DEFAULT = "gpt-5.5"


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return slug or "model"


def catalog_slug_with_prefix(prefix: str, body: str) -> str:
    normalized_prefix = prefix if prefix.endswith("-") else f"{prefix}-"
    normalized_body = slugify(body)
    if normalized_body.startswith(normalized_prefix):
        return normalized_body
    return f"{normalized_prefix}{normalized_body}"


def cursor_catalog_slug(upstream_id: str) -> str:
    return catalog_slug_with_prefix(CURSOR_PREFIX, slugify(upstream_id))


def codex_catalog_slug(upstream_id: str) -> str:
    return catalog_slug_with_prefix(CODEX_PREFIX, slugify(upstream_id))


def upstream_from_codex_catalog_slug(catalog_slug: str) -> str:
    """Best-effort reverse of ``codex_catalog_slug`` for ChatGPT passthrough entries.

    ``slugify`` turns ``gpt-5.6-terra`` into ``gpt-5-6-terra``, so recover the
    dotted minor version for GPT/o-series IDs. Upstream IDs that already start
    with ``codex-`` (e.g. ``codex-auto-review``) are returned unchanged.
    """
    slug = str(catalog_slug or "").strip()
    if not slug.startswith(CODEX_PREFIX):
        return slug
    body = slug[len(CODEX_PREFIX) :]
    if body.startswith("gpt-") or re.match(r"^o\d", body):
        return re.sub(r"^((?:gpt|o\d*)-\d+)-(\d+)", r"\1.\2", body, count=1)
    return slug


CHATGPT_CATALOG_SLUG = codex_catalog_slug(CHATGPT_UPSTREAM_DEFAULT)

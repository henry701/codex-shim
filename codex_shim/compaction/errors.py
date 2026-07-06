from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any


@dataclass(frozen=True)
class UpstreamErrorDetail:
    status: int
    message: str
    code: str | None = None
    error_type: str | None = None
    param: Any = None
    extra: dict[str, Any] = field(default_factory=dict)
    raw_body: str = ""


def parse_upstream_error_detail(error_response: Any, *, fallback: str = "") -> UpstreamErrorDetail:
    status = int(getattr(error_response, "status", 502) or 502)
    text = (getattr(error_response, "text", None) or "").strip()
    if not text:
        return UpstreamErrorDetail(
            status=status,
            message=fallback or f"Upstream returned HTTP {status}",
        )
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return UpstreamErrorDetail(
            status=status,
            message=text[:2000],
            raw_body=text[:2000],
        )
    if not isinstance(payload, dict):
        return UpstreamErrorDetail(
            status=status,
            message=text[:2000],
            raw_body=text[:2000],
        )

    err = payload.get("error")
    message = fallback
    code: str | None = None
    error_type: str | None = None
    param: Any = None
    extra: dict[str, Any] = {}

    if isinstance(err, dict):
        nested_message = err.get("message")
        if isinstance(nested_message, str) and nested_message.strip():
            message = nested_message.strip()
        nested_code = err.get("code")
        if isinstance(nested_code, str) and nested_code.strip():
            code = nested_code.strip()
        nested_type = err.get("type")
        if isinstance(nested_type, str) and nested_type.strip():
            error_type = nested_type.strip()
        if "param" in err:
            param = err.get("param")
        known = {"message", "type", "code", "param"}
        extra = {
            key: value
            for key, value in err.items()
            if key not in known and value is not None
        }
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

    if not message:
        message = fallback or text[:2000]

    top_extra = {
        key: value
        for key, value in payload.items()
        if key not in {"error", "message", "code", "detail"} and value is not None
    }
    if top_extra:
        extra = {**extra, **top_extra}

    return UpstreamErrorDetail(
        status=status,
        message=message,
        code=code,
        error_type=error_type,
        param=param,
        extra=extra,
        raw_body=text[:2000],
    )


def describe_upstream_error(
    error_response: Any,
    *,
    context: str | None = None,
    fallback: str = "",
    include_raw_body: bool = True,
) -> str:
    if error_response is None:
        return fallback
    detail = parse_upstream_error_detail(error_response, fallback=fallback)
    segments: list[str] = []
    if context:
        segments.append(f"[{context}]")
    segments.append(f"HTTP {detail.status}")

    meta: list[str] = []
    if detail.code:
        meta.append(f"code={detail.code}")
    if detail.error_type and detail.error_type != detail.code:
        meta.append(f"type={detail.error_type}")
    if detail.param is not None:
        meta.append(f"param={detail.param!r}")
    if meta:
        segments.append(" ".join(meta))
    if detail.message:
        segments.append(detail.message)
    if detail.extra:
        segments.append(
            f"upstream_fields={json.dumps(detail.extra, default=str, ensure_ascii=False)[:500]}"
        )
    if include_raw_body and detail.raw_body and detail.raw_body not in (detail.message or ""):
        segments.append(f"upstream_body={detail.raw_body[:800]}")
    return " — ".join(segments) if segments else fallback


def upstream_error_message(error_response: Any, *, fallback: str = "") -> str:
    return parse_upstream_error_detail(error_response, fallback=fallback).message


def format_compaction_failure_detail(
    *,
    slug: str,
    provider: str,
    native_message: str,
    summarization_message: str | None = None,
    summarization_attempted: bool = False,
    tertiary_slug: str | None = None,
    tertiary_message: str | None = None,
    tertiary_attempted: bool = False,
) -> str:
    lines = [f"Compaction failed for {slug} ({provider})."]
    lines.append(f"Native compact: {native_message}")
    if summarization_attempted:
        lines.append(
            f"Summarization fallback: {summarization_message or 'returned empty summary'}"
        )
    if tertiary_attempted:
        target = tertiary_slug or "tertiary"
        lines.append(
            f"Tertiary fallback ({target}): {tertiary_message or 'returned empty summary'}"
        )
    elif tertiary_slug is None:
        lines.append(
            "Tertiary fallback: not configured "
            "(set compaction.tertiary_fallback_slug or passthrough_error_fallback for this model)."
        )
    return " ".join(lines)


def byok_upstream_context(route: Any) -> str:
    slug = getattr(route, "slug", None) or "unknown"
    model = getattr(route, "model", None) or slug
    base_url = getattr(route, "base_url", None) or "unknown-upstream"
    return f"{slug} → {model} @ {base_url}"

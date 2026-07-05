from __future__ import annotations

from typing import Any

from .input_audit import CompactionSanitizationAudit, summarize_compaction_input_items


def log_compaction_warnings(warnings: list[str]) -> None:
    for warning in warnings:
        if warning:
            print(f"[warn] compaction: {warning}", flush=True)


def log_compaction_fallback(
    slug: str,
    *,
    provider: str,
    native_status: int | str,
    target: str,
    message: str,
) -> None:
    print(
        f"[fallback] {slug} {provider} native compaction {native_status} -> {target}: "
        f"{message[:200]}",
        flush=True,
    )


def log_compaction_phase(phase: str, **fields: object) -> None:
    parts = [f"{key}={value!r}" for key, value in sorted(fields.items())]
    detail = " ".join(parts)
    print(f"[compaction] {phase} {detail}".strip(), flush=True)


def log_compaction_input_snapshot(
    phase: str,
    items: list[Any],
    *,
    tail: int = 8,
    **fields: object,
) -> None:
    count, summary = summarize_compaction_input_items(items, tail=tail)
    extra = " ".join(f"{key}={value!r}" for key, value in sorted(fields.items()))
    suffix = f" {extra}" if extra else ""
    print(
        f"[compaction] {phase} items={count} tail={summary!r}{suffix}",
        flush=True,
    )


def log_compaction_cache_expansion(
    *,
    context: str,
    session_key: str,
    previous_response_id: str,
    cached_items: int | None,
    delta_items: int,
    total_items: int | None,
    delta_summary: list[str] | None = None,
) -> None:
    if cached_items is None:
        print(
            f"[compaction] expand-{context} MISS session={session_key} "
            f"previous_response_id={previous_response_id} delta_items={delta_items} "
            f"delta_tail={delta_summary!r}",
            flush=True,
        )
        return
    print(
        f"[compaction] expand-{context} HIT session={session_key} "
        f"previous_response_id={previous_response_id} cached_items={cached_items} "
        f"delta_items={delta_items} total_items={total_items} delta_tail={delta_summary!r}",
        flush=True,
    )


def log_compaction_sanitization(audit: CompactionSanitizationAudit) -> None:
    log_compaction_phase(
        "sanitize-summary",
        incoming=audit.incoming_items,
        outgoing=audit.outgoing_items,
        dropped=len(audit.dropped),
        preserved=len(audit.preserved),
    )
    for ref, reason in audit.dropped:
        print(
            f"[compaction] sanitize DROP {ref.label()} reason={reason!r}",
            flush=True,
        )
    for ref, reason in audit.preserved:
        print(
            f"[compaction] sanitize PRESERVE {ref.label()} reason={reason!r}",
            flush=True,
        )
    if not audit.dropped and not audit.preserved:
        print(
            "[compaction] sanitize PASS no shim-side drops or preserves; input unchanged",
            flush=True,
        )


def log_compaction_budget_fit(
    *,
    estimated: int,
    budget: int,
    truncated: int = 0,
    chars_removed: int = 0,
    rewritten: int = 0,
    still_over: bool = False,
) -> None:
    if estimated <= budget and truncated == 0 and rewritten == 0:
        print(
            f"[compaction] budget PASS estimated={estimated} budget={budget}",
            flush=True,
        )
        return
    parts = [f"estimated={estimated}", f"budget={budget}"]
    if truncated:
        parts.append(f"truncated_outputs={truncated}")
    if chars_removed:
        parts.append(f"chars_removed={chars_removed}")
    if rewritten:
        parts.append(f"rewritten_outputs={rewritten}")
    if still_over:
        parts.append("still_over_budget=true")
    print(f"[compaction] budget FIT {' '.join(parts)}", flush=True)


def log_compaction_path(
    path: str,
    *,
    provider: str,
    slug: str,
    native_items: int | None = None,
    summarization_items: int | None = None,
    summary_chars: int | None = None,
    native_error: str | None = None,
    **fields: object,
) -> None:
    parts: list[str] = [
        f"path={path!r}",
        f"provider={provider!r}",
        f"slug={slug!r}",
    ]
    if native_items is not None:
        parts.append(f"native_items={native_items}")
    if summarization_items is not None:
        parts.append(f"summarization_items={summarization_items}")
    if summary_chars is not None:
        parts.append(f"summary_chars={summary_chars}")
    if native_error:
        parts.append(f"native_error={native_error[:200]!r}")
    for key, value in sorted(fields.items()):
        parts.append(f"{key}={value!r}")
    print(f"[compaction] complete {' '.join(parts)}", flush=True)


def log_compaction_upstream_body(
    *,
    route: str,
    phase: str,
    input_items: list[Any],
    **fields: object,
) -> None:
    count, summary = summarize_compaction_input_items(input_items, tail=10)
    extra = " ".join(f"{key}={value!r}" for key, value in sorted(fields.items()))
    suffix = f" {extra}" if extra else ""
    print(
        f"[compaction] upstream-{route} {phase} input_items={count} "
        f"input_tail={summary!r}{suffix}",
        flush=True,
    )

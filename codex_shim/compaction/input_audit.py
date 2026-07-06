from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CompactionInputItemRef:
    index: int
    item_type: str
    call_id: str | None = None
    name: str | None = None
    role: str | None = None

    def label(self) -> str:
        parts = [f"index={self.index}", f"type={self.item_type}"]
        if self.call_id:
            parts.append(f"call_id={self.call_id}")
        if self.name:
            parts.append(f"name={self.name}")
        if self.role:
            parts.append(f"role={self.role}")
        return " ".join(parts)


@dataclass
class CompactionSanitizationAudit:
    incoming_items: int = 0
    outgoing_items: int = 0
    dropped: list[tuple[CompactionInputItemRef, str]] = field(default_factory=list)
    preserved: list[tuple[CompactionInputItemRef, str]] = field(default_factory=list)
    synthesized: list[str] = field(default_factory=list)

    def warning_lines(self) -> list[str]:
        warnings = list(self.synthesized)
        for ref, _reason in self.dropped:
            detail = f"call_id={ref.call_id!r}" if ref.call_id else "missing call_id"
            warnings.append(f"dropped orphan {ref.item_type} ({detail})")
        for ref, reason in self.preserved:
            warnings.append(f"preserved {ref.item_type} ({ref.label()}): {reason}")
        return warnings


def compaction_input_item_ref(index: int, raw: dict[str, Any]) -> CompactionInputItemRef:
    item_type = str(raw.get("type") or raw.get("role") or "?")
    call_id = raw.get("call_id")
    name = raw.get("name")
    role = raw.get("role")
    return CompactionInputItemRef(
        index=index,
        item_type=item_type,
        call_id=str(call_id) if isinstance(call_id, str) and call_id else None,
        name=str(name) if isinstance(name, str) and name else None,
        role=str(role) if isinstance(role, str) and role else None,
    )


def summarize_compaction_input_items(
    input_items: list[Any],
    *,
    tail: int = 8,
) -> tuple[int, list[str]]:
    if not isinstance(input_items, list):
        return 0, []
    summary: list[str] = []
    for index, item in enumerate(input_items[-tail:]):
        absolute_index = len(input_items) - len(input_items[-tail:]) + index
        if not isinstance(item, dict):
            summary.append(f"@{absolute_index}:?")
            continue
        ref = compaction_input_item_ref(absolute_index, item)
        summary.append(f"@{absolute_index}:{ref.item_type}"
                       + (f"(call_id={ref.call_id})" if ref.call_id else "")
                       + (f"(role={ref.role})" if ref.role else ""))
    return len(input_items), summary

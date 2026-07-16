"""When to expand ``previous_response_id`` deltas from the conversation cache.

The Codex OAuth backend at ``chatgpt.com/backend-api/codex`` behaves differently
by transport:

- **HTTP** rejects ``previous_response_id`` outright (400 Unsupported parameter).
- **WebSocket** accepts ``previous_response_id`` only on a **reused** upstream
  connection (connection-scoped chaining). A fresh WS connection returns
  ``previous_response_not_found``.

Every other shim route (BYOK HTTP/WS, Cursor passthrough, compaction, local
bridge) has no native continuation support and must always expand from cache.

Cache **writes** are unconditional on all routes so cross-surface transitions
(e.g. Codex WS turn then HTTP compaction) can replay logical history.
"""

from __future__ import annotations

import os
from enum import Enum
from typing import Any

_PREV_ID_ERROR_MARKERS = (
    "previous_response_id",
    "previous_response_not_found",
    "Unsupported parameter",
)


class ContinuationSurface(Enum):
    HTTP = "http"
    WS = "ws"


class ContinuationRoute(Enum):
    CHATGPT_CODEX = "chatgpt_codex"
    BYOK = "byok"
    CURSOR = "cursor"
    LOCAL = "local"


def chatgpt_ws_force_expand() -> bool:
    env = os.environ.get("CODEX_SHIM_CHATGPT_WS_FORCE_EXPAND", "")
    return env.lower() in {"1", "true", "yes", "on"}


def should_expand_continuation(
    *,
    surface: ContinuationSurface,
    route: ContinuationRoute,
    upstream_connection_reused: bool = False,
    last_upstream_chained_response_id: str | None = None,
    body: dict[str, Any] | None = None,
) -> bool:
    if surface == ContinuationSurface.WS and route == ContinuationRoute.CHATGPT_CODEX:
        previous_response_id = (body or {}).get("previous_response_id")
        if chatgpt_ws_force_expand():
            return bool(previous_response_id)
        if not previous_response_id:
            return False
        if not upstream_connection_reused:
            return True
        if last_upstream_chained_response_id != previous_response_id:
            return True
        return False
    if not (body or {}).get("previous_response_id"):
        return False
    return True


def is_previous_response_id_upstream_error(
    message: str,
    *,
    code: str | None = None,
    param: str | None = None,
) -> bool:
    if param and "previous_response" in param.lower():
        return True
    if code and "previous_response" in code.lower():
        return True
    lowered = message.lower()
    return any(marker.lower() in lowered for marker in _PREV_ID_ERROR_MARKERS)


def is_previous_response_id_upstream_event(event: dict[str, Any]) -> bool:
    if event.get("type") not in {"error", "response.failed"}:
        return False
    err = event.get("error")
    if isinstance(err, dict):
        return is_previous_response_id_upstream_error(
            str(err.get("message") or ""),
            code=str(err.get("code") or "") or None,
            param=str(err.get("param") or "") or None,
        )
    detail = event.get("detail")
    if isinstance(detail, str):
        return is_previous_response_id_upstream_error(detail)
    return is_previous_response_id_upstream_error(str(event))

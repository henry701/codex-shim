# Cursor → Codex tool bridge

When Codex Desktop routes a turn through **Cursor/Composer passthrough**, the
shim normally renders Cursor tool activity as **reasoning** markdown. Codex never
sees `function_call` output items, so local tools such as Goals (`update_goal`,
`create_goal`, `get_goal`) never run and idle continuation can loop.

The **tool bridge** closes that gap: Composer is instructed (via a **suffix**
appended to the cursor-agent prompt) to forward Codex tools through a loopback
HTTP API. The shim injects real Responses `function_call` items into the stream
while Composer waits on curl stdout.

## Flow

```text
Codex Desktop
    │ POST /v1/responses (body.tools)
    ▼
codex-shim
    │ build_cursor_prompt(body)          ← stable prefix (cache-friendly)
    │ + [CODEX_SHIM_CURSOR_BRIDGE v1]…   ← dynamic suffix (bridge id, curl recipe)
    │ register bridge listener
    ▼
cursor-agent (Composer)
    │ Shell: curl POST /_cursor_bridge/v1/invoke
    ▼
codex-shim bridge handler
    │ validate bridge id + tool allowlist
    │ emit function_call SSE to Codex
    ▼
Codex Desktop executes tool locally
```

## Suffix protocol

When `body.tools` is non-empty and the bridge is enabled, the shim appends a
block tagged `[CODEX_SHIM_CURSOR_BRIDGE v1]` containing:

- A random **bridge session id** (16 alphanumeric characters)
- The invoke URL (`http://127.0.0.1:<port>/_cursor_bridge/v1/invoke`)
- An exact **curl** command template
- The **allowed tool list** derived from `body.tools`

Composer must use its **Shell** tool with that curl pattern. Only `TOOL` and
`ARGUMENTS` may be substituted.

## HTTP API

`POST /_cursor_bridge/v1/invoke`

Requirements:

- Loopback peer only (`127.0.0.1` or `::1`)
- Valid `Host` header (same host guard as the rest of the shim)

Request body:

```json
{
  "bridge": "<bridge_session_id>",
  "tool": "update_goal",
  "namespace": "goals",
  "arguments": {"status": "complete", "goal_id": "g1"}
}
```

- `bridge` — session id from the suffix
- `tool` — Codex tool name (short name when using namespaces)
- `namespace` — optional; required for namespaced tools when using short names
- `arguments` — JSON object (required)

Success response:

```json
{
  "ok": true,
  "bridge": "…",
  "tool": "update_goal",
  "namespace": "goals",
  "codex_call_id": "call_<bridge>_1"
}
```

Errors:

| Status | Cause |
|--------|--------|
| 403 | Non-loopback peer |
| 400 | Invalid JSON, missing fields, disallowed tool |
| 404 | Unknown or expired bridge session |

## Allowlist

Any tool present in the Codex request `body.tools` is allowed — including
namespace tools (stored internally as sanitized chat names like
`goals_update_goal`). The server enforces the full allowlist even when the
suffix truncates the displayed list.

## Environment variables

| Variable | Default | Meaning |
|----------|---------|---------|
| `CODEX_SHIM_CURSOR_BRIDGE` | `1` | Set `0` to disable suffix + bridge |
| `CODEX_SHIM_CURSOR_BRIDGE_TTL_S` | `1800` | Bridge session lifetime (seconds) |
| `CODEX_SHIM_CURSOR_BRIDGE_TOOL_LIST_CAP` | `40` | Max tools listed in suffix |

## Stream UX

Bridge shell commands are detected in Cursor NDJSON and rendered compactly as
`→ Codex tool: \`update_goal\`` instead of dumping the full curl into reasoning.
Injection is always driven by the HTTP invoke, not by parsing NDJSON.

## Troubleshooting

- **curl returns 404** — passthrough finished or bridge TTL expired; start a new Codex turn.
- **curl returns 400 tool not allowed** — tool was not in the original Codex `body.tools`.
- **Goals still loop** — confirm Composer actually ran the bridge curl (check reasoning / shell output) and that `update_goal` was invoked with `status: complete` or `blocked`.
- **`create_goal` fails validation** — Codex requires an `objective` string, not `name`/`description`.

## Smoke test

End-to-end tmux smoke (fresh temp dir, `codex exec`, composer-2.5, goal + bridge):

```bash
bash scripts/smoke_cursor_bridge_tmux.sh
# optional: SMOKE_ATTACH=1 to watch panes; SMOKE_PORT=8767 (default)
```

## Related

- [`cursor-agent-tools.md`](cursor-agent-tools.md) — Cursor tool catalog and fixtures
- README — Cursor/Composer passthrough section

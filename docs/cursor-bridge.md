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
    │ validate bridge id + denylist / turn tool set
    │ emit function_call SSE to Codex
    ▼
Codex Desktop executes tool locally
```

## Suffix protocol

When `body.tools` is non-empty and the bridge is enabled, the shim appends a
block tagged `[CODEX_SHIM_CURSOR_BRIDGE v1]` containing:

- A random **bridge session id** (16 alphanumeric characters)
- The invoke URL (`http://127.0.0.1:<port>/_cursor_bridge/v1/invoke`)
- An exact **curl** command template (supports optional `namespace`)
- A **bridged tool catalog** with Codex descriptions + compact JSON parameter
  schemas (collaboration/goals sorted first)
- Sub-agent and goal protocol notes

Composer must use its **Shell** tool with that curl pattern. Only `TOOL`,
`NAMESPACE_OR_EMPTY`, and `ARGUMENTS` may be substituted. File/shell/search/MCP
work stays on Cursor-native tools — never via the bridge.

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

## Bridged tool set (denylist)

The bridge starts from every tool in the Codex request `body.tools`, then
**denies** Cursor-overlapping / hosted / MCP runners:

| Denied | Examples |
|--------|----------|
| Shell / command | `exec_command`, `write_stdin`, `local_shell`, `exec`, `wait` |
| File patch | `apply_patch` |
| Web / computer | `web_search*`, `computer_use*`, `image_generation` |
| MCP | `list_mcp_resources`, `read_mcp_resource`, `tool_search`, `mcp__*` namespaces |

Everything else stays bridged — including **collaboration/sub-agent** tools
(`spawn_agent`, `wait_agent`, `send_message`, `list_agents`, `followup_task`,
`interrupt_agent`), **goals**, `update_plan`, `request_user_input`, and future
Codex-native control-plane tools that appear in `body.tools`.

The server enforces the denylist even when the suffix truncates the displayed
catalog (`CODEX_SHIM_CURSOR_BRIDGE_TOOL_LIST_CAP`, default 40).


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
- **curl returns 400 tool not available** — tool is denylisted (shell/file/web/MCP)
  or was not in this turn's Codex `body.tools`.
- **Goals still loop** — confirm Composer actually ran the bridge curl (check reasoning / shell output) and that `update_goal` was invoked with `status: complete` or `blocked`.
- **Sub-agents appear stuck** — use `send_message` / `followup_task` with a real
  `target` from `spawn_agent`/`list_agents`, then `wait_agent`; do not busy-loop
  `list_agents`/`get_goal` alone.
- **`create_goal` fails validation** — Codex requires an `objective` string, not `name`/`description`.
- **`spawn_agent` fails validation** — requires `task_name` + `message`.

## Verified in live session (2026-06-28)

Real bridged Codex session under Cursor Composer 2.5 (bridge session `uyqpNSBgq2SBawBG` on :8765):

- The full injected suffix block (including workspace path `/home/henry`, exact curl recipe, allowed tools list, and goal rules) was received by the inner agent exactly as emitted by `build_bridge_suffix`.
- `create_goal` was invoked via the prescribed `curl -sS -X POST ...` pattern using Shell tool; shim accepted it and returned `{"ok": true, ..., "codex_call_id": "call_..._1"}`, confirming the emit path to Codex.
- Regular operations (edits, `git`, tests via `uv run pytest`, `codex-shim doctor`) were performed with direct shell — bridge curl used **only** for listed Codex tools per the injected rules.
- `codex-shim doctor` reported: cursor_passthrough true, 1 session dir in chatgpt-conversations cache, 4 cached responses, stale-pid warning handled gracefully (health still OK).
- Goal completion will be signaled via `update_goal` + `status: "complete"` at end of task.

This validates the bridge solves goal/tool visibility for Codex when routed through Cursor passthrough.

## Smoke test

End-to-end tmux smoke (fresh temp dir, `codex exec`, composer-2.5, goal + bridge):

```bash
bash scripts/smoke_cursor_bridge_tmux.sh
# optional: SMOKE_ATTACH=1 to watch panes; SMOKE_PORT=8767 (default)
```

## Related

- [`cursor-agent-tools.md`](cursor-agent-tools.md) — Cursor tool catalog and fixtures
- README — Cursor/Composer passthrough section

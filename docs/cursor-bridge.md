# Cursor → Codex tool bridge

When Codex Desktop routes a turn through **Cursor/Composer passthrough**, the
shim normally renders Cursor tool activity as **reasoning** markdown. Codex never
sees `function_call` output items, so local tools such as Goals (`update_goal`,
`create_goal`, `get_goal`) never run and idle continuation can loop.

The **tool bridge** closes that gap: Composer is instructed (via a **suffix**
appended to the cursor-agent prompt) to forward Codex tools through a loopback
HTTP API. The shim injects real Responses `function_call` items into the stream.
Invoke is **async** (returns a `job_id` immediately). Composer then **wait**s or
**poll**s for the Codex tool output on HTTP — the only mid-turn channel, since we
cannot inject into Cursor chat.

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
    │ Shell: curl POST /_cursor_bridge/v1/invoke  → {job_id, status:accepted}
    │ Shell: curl POST /_cursor_bridge/v1/wait    → blocks until Codex output
    ▼
codex-shim bridge handler
    │ validate bridge id + denylist / turn tool set
    │ emit function_call SSE; on wait/poll, early-complete stream
    │ when next turn carries function_call_output → resolve job
    ▼
Codex Desktop executes tool locally; result returns via wait/poll JSON.
```

## One agent per session

Codex only runs tool calls once the response stream ends, so a bridged turn is
early-completed while cursor-agent is still mid-thought. That early-complete is
the **only** case where we keep the process alive: the bridge session owns it,
buffers whatever it emits, and a tool-output-only follow-up **adopts** it —
replaying the buffer, reopening the turn, and streaming live.

Steer / interrupt / user-cancel are different. Codex drops the HTTP request,
records whatever was already streamed (thinking + tool text) into history, and
sends a new turn with that history plus the steer text (see CLI
`steer_interrupts_wait_agent_and_is_sent_in_follow_up_request`). The shim
mirrors that: cancel the live cursor-agent on unexpected disconnect, and on any
follow-up that is not a tool-output delivery kill the orphan still blocked on
`wait` (including empty or steer-shaped turns). The next turn respawns from the
conversation cache + new input.

Adoption keys off the **tail** of `input`, not the whole list
(`input_items_deliver_tool_outputs`). Codex normally sends just the delta
(`previous_response_id` + outputs), but **after compaction it drops the response
chain and replays the entire conversation inline**. Matching the whole input
classified every post-compaction delivery as a steer, so the shim cancelled the
agent that owned those very results and respawned it — the agent then
re-announced its plan on each turn, producing a visible loop
(`cancel bridge=X (orphaned by steer/interrupt call_id=call_X_1)` immediately
after `early-complete bridge=X`). Steering appends the user's text *after* the
interrupted outputs, so the tail still separates the two cases.

Handoff disconnect is recognized only **after** `finish()` completes successfully
during early-complete — a disconnect while the stream is still open (user cancel)
always kills the agent.

A fresh agent is also spawned when there is no live one to adopt.

## Codex Desktop multi-agent (sub-agents)

Codex Desktop v2 multi-agent runs **spawned sub-agents in their own Desktop
threads** while the parent turn is routed through Cursor passthrough. The bridge
makes that work end-to-end: collaboration tools become real Codex
`function_call` items, results return through wait/poll, and parent/sub
`agent_message` traffic is preserved in the cursor-agent prompt.

### What is bridged

| Collaboration tool | Role in Desktop | Bridge behavior |
|--------------------|-----------------|-----------------|
| `spawn_agent` | Creates child thread (`/root/task_name`) | Invoke → early-complete → Desktop runs sub locally |
| `wait_agent` | Parent blocks until sub finishes | Parent cursor-agent waits on bridge; sub runs in parallel |
| `send_message` / `followup_task` | Inter-agent messaging | Same async invoke + wait/poll pattern |
| `list_agents` / `interrupt_agent` | Team introspection / cancel | Bridged when present in `body.tools` |

Goals (`create_goal`, `update_goal`, `get_goal`), `update_plan`, and other
non-denylisted Codex tools in the same turn use the same bridge path.

### Sub-agent task delivery (`agent_message`)

When Desktop spawns a sub-agent, the child's first turn includes an
`agent_message` input item (`author` → `recipient`, task text often in
`encrypted_content`). The shim's input translator (`translate.py`) turns that into
a normal user message in the cursor-agent prompt so the sub actually receives its
brief. **Dropping this item leaves a sub-agent with no instructions** — a common
failure mode before the translator fix.

Parent ↔ sub **FINAL_ANSWER** replies use the same `agent_message` shape in the
opposite direction. When the sub completes, Desktop injects that message into
the parent's next `/v1/responses` request; the parent cursor-agent respawns from
cached history plus the new input and continues.

### Live flow (verified 2026-07-25)

Parent thread `019f9789-…`, sub-agent `neutral_parity_audit` ("Poincare") on
Cursor Grok 4.5:

```text
Parent cursor-agent
  │ bridge invoke spawn_agent
  │ early-complete (parent HTTP stream ends; agent kept)
  ▼
Codex Desktop
  │ executes spawn_agent locally → new sub thread
  │ sub_agent_activity: started
  ▼
Sub-agent cursor-agent (separate thread)
  │ receives agent_message NEW_TASK via prompt translation
  │ read-only audit work (shell, logs, git)
  │ writes neutral-audit-20260725.md
  │ task_complete + FINAL_ANSWER agent_message → parent
  ▼
Codex Desktop
  │ next parent request carries agent_message + tool history (input≈36)
  ▼
Parent cursor-agent (respawned / adopted)
  │ reads audit verdict ("Goal not achieved. Do not commit/push.")
  │ update_plan + resumes fixing timeline-scroll failures
```

Shim log signatures for a healthy run:

```text
[cursor-bridge] invoke … tool=spawn_agent namespace=collaboration
[cursor-bridge] early-complete bridge=… so Codex can run pending tool calls
[cursor-bridge] invoke … tool=wait_agent namespace=collaboration
[cursor-bridge] delivery-wake ingested=1; no leftover text — continuing with Cursor passthrough
[req] … input=… agent_message(from=/root/neutral_parity_audit to=/root)
[cursor-bridge] adopt bridge=… ingested=N — continuing the in-flight cursor-agent
```

While the sub runs, the parent **looks idle in Desktop** but the parent's
cursor-agent is correctly blocked on `wait_agent` — not stopped. Desktop wakes
the parent when the sub's `FINAL_ANSWER` arrives.

### Cancel / steer during sub-agent work

If the user cancels or Desktop sends a non-adopt follow-up (`input=0` wake,
steer text, new user message) while an agent is blocked on bridge wait,
`cancel_live_agents_for_session` kills the orphan cursor-agent immediately so it
cannot keep editing files in the background. Tool-output deliveries (next bridge
round-trip) still preserve the agent. See **One agent per session** above.

A sub-agent's `FINAL_ANSWER` only reaches the user if the parent turn that
receives it survives long enough to speak. If the parent is caught in the
respawn loop above, the verdict lands in history but is never rendered, and a
following compaction can summarize it away — after which the parent re-reads
`list_agents`, believes the completed sub-agent is stuck, and calls
`interrupt_agent` on it repeatedly.

## Suffix protocol

When `body.tools` is non-empty and the bridge is enabled, the shim appends a
block tagged `[CODEX_SHIM_CURSOR_BRIDGE v1]` containing:

- A random **bridge session id** (16 alphanumeric characters)
- Invoke / wait / poll URLs
- Curl templates for async invoke + wait + poll
- A **bridged tool catalog** with Codex descriptions + compact JSON parameter
  schemas (collaboration/goals sorted first)
- Sub-agent and goal protocol notes

Composer must **batch invokes**, then **wait/poll** for results. The first
wait/poll early-completes the Codex stream so tools can run; further invokes
block until the next Codex turn adopts the agent and reopens the turn.
File/shell/search/MCP work stays on Cursor-native tools — never via the bridge.

## HTTP API

### `POST /_cursor_bridge/v1/invoke`

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

Accepted response (tool not finished yet):

```json
{
  "ok": true,
  "status": "accepted",
  "bridge": "…",
  "job_id": "…",
  "tool": "update_goal",
  "namespace": "goals",
  "codex_call_id": "call_<bridge>_1"
}
```

### `POST /_cursor_bridge/v1/wait`

Blocks until that job's Codex `function_call_output` arrives (or timeout).
Consumes the job so a later poll does not return it again.

```json
{"bridge": "…", "job_id": "…", "timeout_ms": 180000}
```

Success:

```json
{
  "ok": true,
  "job_id": "…",
  "codex_call_id": "…",
  "tool": "update_goal",
  "namespace": "goals",
  "status": "consumed",
  "output": "<raw Codex tool output string or object>"
}
```

### `POST /_cursor_bridge/v1/poll`

Pull channel: returns all **ready** jobs for the bridge session and removes them
from in-flight memory (no duplicates on later polls). Optional `timeout_ms`
waits for at least one.

```json
{"bridge": "…", "timeout_ms": 60000}
```

```json
{"ok": true, "bridge": "…", "jobs": [/* consumed job objects */], "pending": 0}
```

When a session has neither ready nor pending jobs the reply adds `"idle": true` and a
`hint`, because agents otherwise re-poll an empty session in a loop.

Errors:

| Status | Cause |
|--------|--------|
| 403 | Non-loopback peer |
| 400 | Invalid JSON, missing fields, disallowed tool |
| 404 | Unknown/expired bridge session or unknown `job_id` |

A 404 for a stale bridge returns a JSON body (`error: "unknown_bridge"`, `retryable: false`)
that tells the agent to switch to the current turn's bridge id instead of retrying. Bare
404s previously produced reconnect storms and goals marked blocked after a shim restart.

Sessions are reclaimed when idle, and a TTL sweep drops sessions whose jobs were left
ready-but-unconsumed (a turn that invokes three tools but waits on one).

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
| `CODEX_SHIM_CURSOR_BRIDGE_WAIT_TIMEOUT_MS` | `180000` | Default wait/poll timeout |

## Stream UX

Bridge shell commands are detected in Cursor NDJSON and rendered compactly as
`→ Codex tool: \`update_goal\`` instead of dumping the full curl into reasoning.
Injection is always driven by the HTTP invoke, not by parsing NDJSON.

## Troubleshooting

- **curl returns 404** — passthrough finished or bridge TTL expired; start a new Codex turn.
- **curl returns 400 tool not available** — tool is denylisted (shell/file/web/MCP)
  or was not in this turn's Codex `body.tools`.
- **invoke ok but no tool effect** — you must `wait`/`poll` for `output`; invoke only accepts the job.
- **Goals still loop** — confirm Composer ran invoke **and** wait/poll (check shell output for `output`), and that `update_goal` used `status: complete` or `blocked`.
- **Sub-agents appear stuck** — batch `spawn_agent` / `send_message` / `wait_agent` invokes, then wait/poll; do not busy-loop `list_agents`/`get_goal`.
- **Sub-agent polls the bridge instead of working** — its task arrives as an `agent_message`
  input item (`author`/`recipient` plus the payload in an `encrypted_content` part).
  The translator renders it as an attributed user message; if it is dropped the sub-agent
  starts with no instructions at all.
- **`create_goal` fails validation** — Codex requires an `objective` string, not `name`/`description`.
- **`spawn_agent` fails validation** — requires `task_name` + `message`.

## Verified in live session (2026-07-25)

Multi-agent + goals under Cursor Grok 4.5 on session `019f9789-…`:

- Parent spawned `neutral_parity_audit` via bridged `spawn_agent`; Desktop
  recorded `sub_agent_activity: started` on child thread `019f9a9b-…`.
- Sub-agent received `agent_message` NEW_TASK, produced
  `neutral-audit-20260725.md`, and completed with `task_complete`.
- Parent received `FINAL_ANSWER` (`Goal not achieved. Do not commit/push.`),
  resumed via adopt on a fresh bridge turn, and continued `update_plan` +
  code fixes (timeline scroll / `composerLayoutEpoch`).
- Bridged `wait_agent` correctly blocked the parent cursor-agent while the sub
  worked; `delivery-wake` + `agent_message` follow-up woke the parent without a
  manual user message.
- User-cancel path: `cancel-live` fired on superseding turns; orphan agents were
  killed before respawn (see shim log `cancel-live count=1`).

Earlier verification (2026-06-28, bridge session `uyqpNSBgq2SBawBG` on :8765):

- The full injected suffix block (including workspace path `/home/henry`, exact curl recipe, allowed tools list, and goal rules) was received by the inner agent exactly as emitted by `build_bridge_suffix`.
- `create_goal` was invoked via the prescribed `curl -sS -X POST ...` pattern using Shell tool; shim accepted it and returned `{"ok": true, ..., "codex_call_id": "call_..._1"}`, confirming the emit path to Codex.
- Regular operations (edits, `git`, tests via `uv run pytest`, `codex-shim doctor`) were performed with direct shell — bridge curl used **only** for listed Codex tools per the injected rules.
- `codex-shim doctor` reported: cursor_passthrough true, 1 session dir in chatgpt-conversations cache, 4 cached responses, stale-pid warning handled gracefully (health still OK).
- Goal completion will be signaled via `update_goal` + `status: "complete"` at end of task.

This validates the bridge for goals **and** Codex Desktop sub-agent collaboration when routed through Cursor passthrough.

## Smoke test

End-to-end tmux smoke (fresh temp dir, `codex exec`, composer-2.5, goal + bridge):

```bash
bash scripts/smoke_cursor_bridge_tmux.sh
# optional: SMOKE_ATTACH=1 to watch panes; SMOKE_PORT=8767 (default)
```

## Related

- [`cursor-agent-tools.md`](cursor-agent-tools.md) — Cursor tool catalog and fixtures
- README — Cursor/Composer passthrough section

# cursor-agent tools (Composer passthrough)

Codex-shim routes Composer subscription models through `cursor-agent --output-format stream-json`.
Tool activity is translated into **reasoning** summary markdown for Codex Desktop — not executable
`function_call` items — so users see what Composer did without the shim re-running tools locally.

## NDJSON event types

| `type` | `subtype` | Shim event |
|--------|-----------|------------|
| `system` | `init` | ignored |
| `user` | — | ignored |
| `thinking` | `delta` / `completed` | `thinking_delta` / `thinking_completed` |
| `tool_call` | `started` / `completed` | `tool_started` / `tool_completed` |
| `assistant` | streaming / boundary | `text_delta` / `segment_boundary` |
| `connection` | `reconnecting` | `connection_interrupted` |
| `result` | `success` / `error` | usage + final text / error |

Implementation: [`codex_shim/cursor_passthrough.py`](../codex_shim/cursor_passthrough.py) (`CursorStreamParser`).

## Discovered tool keys (composer-2.5, 2026-06)

Captured with [`scripts/capture_cursor_tool_traces.sh`](../scripts/capture_cursor_tool_traces.sh).
Fixtures live under [`tests/fixtures/cursor_stream/`](../tests/fixtures/cursor_stream/).

| Tool key | Display kind | Args (sample) | Result (sample) |
|----------|--------------|---------------|-----------------|
| `readToolCall` | `read` | `path`, optional `limit` | `success.content` |
| `writeToolCall` | `write` | `path`, `fileText` | file write metadata |
| `editToolCall` | `edit` | `path`, `streamContent` | `success.diffString`, `afterFullFileContent` |
| `deleteToolCall` | `delete` | `path` | `success.deletedFile`, `prevContent` |
| `globToolCall` | `glob` | `globPattern`, `targetDirectory` | matched file list / counts |
| `grepToolCall` | `grep` | `pattern`, `path`, `caseInsensitive`, … | `success.workspaceResults` |
| `shellToolCall` | `shell` | `command`, … | `success.stdout` / `exitCode` |
| `runTerminalCommand`, `bashToolCall`, `terminalToolCall` | `shell` | same family as shell | same |
| `function` | function name | `name`, `arguments` | varies |

Notes from live captures:

- Composer often uses **`editToolCall`** instead of `writeToolCall` for new files.
- **Move/rename** may use `shellToolCall` (`mv`) rather than a dedicated move tool.
- Unknown or future tools fall back to JSON-fenced reasoning blocks (see below).

## Rendering in Codex Desktop

### Thinking

Streamed thinking deltas become:

```markdown
**cursor-agent · thinking**

{text}
```

### Known tools

```markdown
**cursor-agent · read**

> `path/to/file`
```

On completion, a result block is appended:

```markdown
**Result**

```
{extracted text}
```
```

### Unknown tools

When a tool key is not in `CURSOR_TOOL_SPECS`, the shim emits transparent JSON:

```markdown
**cursor-agent · unknown**

```json
{
  "tool": "mysteryToolCall",
  "phase": "started",
  "args": { ... }
}
```
```

Completed unknown tools include `"phase": "completed"` and `"result"` in the JSON block.

Noise fields (`hookAdditionalContexts`, `parsingResult`, `toolCallId`, …) are stripped unless
`CODEX_SHIM_CURSOR_TOOL_VERBOSE=1`. Large payloads truncate at 4096 characters.

## Refreshing captures

```bash
# Requires: cursor-agent login, composer-2.5 subscription
bash scripts/capture_cursor_tool_traces.sh

# Inventory tool keys in fixtures or ad-hoc captures
python3 scripts/extract_cursor_tool_keys.py tests/fixtures/cursor_stream/*.ndjson

# Visual replay
python3 -m codex_shim.cursor_stream_visualizer tests/fixtures/cursor_stream/grep_search.ndjson \
  --workdir /tmp/example
```

Optional tmux smoke (agent + live visualizer):

```bash
bash scripts/cursor-passthrough-smoke-tmux.sh
```

## Adding a new tool mapping

1. Capture a trace that exercises the tool (capture script or manual `cursor-agent --print …`).
2. Add a `CursorToolSpec` entry to `CURSOR_TOOL_SPECS` in `cursor_passthrough.py`.
3. Add a fixture under `tests/fixtures/cursor_stream/` and replay tests in `test_cursor_passthrough_smoke.py`.
4. Update this document’s tool table.

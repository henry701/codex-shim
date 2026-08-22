# Changelog

All notable changes to this project will be documented in this file.

The format is loosely based on [Keep a Changelog](https://keepachangelog.com/),
and this project does not yet follow semantic versioning (pre-1.0).

## Unreleased

### Added

- `nous` / `nous-*` matches Hermes Agent Nous Portal: inference at
  `https://inference-api.nousresearch.com/v1`, `NOUS_API_KEY` or the OAuth JWT
  in `~/.hermes/auth.json`. `serve` / `sync-desktop` / `discover` force-refresh
  the device-code grant on startup (same `/api/oauth/token` +
  `x-nous-refresh-token` shape as Hermes CLI) and persist the rotated refresh
  token immediately. Hermes Referer / X-Title as setdefaults. Catalog listing
  is Portal `/v1/models` plus `stealth/ox-alpha`.

- `zen_public` / `oc-free-*` now matches hermes-cli OpenCode Free: models.dev
  listing, keyless `https://opencode.ai/zen/v1` chat (no `Authorization`; the
  internal `public` sentinel is not sent), and Hermes `HTTP-Referer` / `X-Title`
  as setdefaults. Desktop's User-Agent still wins. `big-pickle` may 429 under
  that UA.

- ChatGPT passthrough catalog prefers live
  `https://chatgpt.com/backend-api/codex/models?client_version=…` using the Codex
  OAuth token in `~/.codex/auth.json` (retries with exponential backoff via
  stdlib `urllib`). Falls back to `~/.codex/models_cache.json` or hardcoded
  slugs only when the backend listing is unavailable, with a warning log —
  breaks the circular dependency when Codex `openai_base_url` points at the shim.

- User logrotate for `~/.codex-shim/shim.log`: rotate at 30M, keep 10 compressed
  archives (`copytruncate`), via `codex-shim install-logrotate` and an hourly
  systemd user timer. `install-service` installs it automatically and restores
  `StandardOutput`/`StandardError` append to the service log path.

- `codex-shim serve` binds the HTTP server without catalog sync. The systemd
  user unit uses `ExecStartPre=sync-desktop` and `ExecStart=serve` so a restart
  is reachable before Desktop's request timeout. `codex-shim run` still syncs
  then serves for interactive use.

- `codex-shim prune-chatgpt-cache [--target 0.8]` evicts cold expansion-cache
  files down to a fraction of `CODEX_SHIM_CHATGPT_CACHE_MAX_BYTES`. Doctor WARN
  at ≥90% disk points at this command.

### Fixed

- Truncated JSON `apply_patch` envelopes at stream end are no longer emitted
  as complete `custom_tool_call` input. JSON-looking custom-tool arguments
  must parse; leftover non-JSON patch text is still accepted at stream end.

- Nous OAuth refresh treats persist as part of the critical section: both
  `auth.json` and the shared Nous store are retried after HTTP 200, and a
  failed write is retried from memory without re-posting the old
  single-use refresh token. Startup refresh is marked done only after a
  successful persist (or when there is no refresh token). HTTP failures
  retry on the next serve/sync/discover call. Corrupt `auth.json` is not
  rewritten. Hermes `active_provider` is left unchanged. The shared store
  follows `HERMES_SHARED_AUTH_DIR` and is locked as `nous_auth.json.lock`.

- BYOK `apply_patch` is a Codex freeform custom tool. When the client advertises
  it as `type: custom` or `type: function`, the shim previously echoed a
  `function_call` with JSON `{"input": "..."}`. Codex then fatals with
  `apply_patch invoked with incompatible payload`. Name `apply_patch` (and other
  `custom`/`freeform` tools) now round-trip as `custom_tool_call`, JSON
  `{input}`/`{patch}` envelopes are unwrapped, and stream events use
  `custom_tool_call_input` instead of `function_call_arguments`.

- ChatGPT Lite rejects ``function_call.id`` values that start with ``call_``
  (``invalid_id_prefix``, expected ``fc``). Cursor-bridge and BYOK translation
  reused the Chat Completions call id for both fields. Item ``id`` is now
  ``fc_<suffix>``; ``call_id`` stays ``call_<suffix>`` so tool outputs still
  match. Passthrough sanitizes already-stored Desktop history the same way and
  drops leaked ``call_``/``fc_`` ids from ``function_call_output``.

- ChatGPT conversation cache is an LRU (RAM + disk) with an incremental size
  index: `get()` promotes entries, eviction no longer walks the tree on every
  write, and passthrough stores run on the event loop instead of
  `asyncio.to_thread(put)` (that worker-thread mutation raced with eviction).
  Cache store failures still do not abort an already-streamed turn.
  `codex-shim prune-chatgpt-cache` shrinks a full disk cache without raising
  the 512M default.

- ChatGPT passthrough retries edge blips (HTTP 408/425/429/5xx, HTML
  “Unable to load site” 403, connect timeout/`ECONNRESET`/`EPIPE`) on a fresh
  TCP connection. Exhausted HTML site-down becomes a short 502 instead of an
  HTML blob. JSON 403/401 are not retried. Compact 404 stays non-retryable.

- HTTP SSE to Desktop writes `: ping` every 15s while ChatGPT is silent, so
  Luna think no longer trips Desktop’s idle request timeout.

- Upstream ChatGPT/BYOK websockets enable aiohttp ping/pong (`heartbeat=30`).
  Lanes with `exception()` set are not reused; a failed `send_str` or
  CLOSE/ERROR during relay drops the lane. Missing model tokens are not treated
  as a dead socket. If heartbeat pong hits a closing transport
  (`ClientConnectionResetError`), the lane is dropped and ChatGPT WS falls back
  to HTTP+SSE instead of 500ing the Desktop request.

- `codex-shim stop` / `restart` talk to the systemd user unit when it is
  installed, so `install-service --now` actually replaces a running `serve`.
  `sync-desktop` as ExecStartPre has a 45s discovery budget (keeps the existing
  catalog on timeout) and `TimeoutStartSec=180`. `serve` binds a cached health
  snapshot without a live ChatGPT `/models` fetch (explicit settings models only,
  no provider discovery).

- ChatGPT passthrough forwards ``reasoning.effort`` unchanged (including
  ``max`` / ``ultra``). ChatGPT Codex accepts those values; clamping them to
  ``xhigh`` was the public ``api.openai.com`` constraint applied on the wrong
  host. Desktop/catalog own the effort string.

- Doctor treats a systemd-managed listener as INFO when the pid file is unused,
  instead of warning “stale pid file; run stop”.

- `codex-shim --port <not 8765> restart` no longer bounces the systemd user unit.
  ChatGPT e2e smoke starts `serve` on 8766 so it cannot rewrite Desktop’s catalog
  or restart the live session.

- ChatGPT/BYOK upstream forwarding no longer relays client `Content-Encoding`.
  Desktop may zstd-compress the body to the shim; after decompress + JSON rewrite
  the shim POSTs plain JSON, so a leftover `Content-Encoding: zstd` made ChatGPT
  return a bare `{"detail":"Bad Request"}` (especially on large full-history
  resumes where request compression is used).

- Compaction tertiary fallback resolver now awaits async `route_fn` (fixes
  `tertiary-skip` when `compaction.tertiary_fallback_slug` or
  `passthrough_error_fallback` target had valid credentials). Skip logging reports
  `not_configured`, `no_credentials`, or `route_error` instead of a generic
  message.

### Changed

- ChatGPT passthrough no longer strips `service_tier`, `max_output_tokens`, or
  `max_tokens`. Fast mode (`priority`) and output caps are forwarded. Legacy
  `/compact` still omits `store`/`stream`.

- ChatGPT passthrough compaction is a thin proxy: `compaction_trigger` on
  `/v1/responses` is forwarded to `/codex/responses` (Desktop remote compact v2)
  instead of the shim orchestrator. Legacy `/v1/responses/compact` maps 1:1 to
  `/codex/responses/compact` and returns upstream status with no summarization
  fallback. A compact 404 is logged as a known upstream gap rather than a
  verbose io-resp dump. Cursor and BYOK still use the compaction orchestrator.

- Compaction prep collapses consecutive duplicate `message(role=user)` items
  before sanitization (client compaction retry artifact).

- Transport-aware continuation expansion: HTTP, BYOK (HTTP+WS), Cursor passthrough,
  and compaction always expand `previous_response_id` from the conversation cache.
  Only ChatGPT Codex OAuth WebSocket passthrough forwards native `previous_response_id`
  on a reused upstream connection (expand on new connect or upstream prev_id error).
  Cache writes are unconditional on all routes. Removed deprecated
  `CODEX_SHIM_CHATGPT_EXPAND_CONTINUATIONS` behavior (use `CODEX_SHIM_CHATGPT_WS_FORCE_EXPAND`
  to force Codex WS expansion).

- ChatGPT Codex WS `previous_response_id` error retry closes and reconnects upstream
  before the expanded retry. Conversation cache persists only at turn completion
  (no mid-stream disk writes). Terminal `put` replaces existing on-disk entries.

- Compaction failures now return a chained error message to Codex (native,
  summarization, and tertiary attempts) instead of only the first upstream error.
  Each phase includes upstream route context, HTTP status, provider error code,
  and a truncated raw upstream response body when available. Logs add
  `summarization-fail`, `tertiary-fail`, and `tertiary-skip` phases.

- Default `compaction_output_token_reserve` is now 20000 tokens (was
  `summary_max_output_tokens` + 8192 instruction overhead).

### Added

- BYOK routes now persist Responses turn snapshots to the conversation cache so
  `previous_response_id` delta continuations expand correctly (fixes blank-message
  regressions on chat-completions providers).

- Global byte cap for the expansion cache (`CODEX_SHIM_CHATGPT_CACHE_MAX_BYTES`,
  default 512M). Oldest entries are evicted FIFO across all sessions when over
  limit; per-session count cap (1024) still applies.

- Unified compaction engine (`codex_shim/compaction/`): Codex-aligned input
  preparation, OpenCode-style summarization prompts, provider-agnostic fallback
  chain (native → summarization → tertiary BYOK), configurable compaction model,
  and prompt-cache-friendly stable instruction prefixes. Cursor and BYOK routes
  now use the same fallback path as ChatGPT when native compact fails.

- ChatGPT passthrough conversation cache persistence: expansion snapshots are stored
  per Codex `session-id` (or `thread-id`) as immutable JSON files under
  `~/.codex-shim/chatgpt-conversations/` (`CODEX_SHIM_CHATGPT_CONVERSATIONS_DIR`).
  Survives shim restarts; mtime FIFO eviction keeps 1024 responses per session.
  `codex-shim doctor` reports cache stats.

- Cursor-agent tool translation registry: explicit mappings for `deleteToolCall`,
  `globToolCall`, and `grepToolCall`; unknown tools render as JSON-fenced reasoning
  blocks (`CODEX_SHIM_CURSOR_TOOL_VERBOSE=1` keeps noisy fields). Live composer-2.5
  fixtures and capture scripts: `scripts/capture_cursor_tool_traces.sh`,
  `scripts/extract_cursor_tool_keys.py`, `docs/cursor-agent-tools.md`.

- Bridge live verification: end-to-end curl invoke from Composer 2.5 session (`create_goal`, `update_goal`) works; `codex-shim doctor` now includes conversation cache stats; injected suffix matches production usage in Cursor passthrough.

### Fixed

- Compaction input budgeting now derives from the compaction model's context window
  minus `compaction_output_token_reserve` (replacing the flat
  `context_window_token_budget` input cap). Truncation and pruning run only when
  estimated input exceeds that budget; if still over budget afterward the shim
  logs a warning and continues with fallback compaction.

- Shared responses input pipeline (`responses_input_pipeline.py`): ChatGPT
  conversation cache expansion now runs for all models (BYOK, compaction, WS),
  not only ChatGPT passthrough. Orphan tool outputs synthesize a placeholder
  `unknown_tool` call instead of being dropped, with WARN logging. Chat
  translation gets the same safety net for deferred orphan tool messages.

- ChatGPT compaction v2 now expands `previous_response_id` deltas from the
  conversation cache (same as normal turns). Detached tail tool outputs are
  repaired via synthetic tool calls when no matching call exists in the batch.

- WebSocket upstream passthrough for ChatGPT and BYOK `openai-responses` routes:
  Codex's `/v1/responses` WebSocket now proxies to upstream WSS (`wss://chatgpt.com/backend-api/codex/responses`
  or `{base_url}/responses`) instead of translating each frame through HTTP+SSE. BYOK chat-completions and
  Anthropic routes still use the internal HTTP bridge. Set `CODEX_SHIM_WS_PASSTHROUGH=0` to force the legacy
  HTTP upstream path; upstream connect failures automatically fall back to HTTP+SSE.
- Remote compaction v2 over WebSocket and ChatGPT HTTP passthrough: requests ending in
  `compaction_trigger` now hit the compact summarization path (one `compaction` output item) instead of
  being forwarded to upstream ChatGPT/BYOK responses, which returned reasoning/message pairs Codex rejects.
- `codex-shim stop` now terminates orphan listeners when the pid file is stale but
  `/health` still responds on the default port (uses `ss` when available).
- `codex-shim doctor` warns when the pid file and port listener disagree.
- ChatGPT passthrough now forwards Codex client headers (session/thread metadata,
  `x-codex-*`) to ChatGPT and relays safe upstream response headers back. WebSocket
  upgrade headers are no longer forwarded to the HTTP upstream. ChatGPT rejects
  native `previous_response_id`; the shim strips it and replays delta continuations
  by default (disable with `CODEX_SHIM_CHATGPT_EXPAND_CONTINUATIONS=0`).
- BYOK routes (`openai-responses`, OpenAI chat, Anthropic) use the same
  client-header-first forwarding and upstream response header relay.
- Remote compaction v2 for BYOK/OpenCode models: when Codex appends a terminal
  `compaction_trigger` to `POST /v1/responses`, the shim now runs a compact
  summarization request and streams exactly one `compaction` output item instead
  of forwarding reasoning/message pairs that Codex rejects.
- BYOK `/v1/responses/compact` now returns a canonical `compaction` output item
  (with shim-encoded `encrypted_content`) instead of a plain assistant message.
- BYOK translation now feeds hosted `web_search` round-trips back to the upstream
  model. When Codex's hosted web search returns absolutely empty results, the shim
  substitutes a clear unavailable message so the model can fall back to MCP search
  tools instead of ending the turn with no feedback.

### Changed

- Added `scripts/smoke_chatgpt_passthrough.sh` for live validation on an alternate
  port with trace logging (`CODEX_SHIM_UPSTREAM_HEADER_LOG`,
  `CODEX_SHIM_PASSTHROUGH_TRACE`). The script summary includes `[ws-passthrough]`
  vs `http-fallback` markers when WS upstream is active or falls back.
- ChatGPT passthrough conversation cache raised from 128 to 1024 responses.

- Desktop catalog picker order now matches alphabetical slug order: `write_catalog`
  assigns ascending `priority` values per tier (BYOK, ChatGPT passthrough, Cursor,
  router) because Codex sorts by `priority`, not JSON array order. Both
  `~/.codex-shim/.codex-shim/custom_model_catalog.json` and
  `~/.codex/custom_model_catalog.json` are updated.
- `opencode-go refresh` and `discover --refresh` regenerate the Desktop catalog
  after refreshing model listings.
- `~/.codex-shim/models.json` is intended for local/niche routes plus shim
  config (`discover`, `passthrough_error_fallback`, router); cloud providers are
  auto-discovered at catalog sync time instead of being duplicated manually.
- Merged BYOK model lists and shim API model endpoints (`GET /api/models`,
  `GET /v1/models`) return routes sorted alphabetically by slug.
- OpenCode Zen picker labels are normalized at catalog time: free routes use
  `OpenCode Zen (free) — …`, paid routes use `OpenCode Zen — …`, regardless of
  whether the model came from explicit `models.json` entries or auto-discovery
  (`oc-free-*` vs `zen-*` slug prefixes).
- Cursor subscription models from `cursor-agent --list-models` are prefixed with
  `Cursor - ` in the picker.
- Responses `input_image.detail` values are normalized before OpenAI-chat
  forwarding: Codex's `original` becomes `high`, and unknown values become
  `auto`.
- Shim routing again uses Codex's built-in `openai` provider with
  `openai_base_url` pointed at the local shim, so Desktop recent threads stay in
  the same provider namespace across reboots. `enable` / `model use` migrate
  legacy `codex_shim` thread rows to `openai`. ChatGPT passthrough models can
  use Codex's Responses WebSocket transport through the shim; BYOK models are
  bridged through the shim's existing HTTPS-backed `/v1/responses` translation
  path and relayed back over the WebSocket.
- Managed-config backup metadata stores displaced top-level values as TOML RHS
  fragments (not full `key = value` lines), with backward-compatible restore for
  older installs.
- BYOK chat routes learn and persist upstream quirks in
  `~/.codex-shim/upstream-compat.json`. Models that reject `parallel_tool_calls`
  (e.g. OpenCode Zen North Mini Code) trigger one transparent retry with the
  field stripped; later requests and catalog metadata omit it proactively.
- Namespace tools forwarded to strict OpenAI-compatible upstreams use
  underscore-separated chat tool ids (e.g. `codex_app_load_workspace_dependencies`
  instead of `codex_app.load_workspace_dependencies`). A per-request resolve map
  restores `namespace` + short `name` in Responses output.
- Model catalog responses (`GET /api/models`, `GET /v1/models`, and
  `write_catalog` JSON) are sorted alphabetically by slug for deterministic
  ordering across harnesses.
- Cursor passthrough (`cursor-agent` stream-json) filters duplicate assistant
  flushes per Cursor CLI rules (`timestamp_ms` / `model_call_id`), surfaces
  tool activity as non-executable reasoning blocks with blockquote markdown,
  supports multi-segment assistant messages per turn, merges thinking deltas,
  and handles both incremental and cumulative assistant text fragments.
- `editToolCall` / `writeToolCall` tool markdown includes path and content preview.
- `codex_shim.cursor_stream_visualizer` replays captured NDJSON with colored
  terminal output; `scripts/cursor-passthrough-smoke-tmux.sh` splits agent vs
  visualizer in tmux for manual smoke runs.
- Fixture-backed smoke tests in `tests/test_cursor_passthrough_smoke.py` cover
  sleep/shell/read and write/edit/patch captures (`tests/fixtures/cursor_stream/`).
- `sync-desktop` and the systemd/`run` path refresh `~/.codex/custom_model_catalog.json`
  only; they no longer write `~/.codex/config.toml`. Use `codex-shim enable` (or `app` /
  `model use`) to install the managed OpenAI-provider shim routing while the
  background service keeps the catalog current.
- Package management and docs now use uv (`uv.lock`, `uv sync`, `uv tool install
  -e .`, `uv run pytest`) instead of pip; CI uses `astral-sh/setup-uv`.
- Namespace tool translation uses dot notation (`namespace.tool`) instead of
  double underscores for BYOK chat/anthropic routes; `type: "namespace"` tools
  in requests are expanded for upstream models.
- `responses_compact` anthropic path now calls `_post_anthropic` instead of
  incorrectly posting to chat completions.

### Added

- `codex-shim doctor`: read-only diagnostics for Python/Codex CLI availability,
  settings, generated runtime files, daemon health, ChatGPT/Cursor passthrough,
  loopback proxy variables, and this fork's managed `openai_base_url` routing.
- `openai-responses` provider (`ShimModel.is_openai_responses`): passthrough to
  upstream `/v1/responses` without chat-completions translation (raw SSE/JSON).
- `parse_mcp_tool_reference()` accepts MCP chat names in both `mcp__srv__tool`
  and `mcp__srv.tool` forms; streaming and non-streaming response paths preserve
  `namespace` + `name` on `function_call` items for generic namespaces and MCP.
- `codex_shim/tool_translate.py`: rewrite upstream MCP `function_call` items into
  passthrough shape (`namespace: "mcp__<server>"`, short tool name) so Codex
  CLI/Desktop executes MCP locally and `codex exec --json` shows `mcp_tool_call`
  parity with ChatGPT passthrough.
- Auto Router (`codex_shim/router.py`): an optional `Auto (smart routing)` picker
  entry (slug `codex-auto`) that routes each task to the cheapest configured
  model that can handle it. A cheap classifier model scores every candidate
  `0.0–1.0` from a capability card, the shim picks the cheapest candidate whose
  score clears `threshold` (default `0.7`), caches the decision per task, and
  falls back safely on any error. Configured via an optional `router` block in
  `~/.codex-shim/models.json`; gated in `/health`, `/v1/models`, `/api/models`,
  the generated catalog, and `codex-shim list`. Env knobs:
  `CODEX_SHIM_DISABLE_ROUTER`, `CODEX_SHIM_ROUTER_TIMEOUT`,
  `CODEX_SHIM_ROUTER_MAX_TOKENS`, `CODEX_SHIM_ROUTER_LOG`. Documented in
  `docs/AUTO_ROUTER.md` with a runnable offline proof at
  `examples/auto_router_demo.py` and 48 offline tests
  (`tests/test_router.py`, `tests/test_router_integration.py`) covering
  scoring/selection, streaming, compaction, the chat endpoint, the agent
  tool-loop cache, OpenAI/Anthropic classifiers, the exact classifier HTTP,
  fallbacks, availability gating, and concurrency.
- Cursor/Composer subscription passthrough for slug `composer-2-5`. When
  `cursor-agent login` is active, the shim spawns `cursor-agent --print` with
  CLI OAuth (no Dashboard API key). The slug is auth-gated in `/health`,
  `/v1/models`, and the generated catalog like ChatGPT passthrough.
- `POST /v1/responses/compact` support. ChatGPT passthrough forwards to the
  native ChatGPT compact endpoint; BYOK OpenAI/chat and Anthropic routes run a
  non-streaming compact summarization request and return a Responses-shaped
  compacted window for the next Codex turn.
- BYOK fallback schemas for native Responses-only tools: `computer_use`,
  `web_search`, `apply_patch`, and `local_shell` now translate into ordinary
  function tools for chat-completions / Anthropic providers instead of being
  dropped. Codex MCP function tools continue to pass through unchanged.
- Streaming `response.completed` events now include upstream `usage` when chat
  or Anthropic streams provide it, so Codex can track token counts and trigger
  auto-compaction.
- BYOK visual feedback passthrough for computer-use loops: Responses
  `input_image`, `computer_call_output` screenshots, and visual
  `function_call_output` payloads now reach OpenAI chat providers as
  `image_url` parts and Anthropic providers as image blocks.
- GitHub Actions CI (`.github/workflows/ci.yml`) running pytest and
  `compileall` on Python 3.11 and 3.12.
- `[project.optional-dependencies] dev` in `pyproject.toml` so
  `uv sync --extra dev` pulls `pytest` and `pytest-asyncio` in one step.
- `CONTRIBUTING.md` documenting the dev loop, what kinds of PRs are useful,
  and what to include in bug reports.
- `.github/ISSUE_TEMPLATE/` with structured bug and feature request templates.
- `CHANGELOG.md` (this file).
- Web-based model picker at `GET /picker` (with `GET /api/models` and
  `POST /api/switch`) so the active shim model can be swapped from a browser
  without the CLI. Switching rewrites the top-level `model = "..."` in
  `~/.codex/config.toml`. Optional auto-restart of Codex Desktop is
  cross-platform (`taskkill` + `Codex.exe` on Windows, `osascript` + `open -a
  Codex` on macOS). All picker routes are behind the existing `Host`-header
  allowlist, so a visited web page still cannot drive them via DNS rebinding.
- Responses WebSocket bridge for ChatGPT passthrough models. The shim accepts
  Codex `response.create` WebSocket frames on `/v1/responses`, forwards them to
  ChatGPT's native Responses endpoint, and relays upstream SSE events back as
  WebSocket JSON frames with model metadata rewritten to the selected shim slug.
  Non-ChatGPT models use the same WebSocket entrypoint but are internally
  bridged through the existing local HTTP `/v1/responses` route.
- zstd request/response compression support via `backports.zstd` on Python
  versions before 3.14. The managed config no longer disables Codex request
  compression, and ChatGPT passthrough requests advertise `zstd, gzip, deflate`.
- Best-effort dump of the last forwarded chat-completions request body to
  `.codex-shim/last_request.json` to make strict-provider tokenization /
  schema errors easier to triage. Upstream error bodies are now logged with
  the model slug before being forwarded back.

### Changed

- Reframed the project around a generic all-model Codex shim instead of any
  single upstream app or model store.
- Made `~/.codex-shim/models.json` the canonical default settings file.
- Renamed the generated Codex provider to `codex_shim` / "Codex Shim".
- Settings now prefer a generic top-level `models` array with snake_case keys,
  while still accepting `customModels` and camelCase aliases for existing
  exports.

### Fixed

- BYOK response translation now preserves native Responses tool item types where
  Codex expects them: `apply_patch` returns `custom_tool_call`, and
  `web_search*` returns `web_search_call`. MCP and `tool_search_call` handling
  remains client-executed; no server-side web-search fallback was added.

- Protected the state-changing picker `/api/switch` endpoint with a
  per-process picker token so third-party pages cannot trigger model switches
  or Desktop restarts through the loopback server.
- `codex-shim status` now treats a healthy `/health` endpoint as running even
  when the shim is managed by foreground/systemd `run` mode instead of a local
  `.codex-shim/shim.pid` file.
- `codex-shim enable` / `disable` now manage `tool_search_always_defer_mcp_tools`
  in a shim-owned `[features]` block (with restore of any prior user value).
  Ephemeral `codex-shim codex` / `app` runs apply the same override via CLI flags.
- MCP system hint for BYOK models now steers `tool_search_call` queries toward
  short tokens (`exa`, `web_search_exa`) instead of `mcp__`-prefixed strings that
  often return empty BM25 results.
- BYOK catalog entries now set `supports_search_tool: true` (matching ChatGPT
  passthrough models) so Codex builds the client-side `tool_search` BM25 index
  instead of returning empty `tool_search_output.tools` for shim-routed models.
- Shim injects `tool_search_call` for upstream BYOK models when Codex sends the
  native `tool_search` tool (deferred MCP mode), not only bare `mcp__*` stubs.
- Removed MCP `tools/list` pre-injection; discovery is via Codex-native
  `tool_search` only.

- Anthropic route requests now send only `x-api-key` (plus `anthropic-version`)
  for authentication and no longer also attach `Authorization: Bearer <apiKey>`.
  Some Anthropic-compatible gateways reject requests that carry both headers.
  Providers that genuinely require a bearer token can still supply one via
  `extraHeaders`.
- `codex-shim patch-app` now also patches the Codex Desktop sidebar's recent
  thread loader so native `openai` chats remain visible while Desktop is routed
  through any custom-provider shim config. Tested on Codex Desktop 26.519.41501 /
  `codex-cli 0.133.0-alpha.1` on macOS arm64.
- `patch-app` now updates `ElectronAsarIntegrity` in `Info.plist` after
  repacking `app.asar`, and `restore-app` restores or recomputes that metadata
  before re-signing the app bundle.

## 2026-05-25 — Auth-gated ChatGPT passthrough + docs hardening

### Added

- `settings.chatgpt_passthrough_available()` checks `~/.codex/auth.json` for a
  usable `tokens.access_token`. The synthetic `gpt-5.5` slug is now only
  advertised in `/health`, `/v1/models`, `codex-shim list`, and the generated
  `custom_model_catalog.json` while that token is present.
- `_load_models()` in the CLI wraps model settings loading with actionable
  errors for missing files and invalid JSON.
- `_entrypoint()` in the CLI catches `BrokenPipeError` at the boundary so
  piping `codex-shim list` into `head`/`grep` exits cleanly instead of dumping
  a traceback.
- Regression tests covering auth-gating, CLI error UX, settings aliases, and
  catalog generation.

### Changed

- `/health` payload now includes `chatgpt_passthrough: bool` and reports the
  real model count instead of always-plus-one.
- `cli._resolve_model_slug("gpt-5.5", ...)` raises `SystemExit` telling the
  user to run `codex login` when auth.json is missing, instead of returning a
  slug that would 401 on first request.
- `default_model_slug` picks the first configured BYOK model when passthrough
  is not usable, instead of unconditionally returning `gpt-5.5`.
- README install section recommends `pip install -e .` as the primary path.
- README benchmarking section: replaced an unsupported "7x fewer input tokens
  / 5–10x faster" claim with honest anecdata and a note that no reproducible
  benchmark script ships with the repo yet.

### Fixed

- Codex Desktop picker / `/v1/models` no longer offers `gpt-5.5` when there's
  no Codex login, removing the misleading "select it to get a 401" footgun.

## 2026-05-25 — Initial public hardening

### Added

- Public-grade README rewrite covering install, ChatGPT passthrough, tool
  calls, computer use, prompt catching/proxy patterns, benchmarking, security,
  limitations, troubleshooting, and contributing.
- `pyproject.toml` build-system, `readme`, `license`, `authors`, `keywords`,
  classifiers, and project URLs.

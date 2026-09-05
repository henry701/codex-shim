# codex-shim

> **Canonical upstream:** [0xSero/codex-shim](https://github.com/0xSero/codex-shim) is the
> original project and the reference implementation most users should start from.
>
> **This repository** ([henry701/codex-shim](https://github.com/henry701/codex-shim)) is a
> personal fork maintained for local Codex Desktop + BYOK workflows on my machines. It is
> **not** focused on upstreaming changes. Upstream (or other forks) are welcome to borrow
> ideas from here and reimplement them independently if they fit their goals.

Run **Codex Desktop** against any BYOK model you can describe in
`~/.codex-shim/models.json`, plus an optional passthrough to your **ChatGPT
subscription's Codex model** — without rebuilding Codex.

The shim is a local Python/aiohttp server that exposes an OpenAI
Responses-compatible endpoint on loopback. Codex points at the shim; the shim
routes each request to the matching upstream (OpenAI chat completions,
Anthropic Messages, a generic OpenAI-shaped chat endpoint, or ChatGPT Codex
passthrough), then translates streaming responses back into the shape Codex
expects.

> Tested on Codex Desktop **0.133.0-alpha.1** for macOS arm64 and **26.602.x** on Linux
> x64 (with an optional shim-picker rebuild). The shim server and routing layer are plain
> Python/aiohttp and work on Windows, macOS, Linux, WSL, and Git Bash. macOS and Linux
> Desktop builds may need an ASAR patch when Codex hides custom catalog entries.

---

## This fork (henry701)

Upstream covers the core shim, translation layer, ChatGPT/Cursor passthrough, Auto Router,
and the macOS picker patch. **This fork adds and hardens the pieces below** for a
multi-provider local catalog that stays in sync with Codex Desktop.

| Area | What the fork adds |
|---|---|
| **Auto-discovery** | `codex-shim discover` lists models pulled from provider APIs/CLIs: OpenCode Zen (keyless free via models.dev + paid), OpenRouter `:free` models, NVIDIA Integrate, Nous Portal (`stealth/ox-alpha` and `/v1/models`), and local OpenAI-compatible endpoints. Discovery copies useful models.dev fields into the Desktop catalog (context/output limits, reasoning efforts/variants, input modalities, descriptions). ChatGPT passthrough already maps the Codex backend `/models` row. Cursor stays CLI-driven. `discover --refresh` busts cached Cursor catalog metadata. |
| **Provider-prefixed slugs** | Discovered routes get stable prefixes (`or-`, `zen-`, `nvidia-`, `oc-free-`, `nous-`, …) so hundreds of models stay identifiable in the picker and logs. |
| **`sync-desktop`** | Writes `~/.codex/custom_model_catalog.json` only. Does **not** change `~/.codex/config.toml` — use `codex-shim enable` (or `app`) to wire OpenAI-provider shim routing. |
| **systemd user service** | `codex-shim install-service` installs a user unit with `ExecStartPre=sync-desktop` (45s discovery budget; keeps the existing catalog on timeout) and `ExecStart=serve`. `TimeoutStartSec=180`. After bind, `serve` refreshes the Desktop catalog **and** `~/.codex/models_cache.json` (15s, then every 3h) so ChatGPT passthrough rows and stale oc-free/NVIDIA slugs are not stuck at boot. `codex-shim restart` reloads and restarts the user unit when it is installed. CLI `run` still syncs then serves for interactive use. Config stays untouched until `enable`. Targets `graphical-session.target` and, when present, a local `network-ready-user.service` drop-in so model refresh waits for NM + DNS. Also installs an hourly user logrotate timer for `~/.codex-shim/shim.log` (30M, keep 10 compressed). |
| **Namespace tools (dot notation)** | Responses `type: "namespace"` tools (including `multi_agent_v1` / multi-agent V2) expand to `namespace.tool` on BYOK chat/anthropic routes and round-trip back to `namespace` + `name` on responses and streams. MCP refs accept `mcp__srv__tool` and `mcp__srv.tool`. |
| **`openai-responses` provider** | Raw passthrough to upstream `/v1/responses` (no chat-completions translation) for providers that speak the Responses API natively. |
| **BYOK agent-loop parity** | Codex-native `tool_search_call` / deferred MCP, namespaced MCP `function_call` items, streaming narration before tool calls, and fuller tool-output round-trips on BYOK routes (see changelog for the full fix list). |
| **Toolchain** | [uv](https://docs.astral.sh/uv/) for install, lockfile, dev deps, and CI (`uv sync`, `uv tool install -e .`, `uv run pytest`). |

**Typical fork workflow**

```bash
uv tool install -e .                    # or: uv sync && uv tool install -e .
codex-shim discover --refresh           # preview auto-discovered routes
codex-shim sync-desktop                 # refresh ~/.codex catalog (config unchanged)
codex-shim enable                       # wire OpenAI-provider shim routing into ~/.codex/config.toml + start daemon
codex-shim install-service              # optional: user systemd unit at graphical login
codex-shim install-logrotate            # optional: rotate ~/.codex-shim/shim.log at 30M (keep 10)
```

On Linux, pair with [CodexDesktop-Rebuild](https://github.com/henry701/CodexDesktop-Rebuild) and
`USE_SHIM_MODEL_PICKER=1` if Desktop's Statsig allowlist hides shim catalog entries.

### Catalog context headroom (`catalog_context`)

Tune Codex's advertised context window when the shim refreshes
`~/.codex/custom_model_catalog.json` (`sync-desktop` or in-process `serve`
refresh). Codex reads these limits from `model_catalog_json`, not from a
separate field in `~/.codex/config.toml`.

```json
{
  "catalog_context": {
    "$comment": "Tiers: chatgpt | cursor | byok | router — see table below. $comment keys are ignored.",
    "modifier": 0.9,
    "apply_to_tiers": ["chatgpt"],
    "override_patterns": {
      "codex-gpt-5-6-*": { "context_window": 240000 }
    }
  }
}
```

`override_patterns` beat `modifier`. GPT-5.6 models cap at 240K (under the API
272K long-context billing cliff); other ChatGPT passthrough models still get 0.9×.

| Field | Meaning |
| --- | --- |
| `modifier` | Multiply the upstream `context_window` (e.g. `0.9` → 10% headroom). Only for tiers in `apply_to_tiers`. |
| `apply_to_tiers` | Closed set of catalog discovery tiers (default: `["chatgpt"]`). Unknown names are dropped. |
| `override_patterns` | Glob slug overrides (fnmatch). Beat `modifier`. |
| `overrides` | Exact slug overrides. Beat patterns and `modifier`. |
| `$comment` / `$comments` | Documentation only; ignored when loading. |

**Catalog tiers** (assigned by the shim when building the catalog; not OpenAI plan tiers):

| Tier | Models | How they appear |
| --- | --- | --- |
| `chatgpt` | ChatGPT subscription passthrough (`codex-gpt-*`) | `codex login` + backend `/models` cache |
| `cursor` | Cursor subscription passthrough | Cursor auth bridge. **Usually omit from `apply_to_tiers`** — cursor-agent enforces its own context limits. |
| `byok` | Local / OpenRouter / Zen / NVIDIA / custom `models[]` | `discover` flags + `models[]`. Prefer per-model `max_context_limit` over the global modifier. |
| `router` | Shim auto-router entry | `router` block in `models.json` |

Priority: exact `overrides` > glob `override_patterns` > tier `modifier`. Slug
overrides apply across tiers; `modifier` is tier-gated. Applied fields:
`context_window`, `max_context_window`, `auto_compact_token_limit`, and
`truncation_policy.limit`.

Provider discovery:

```jsonc
{
  "discover": {
    "zen": true,
    "zen_public": true,
    "openrouter_free": true,
    "nvidia_integrate": true,
    "nous": true,
    "local": true
  },
  "models": [ /* local or niche routes only; cloud catalogs are auto-discovered */ ]
}
```

Set `"discover": false` to disable all auto-discovery, or toggle individual keys.

`zen_public` (slug prefix `oc-free-`) is the hermes-cli OpenCode Free path:
models from `https://models.dev/api.json` (provider `opencode`, `cost.input == 0`,
not deprecated) and inference at `https://opencode.ai/zen/v1` with **no**
`Authorization` header. The same models.dev row is mapped into the Desktop
catalog: context/output limits, `reasoning_options` effort values, input
modalities, and the upstream name/description. Nous Portal overlays the
OpenRouter models.dev row for `stealth/ox-alpha` (efforts `low`/`high`/`max`).
OpenRouter `:free` and NVIDIA Integrate do the same against their models.dev
providers. ChatGPT passthrough keeps the Codex backend `/models` payload,
including `max`/`ultra`. Cursor remains CLI-driven and is not scraped this way.
Free Zen 401s unrecognized bearers, including
`Bearer public`. The shim keeps `public` as an internal keyless sentinel so
the route still counts as credentialed; it is never sent upstream, and Desktop's
ChatGPT bearer is stripped so it cannot leak. Desktop's User-Agent is left
as-is. Hermes `HTTP-Referer` / `X-Title` are set only when Desktop omitted them.
`big-pickle` may 429 under a non-OpenCode-CLI User-Agent; that is expected.

Paid `zen` is unchanged and still sends `OPENCODE_API_KEY`.

`nous` (slug prefix `nous-`) is the Hermes Agent Nous Portal path:
`https://inference-api.nousresearch.com/v1`. Auth is `NOUS_API_KEY` (static
`sk-nous-…` key) or, if that env var is empty, the OAuth JWT in
`~/.hermes/auth.json` or `HERMES_SHARED_AUTH_DIR` / `~/.hermes/shared/nous_auth.json`.
`serve`, `sync-desktop`, and `discover` force-refresh that OAuth grant on startup
(Hermes device-code `/api/oauth/token` with `x-nous-refresh-token`) and
persist the rotated refresh token immediately — those tokens are single-use.
The shim updates `providers.nous` only and does not change Hermes
`active_provider`.
Hermes `HTTP-Referer` / `X-Title` are setdefaults; Desktop's User-Agent still
wins. Listing uses inference `/v1/models` and always includes `stealth/ox-alpha`.
Ox Alpha is advertised at 1,048,576 tokens (same window as OpenCode Zen
`x-preview-f-free`); Nous listings that omit `context_length` no longer fall
back to 128k.

---

## What this gives you

Codex Desktop only shows models allowed by its server-side config. If you have
OpenAI / Anthropic / Z.ai / DeepSeek / Gemini / OpenRouter / local proxy models
you want as first-class picker entries, this wires them in locally.

The practical win is that Codex keeps its native UX while model routing moves
local:

- **BYOK models in the normal Codex picker.** No Codex rebuild, no request
  replay workflow.
- **Native Codex agent loops stay intact.** Function calls, tool outputs,
  reasoning blocks, image-capable models, shell-command metadata, and streaming
  SSE are translated instead of flattened into plain text.
- **ChatGPT/Codex passthrough.** If `~/.codex/auth.json` has a valid Codex
  access token, the shim can route Codex's native `/v1/responses` traffic to
  ChatGPT's Codex backend under the `gpt-5.5` slug used by current Codex builds.
- **Cursor/Composer passthrough.** If `cursor-agent login` is active, the shim
  exposes `composer-2-5` and routes through your Cursor subscription — no
  Dashboard API key (`crsr_…`) required. See
  [`docs/subscription-integration.md`](docs/subscription-integration.md).
- **Auto Router (optional).** Add an `Auto (smart routing)` picker entry that
  uses a cheap classifier model to route each task to the cheapest configured
  model that can handle it — trivial turns stay cheap, hard turns escalate. See
  [`docs/AUTO_ROUTER.md`](docs/AUTO_ROUTER.md).
- **Prompt-catching/proxy-friendly architecture.** Put a local proxy in front
  of the shim to dedupe boilerplate, inject stable instructions, repair
  pseudo-tool text, or route prompts by policy before they hit an upstream.
- **Maintainer-side wins on real coding-agent runs.** In the maintainer's
  internal Codex tasks, ChatGPT passthrough plus a prompt-catching proxy in
  front of the shim has produced multi-x reductions in billed input tokens
  and noticeably faster wall time vs. the baseline route. No reproducible
  benchmark script ships with the repo yet, so treat that as anecdata — the
  benchmark section below explains how to measure your own setup against
  an explicit oracle before quoting numbers.

---

## Requirements

- Python 3.11+.
- Codex CLI/Desktop installed and authenticated.
- One of:
  - `~/.codex-shim/models.json` with configured BYOK/upstream models;
  - a compatible JSON file passed with `--settings`;
  - `~/.codex/auth.json` containing `tokens.access_token` for ChatGPT/Codex
    passthrough-only use.
- Windows: PowerShell/cmd works when installed via the Python package entry
  point; WSL or Git Bash is needed only for the optional `bin/` shell wrappers.
- macOS only: `npx` and `codesign` if you need the optional Desktop picker
  patch.

---

## Install

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) first.
uv manages a project virtualenv from `uv.lock`, so you do not need
`pip install` or PEP 668 workarounds on Arch and other externally managed
Python installs.

Recommended on macOS/Linux/WSL/Git Bash (installs the `codex-shim` entry
point from `pyproject.toml`):

```bash
git clone https://github.com/henry701/codex-shim ~/codex-shim
cd ~/codex-shim
uv sync
uv tool install -e .
```

For the canonical upstream tree instead:

```bash
git clone https://github.com/0xSero/codex-shim ~/codex-shim
```

Recommended on native Windows PowerShell/cmd:

```powershell
git clone https://github.com/henry701/codex-shim $HOME\codex-shim
cd $HOME\codex-shim
uv sync
uv tool install -e .
```

That pulls in `aiohttp` and installs the portable Python console command
`codex-shim`. On POSIX-like shells, the optional `codex-app` and `codex-model`
shortcuts live in `bin/`; symlink them if you want them on `PATH` too:

```bash
mkdir -p ~/.local/bin
ln -sf "$PWD/bin/codex-app" ~/.local/bin/codex-app
ln -sf "$PWD/bin/codex-model" ~/.local/bin/codex-model
```

If you move the checkout, recreate those symlinks; `codex-shim app` launches
`codex app` through the installed Python entry point and does not need them.

Alternative on macOS/Linux/WSL/Git Bash (no global install, run from the
checkout via uv):

```bash
git clone https://github.com/henry701/codex-shim ~/codex-shim
cd ~/codex-shim
uv sync
uv run codex-shim generate
```

For running the test suite:

```bash
uv sync --extra dev
uv run pytest tests/ -q
```

If your POSIX shell cannot find the commands, make sure `~/.local/bin` is on
`PATH`:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

If PowerShell cannot find `codex-shim`, add your Python user Scripts directory
to `Path`. For Python 3.11 installed from python.org, the usual path is:

```powershell
$env:APPDATA\Python\Python311\Scripts
```

You can also skip `PATH` entirely and run through Python:

```powershell
py -3.11 -m codex_shim.cli status
```

---

## Windows support

Yes, the shim works on Windows. The core shim is Python/aiohttp, binds to
`127.0.0.1`, and writes the same Codex provider config that macOS/Linux use.
Use one of these setups:

| Setup | Status | Notes |
|---|---|---|
| Native Windows PowerShell/cmd | Supported | Install with `uv sync` and `uv tool install -e .`, then run `codex-shim ...`. |
| WSL | Supported | Works like Linux. Best when Codex CLI/Desktop is also being driven from WSL. |
| Git Bash | Supported | Works with the POSIX `bin/` wrappers if Python/Codex are on `PATH`. |
| `bin/codex-app`, `bin/codex-model` in PowerShell/cmd | Not native | These are shell scripts. Use `codex-shim app ...` and `codex-shim model ...` instead. |
| `patch-app` / `restore-app` | macOS only | They target `/Applications/Codex.app` and Electron ASAR signing on macOS. |

Native Windows quick check:

```powershell
uv sync
uv tool install -e .
codex-shim generate
codex-shim start
codex-shim status
codex-shim list
```

If `codex-shim` is not on `Path`, use the module form:

```powershell
py -3.11 -m codex_shim.cli generate
py -3.11 -m codex_shim.cli start
py -3.11 -m codex_shim.cli status
```

Path behavior is intentionally ordinary:

- In native Windows, `~/.codex-shim/models.json` means
  `%USERPROFILE%\.codex-shim\models.json` and Codex config lives under
  `%USERPROFILE%\.codex\config.toml`.
- In WSL, `~/.codex-shim/models.json` and `~/.codex/config.toml` are inside the
  Linux home directory unless you explicitly point `--settings` at a Windows
  path under `/mnt/c/...`.
- Do not mix a WSL-generated `~/.codex/config.toml` with native Windows Codex
  and expect both to share files automatically. If Codex is native Windows, run
  the native Windows install path or manually keep the Windows config in sync.
- The local provider URL is still `http://127.0.0.1:8765/v1`.

The optional macOS picker patch is not required for the shim server to work. On
Windows, if Codex can read the generated catalog/provider config, requests route
through the same local endpoint as every other platform.

Windows Store/MSIX Codex Desktop builds are stricter than the CLI. They may treat
custom local/BYOK slugs as unavailable, rewrite `model = "<custom-slug>"` back to
`gpt-5.5`, and add `[tui.model_availability_nux]` entries on launch. That is a
Desktop allowlist behavior, not a shim routing behavior: `codex exec`, the TUI,
and the shim endpoint still use the configured model slug. The macOS `patch-app`
helper does not apply to MSIX packages under `C:\\Program Files\\WindowsApps`.

If Windows has a system proxy such as Clash/V2Ray, make sure loopback bypasses it:

```powershell
setx NO_PROXY "127.0.0.1,localhost,::1"
setx no_proxy "127.0.0.1,localhost,::1"
```

`codex-shim codex -- ...` and `codex-shim app ...` add those loopback entries to
the launched process environment automatically; set them globally too if you run
`codex.exe` directly.

---

## Quick start

### 1. Generate the catalog and start the shim

```bash
codex-shim generate          # reads ~/.codex-shim/models.json if present
codex-shim start             # background daemon on 127.0.0.1:8765
codex-shim list              # show generated slugs and upstream routes
codex-shim status            # health probe + model count
```

Generated runtime files live under the repo-local `.codex-shim/` directory:

```text
.codex-shim/custom_model_catalog.json   # model picker catalog for Codex
.codex-shim/config.toml                  # opt-in Codex provider config
.codex-shim/shim.pid                     # daemon pid
.codex-shim/shim.log                     # stdout/stderr + request summaries
```

The server binds `127.0.0.1` by default. It is meant to be a local loopback
adapter, not an Internet-facing proxy.

### 2. Point Codex Desktop at it

```bash
codex-shim sync-desktop      # fork: refresh ~/.codex/custom_model_catalog.json only
codex-shim enable            # wire OpenAI-provider shim routing into ~/.codex/config.toml + start daemon
codex-shim app .             # launch Codex Desktop with the shim wired in
```

`sync-desktop` regenerates the catalog from `models.json` plus auto-discovered routes
and writes `~/.codex/custom_model_catalog.json`. It does **not** modify
`~/.codex/config.toml`; the systemd service can keep the catalog fresh without
changing the CLI's active provider. Run `codex-shim enable` or `codex-shim app`
to install managed `model_provider = "openai"` routing with `openai_base_url`
pointed at the local shim. This keeps Desktop sessions in the same provider
namespace across restarts and migrates legacy `codex_shim` thread rows to
`openai`. The route supports Responses WebSockets for ChatGPT passthrough models
and BYOK `openai-responses` providers: the shim opens an upstream WSS to
`wss://chatgpt.com/backend-api/codex/responses` or `{base_url}/responses` and
relays JSON frames bidirectionally (HTTP+SSE fallback when upstream WSS fails).
BYOK chat-completions and Anthropic slugs still bridge through the shim's local
HTTP `/v1/responses` route and relay SSE events back over the WebSocket. The
shim accepts zstd-compressed Codex request bodies. `app` also starts the
local daemon if needed and
launches Desktop. The previous config is backed up under
`.codex-shim/` and the managed block can be removed with:

```bash
codex-shim disable
```

After this, Codex Desktop sees every entry from `~/.codex-shim/models.json`,
plus ChatGPT passthrough models listed from
`chatgpt.com/backend-api/codex/models` when `~/.codex/auth.json` holds a valid
`tokens.access_token` (falls back to `~/.codex/models_cache.json` / hardcoded
slugs only if that listing fails).

If your Codex Desktop's model picker only shows `default` and refuses to render
the catalog entries, apply the macOS picker patch below.

### 3. Switch the active Desktop model

```bash
codex-model list
codex-model gpt-5.5          # or any other slug from `list`
codex-app                   # relaunch Codex with new default
```

`codex-model <slug>` is a shortcut for `codex-shim model use <slug>`. It writes
only the shim-managed block in `~/.codex/config.toml`.

### 4. Use the Codex CLI without writing config

For one-off CLI runs, use inline `-c` overrides instead of changing
`~/.codex/config.toml`:

```bash
codex-shim codex -- "inspect this repo and summarize the architecture"
```

---

## Custom config file

The shim defaults to `~/.codex-shim/models.json`. If that file is missing, the
shim still generates a catalog — and adds the `gpt-5.5` ChatGPT passthrough
entry only when `~/.codex/auth.json` contains a valid `tokens.access_token`.
You can point it at any compatible file:

```bash
codex-shim --settings /path/to/my-models.json generate
codex-shim --settings /path/to/my-models.json start
```

Recommended schema:

```json
{
  "models": [
    {
      "model": "gpt-5.5",
      "provider": "openai",
      "base_url": "https://api.openai.com/v1",
      "api_key": "sk-…",
      "display_name": "OpenAI GPT-5.5",
      "max_context_limit": 400000
    },
    {
      "model": "claude-opus-4-7-20251109",
      "provider": "anthropic",
      "base_url": "https://api.anthropic.com/v1",
      "api_key": "sk-ant-…",
      "display_name": "Claude Opus 4.7"
    },
    {
      "model": "deepseek-v4-pro",
      "provider": "anthropic",
      "base_url": "https://api.deepseek.com/anthropic",
      "api_key": "…",
      "display_name": "DeepSeek V4 Pro",
      "no_image_support": true
    }
  ]
}
```

The loader also accepts camelCase aliases (`baseUrl`, `apiKey`, `apiKeyEnv`,
`displayName`, `maxContextLimit`, `maxOutputTokens`, `noImageSupport`,
`extraHeaders`) and a legacy top-level `customModels` array, so existing model
config exports can be used directly.

The shim **never writes your API keys** into the generated catalog. Put literal
keys in your settings file or reference them with `api_key_env`; credentials
are resolved when requests are handled.

Supported `provider` values:

| provider | upstream API |
|---|---|
| `openai` | OpenAI `/v1/chat/completions` |
| `openai-responses` | OpenAI `/v1/responses` (passthrough; no chat translation) |
| `generic-chat-completion-api` | OpenAI-shaped chat completions |
| `anthropic` | Anthropic `/v1/messages` |

Discovery sets `openai-responses` when a host implements `/v1/responses` and
honors models.dev SDK npm (`honors_models_dev_sdk`) and the row is
`@ai-sdk/openai`. OpenCode Zen and OpenRouter do. Hosted NVIDIA Integrate
404s `/v1/responses`; Nous Portal does not serve it — those stay on chat.
`@ai-sdk/openai-compatible` stays on chat completions. BYOK openai-responses
passthrough coerces Codex `custom` grammar tools to functions and aligns JSON
Schema `required` with advertised properties (optional keys are dropped, not
forced) — that is host behavior, not a single model id.

Namespace tools (`type: "namespace"` in Responses requests, including
`multi_agent_v1` / multi-agent V2) are expanded to dotted chat names
(`namespace.tool`) on BYOK chat/anthropic routes and split back into
`namespace` + `name` on responses.

The shim also accepts Anthropic Messages requests at
`http://127.0.0.1:8765/v1/messages`. For `openai` and
`generic-chat-completion-api` models, it translates Messages requests to
OpenAI-shaped chat completions and converts responses back to Anthropic shape.
For `anthropic` models, it passes the request through to the upstream
`/messages` endpoint with the configured model name. The bridge supports text,
image inputs, basic function tools/tool results, and streaming SSE. Provider
features such as prompt caching, extended thinking signatures, files, and token
counting remain upstream-specific.

Useful model fields:

| field | behavior |
|---|---|
| `display_name` | Human-readable picker label. |
| `api_key_env` | Name of an environment variable that contains the upstream API key. |
| `max_context_limit` | Catalog context window and compaction limits. |
| `max_output_tokens` | Default max output when translating to Anthropic. |
| `no_image_support` | When true, catalog advertises text-only input. |
| `extra_headers` | Optional upstream headers merged into requests. |

### OpenCode Go

OpenCode Go adds and updates models over time. Refresh the local settings from
the live OpenCode Go catalog instead of copying a hard-coded model list:

```bash
export OPENCODE_GO_API_KEY="..."
codex-shim opencode-go refresh
codex-shim generate
codex-shim start
```

The refresh command calls `https://opencode.ai/zen/go/v1/models`, probes each
model through both `/chat/completions` and `/messages`, and writes `ocgo-*`
entries into `~/.codex-shim/models.json`. Models that work through chat
completions are configured as `generic-chat-completion-api`; models that only
work through Messages are configured as `anthropic`.

Use `--settings` to write a different file, `--api-key-env` to use a different
environment variable name, or `--prefer messages` if you want models that
support both routes to prefer Anthropic Messages:

```bash
codex-shim --settings /path/to/models.json opencode-go refresh --prefer messages
```

If you need a minimal manual fallback, add one model with the same key env:

```json
{
  "models": [
    {
      "slug": "ocgo-glm-5-1",
      "model": "glm-5.1",
      "display_name": "OpenCode Go GLM 5.1",
      "provider": "generic-chat-completion-api",
      "base_url": "https://opencode.ai/zen/go/v1",
      "api_key_env": "OPENCODE_GO_API_KEY"
    }
  ]
}
```

The current OpenCode Go model list and endpoint split are documented at
<https://opencode.ai/docs/go/>.

### Ollama / local OpenAI-compatible chat endpoints

Codex sends the Responses API. Ollama and many local servers expose
OpenAI-shaped `/v1/chat/completions` instead. Keep Codex pointed at the shim with
`wire_api = "responses"`; configure Ollama as `generic-chat-completion-api` so
the shim translates Responses ⇄ chat completions:

```json
{
  "models": [
    {
      "model": "llama3.2",
      "display_name": "Ollama Llama 3.2",
      "provider": "generic-chat-completion-api",
      "base_url": "http://127.0.0.1:11434/v1",
      "api_key": "ollama"
    }
  ]
}
```

`codex-shim --settings /path/to/ollama-launch-models.json generate` also accepts
launch-model style files with a top-level `launchModels` / `launch_models` array,
including bare strings. `provider: "ollama"` is normalized to
`generic-chat-completion-api` with `http://127.0.0.1:11434/v1` when no base URL
is supplied.

Repeated `codex-shim enable`, `codex-shim app`, and `codex-shim model use ...`
runs are idempotent: the shim-managed top-level keys,
legacy `[model_providers.codex_shim]` block, and managed `[features]` block are
removed before the new managed blocks are written, so duplicate profile/provider
keys should not accumulate.

Codex may make small background calls to OpenAI model slugs such as
`gpt-5.4-mini` for its own product behavior. Those calls are not Ollama routing
failures; use the shim request log to confirm the actual selected model for the
agent turn.

---

## Picker patch for Codex Desktop on macOS

Codex Desktop has a Statsig server-side allowlist (`use_hidden_models: true`)
that hides any model whose slug is not on a hardcoded list. Custom catalog
entries fall into the hidden bucket and never render in the picker.

A single-boolean ASAR patch flips the allowlist branch off so the picker only
checks the local `hidden` flag (which this catalog never sets). On recent
Codex Desktop builds, the patch also changes the local recent-thread loader
from `modelProviders: null` to `modelProviders: []` so the sidebar continues to
show existing native `openai` chats even if Desktop is routed through a custom
provider.

The combined patch has been tested on Codex Desktop **26.519.41501** /
`codex-cli 0.133.0-alpha.1` on macOS arm64.

> Back up `app.asar` and `Info.plist` before patching.

```bash
APP=/Applications/Codex.app
sudo cp -R "$APP" "$APP.unpatched-$(date +%Y%m%d-%H%M%S)"

# 1. Extract the ASAR
cd /tmp && rm -rf codex-asar-patch && mkdir codex-asar-patch && cd codex-asar-patch
npx --yes @electron/asar extract "$APP/Contents/Resources/app.asar" extracted

# 2. Patch the picker filter (single occurrence in tested builds)
PATCH_FILE=$(grep -RIl 'useHiddenModels' extracted/webview/assets/model-queries-*.js | head -n1)
sed -i.bak -E 's/let u=c\.useHiddenModels&&o!==`amazonBedrock`,d;/let u=!1,d;/' "$PATCH_FILE"
diff "$PATCH_FILE.bak" "$PATCH_FILE" || true
rm "$PATCH_FILE.bak"

# 3. Patch the sidebar recent-thread provider filter (single occurrence)
SIDEBAR_FILE=$(grep -RIl 'listRecentThreads' extracted/webview/assets/app-server-manager-signals-*.js | head -n1)
python3 - "$SIDEBAR_FILE" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()
old = "listRecentThreads({cursor:e,limit:t}){return this.params.requestClient.sendRequest(`thread/list`,{limit:t,cursor:e,sortKey:this.recentConversationSortKey,modelProviders:null,archived:!1,sourceKinds:ke})}"
new = "listRecentThreads({cursor:e,limit:t}){return this.params.requestClient.sendRequest(`thread/list`,{limit:t,cursor:e,sortKey:this.recentConversationSortKey,modelProviders:[],archived:!1,sourceKinds:ke})}"
if text.count(old) != 1:
    raise SystemExit("expected one sidebar provider filter occurrence")
path.write_text(text.replace(old, new, 1))
PY

# 4. Repack
npx --yes @electron/asar pack extracted app.asar.new
sudo cp app.asar.new "$APP/Contents/Resources/app.asar"
```

That alone can crash Codex on next launch with `EXC_BREAKPOINT`. Electron's
`ElectronAsarIntegrity` field in `Info.plist` is a SHA-256 of the **JSON
header** of the ASAR archive (not the whole file). Recompute it and re-sign:

```bash
# 5. Compute new header hash
HEADER_HASH=$(python3 - "$APP/Contents/Resources/app.asar" <<'PY'
import struct, hashlib, sys
with open(sys.argv[1], 'rb') as f:
    data_size, header_size, _, json_size = struct.unpack('<4I', f.read(16))
    header_json = f.read(json_size)
print(hashlib.sha256(header_json).hexdigest())
PY
)
echo "new header hash: $HEADER_HASH"

# 6. Patch Info.plist (replaces the hash for Resources/app.asar)
sudo /usr/libexec/PlistBuddy -c \
  "Set :ElectronAsarIntegrity:Resources/app.asar:hash $HEADER_HASH" \
  "$APP/Contents/Info.plist"

# 7. Ad-hoc re-sign
sudo codesign --force --deep --sign - "$APP"

# 8. Launch
open "$APP"
```

To roll back: `sudo rm -rf "$APP" && sudo mv "$APP.unpatched-…" "$APP"`.

The CLI also has helper commands for patching/restoring `app.asar` and the
matching ASAR integrity metadata:

```bash
codex-shim patch-app
codex-shim restore-app
```

If Codex still crashes after `patch-app`, restore with `codex-shim restore-app`
and re-check the manual patch needles against the installed Desktop build.

---

## ChatGPT/Codex passthrough

If `~/.codex/auth.json` exists and contains `tokens.access_token`, the shim
exposes a synthetic `gpt-5.5` catalog entry that proxies straight to:

```text
https://chatgpt.com/backend-api/codex/responses
```

The entry is **only** advertised in `/health`, `/v1/models`, `codex-shim list`,
and the generated `custom_model_catalog.json` while that token is present. Once
you `codex logout` or the file is missing, the slug stops appearing — so the
picker never shows an option that would 401 on first use. Run `codex login` to
mint a new token and the entry comes back automatically on the next
`codex-shim generate`.

The passthrough forwards Codex request headers wholesale (session/thread metadata,
`x-codex-*`, etc.), overrides only `Authorization` with your Codex access token,
rewrites the model slug to the upstream ChatGPT id, and strips
`previous_response_id` while replaying delta continuations from a session-scoped
JSON cache under `~/.codex-shim/chatgpt-conversations/` (override with
`CODEX_SHIM_CHATGPT_CONVERSATIONS_DIR`). The cache survives shim restarts; use one
shim instance per home directory. When Codex uses the Responses
WebSocket transport, the shim proxies to `wss://chatgpt.com/backend-api/codex/responses`
and relays JSON frames (with the same expansion and model rewrite). If upstream
WSS is unavailable, it falls back to the legacy HTTP+SSE path automatically.
Upstream response headers and usage (`cached_tokens` / prefix cache) are logged when
`CODEX_SHIM_UPSTREAM_HEADER_LOG=1` and relayed back to Codex where applicable.

Debug env knobs:

| Variable | Effect |
|----------|--------|
| `CODEX_SHIM_UPSTREAM_HEADER_LOG=1` | Log upstream response headers + usage |
| `CODEX_SHIM_PASSTHROUGH_TRACE=1` | Log forwarded client headers per ChatGPT request |
| `CODEX_SHIM_STREAM_LOG=1` | Log SSE/WS event types |
| `CODEX_SHIM_RETRY_ATTEMPTS` | HTTP transport retries for aiohttp POST and discovery GET (default: `3`). OAuth refresh stays 1 attempt — refresh tokens are single-use. |
| `CODEX_SHIM_RETRY_WAIT_BUDGET` | Extra seconds of Retry-After waits after `CODEX_SHIM_RETRY_ATTEMPTS` is exhausted (default: `30`). One-shot OAuth refresh stays 1 attempt. |
| `CODEX_SHIM_RETRY_BACKOFF_BASE` | Initial retry delay in seconds (default: `0.5`) |
| `CODEX_SHIM_RETRY_BACKOFF_FACTOR` | Exponential backoff multiplier (default: `2`) |
| `CODEX_SHIM_SSE_KEEPALIVE_INTERVAL` | Downstream SSE `{"type":"ping"}` interval in seconds while a stream is open (default: `15`) |
| `CODEX_SHIM_WS_PASSTHROUGH=0` | Force legacy HTTP+SSE upstream for ChatGPT/BYOK WS routes (default: on) |
| `CODEX_SHIM_CHATGPT_WS_FORCE_EXPAND=1` | Force cache expansion on ChatGPT Codex WS (default: native passthrough on reused upstream WS) |
| `CODEX_SHIM_CHATGPT_CONVERSATIONS_DIR` | Root for persisted expansion cache (default: `~/.codex-shim/chatgpt-conversations`) |
| `CODEX_SHIM_CHATGPT_CACHE_MAX_BYTES` | Global disk cap for expansion cache; coldest entries evicted LRU (default: `512M`) |
| `CODEX_SHIM_CHATGPT_CACHE_MAX_MEMORY_ENTRIES` | In-memory LRU cap for deserialized snapshots (default: `256`) |

BYOK OpenAI-chat streams that end without `stop`, `tool_calls`, or `[DONE]`
reconnect on the same Codex turn using assistant prefill: a trailing truncated
assistant message, with no user nudge, up to three times. A conclusive
`stop` / `tool_calls` still completes even when `[DONE]` is missing.

Live smoke test (alternate port, `codex exec` with tool call + cache check):

```bash
SMOKE_PORT=8766 bash scripts/smoke_chatgpt_passthrough.sh
```

**Two different caches:** ChatGPT **prefix cache** (`cached_tokens` in upstream usage) is
server-side and keyed by stable session/thread headers the shim forwards. The shim
**conversation cache** (1024 responses per session, global byte cap default 512M, in-memory LRU default 256, JSON on disk under
`~/.codex-shim/chatgpt-conversations/`) replays delta continuations when upstream cannot
accept native `previous_response_id`.

**Expansion policy (transport-aware):**

| Surface | Routes | Behavior |
|---------|--------|----------|
| HTTP | ChatGPT Codex, BYOK, Cursor, compaction | Always expand + strip `previous_response_id` |
| WS | BYOK `openai-responses` | Always expand |
| WS | ChatGPT Codex OAuth | Native `previous_response_id` on **reused** upstream connection; expand on new connect or upstream prev_id error |

Cache writes are unconditional so a Codex WS turn can be replayed over HTTP (compaction, fallback, model switch). Orphan tool-call synthesis runs on every route.

It binds stdin closed (`</dev/null`) so `codex exec` does not hang waiting for
stdin when stdout is captured.

It bypasses configured BYOK routes entirely and uses your ChatGPT subscription quota.

It is already in `.codex-shim/custom_model_catalog.json` after `codex-shim
generate`. Select `GPT-5.5` in the picker, or run:

```bash
codex-model gpt-5.5
```

Older local configs or notes may refer to `openai-gpt-5-5`; the server accepts
that prefix as an alias and routes it to the same passthrough.

---

## Cursor/Composer passthrough (subscription)

If `cursor-agent status` shows you are logged in, the shim exposes
**Composer 2.5** as slug `composer-2-5` and routes each request by spawning
`cursor-agent` with your CLI OAuth session — the same pattern
[Open Design](https://github.com/nexu-io/open-design) uses for Cursor Agent.

```bash
cursor-agent login
scripts/codex-shim-install-cursor-composer
codex-shim model use composer-2-5
codex-app
```

The install helper is optional; it regenerates the local catalog/config and
sets `composer-2-5` as the active model when `cursor-agent status` reports an
active login. Troubleshoot with `cursor-agent status` and `/health`, which
reports `cursor_passthrough: true` when the shim can expose Composer.

Do **not** configure Composer via `cursor-api.standardagents.ai` unless you
intentionally want Dashboard API-key billing (`crsr_…`). That path is BYOK,
not CLI subscription.

Tool and thinking activity from `cursor-agent` is rendered as **reasoning**
markdown in Codex Desktop (read/write/edit/delete/glob/grep/shell, etc.).
Unknown tools are shown as JSON-fenced blocks for transparency. See
[`docs/cursor-agent-tools.md`](docs/cursor-agent-tools.md) for the tool catalog,
capture scripts, and how to refresh fixtures.

When Codex sends tools in the request, the shim also appends a **bridge suffix**
so Composer can forward those tools back as real Codex `function_call` items via
a loopback curl API (Goals, `update_plan`, **sub-agent collaboration**
`spawn_agent` / `wait_agent`, etc.). See [`docs/cursor-bridge.md`](docs/cursor-bridge.md).

---

## How routing works

```text
Codex Desktop ── /v1/responses ──▶ codex-shim (127.0.0.1:8765)
                                     │
                                     ├── slug "gpt-5.5"
                                     │       └─▶ chatgpt.com/backend-api/codex/responses
                                     │           (Authorization: Bearer <auth.json access_token>)
                                     │
                                     ├── provider "openai" / "generic-…"
                                     │       └─▶ baseUrl/chat/completions
                                     │           (Authorization: Bearer apiKey)
                                     │
                                     └── provider "anthropic"
                                             └─▶ baseUrl/messages
                                                 (x-api-key: apiKey, anthropic-version: …)
```

The shim translates Codex's Responses-API request into the upstream's shape
(chat completions or Anthropic Messages) and translates the streamed reply back.
Extended-thinking blocks from Anthropic-shaped upstreams (Claude, DeepSeek,
GLM, etc.) round-trip through `reasoning.encrypted_content` items.

---

## Auto Router (smart routing)

Optionally add one extra picker entry — **`Auto (smart routing)`** (slug
`codex-auto`) — that chooses the right model *per task*: trivial turns go to a
cheap model, hard turns escalate to your strongest one. It runs entirely on the
models you already configure.

On each new task the shim asks a cheap **classifier** model you nominate to score
every candidate `0.0–1.0` (how likely it nails the task first try), reading a
short **capability card** per candidate. It then routes to the **cheapest
candidate whose score clears `threshold`** (default `0.7`), caches that decision
for the task's tool-call round-trips, and falls back safely on any error. The
classifier never sees price, so it can't be biased toward expensive models.

Turn it on by adding a `router` block to `~/.codex-shim/models.json`:

```jsonc
"router": {
  "enabled": true,
  "slug": "codex-auto",
  "classifier": "minimax-m3",        // slug of a cheap configured model
  "threshold": 0.7,
  "default": "minimax-m3",
  "cache": true,
  "candidates": [
    { "slug": "minimax-m3", "cost": 0.3, "supports_images": false,
      "card": "Cheap, fast. Single-file edits, codegen, simple refactors." },
    { "slug": "opus", "cost": 5.0, "supports_images": true,
      "card": "Frontier. Big multi-file refactors, hard debugging, images." }
  ]
}
```

Prove it end to end with no keys and no network:

```bash
python3 examples/auto_router_demo.py
```

It spins up a mock multi-backend server, starts the **real** shim with the router
on, and shows trivial→cheap, medium→mid, hard→strong, image→image-capable, and a
repeat served from cache. Full configuration, env knobs (`CODEX_SHIM_ROUTER_LOG`,
`CODEX_SHIM_DISABLE_ROUTER`, …), and failure behavior are in
[`docs/AUTO_ROUTER.md`](docs/AUTO_ROUTER.md).

---

## Tool calls and agent loops

Codex expects Responses-API output items. Most BYOK upstreams speak either
OpenAI chat completions or Anthropic Messages. The shim bridges the gap:

| Codex/Responses item | OpenAI-shaped upstream | Anthropic upstream |
|---|---|---|
| `tools: [{type: "function", ...}]` | `tools: [{type: "function", function: ...}]` | `tools: [{name, description, input_schema}]` |
| `function_call` output item | Chat `tool_calls[]` | `tool_use` content block |
| `custom_tool_call` / `apply_patch` history | Chat `tool_calls[]` (`apply_patch`) | `tool_use` content block |
| `function_call_output` input item | Chat `role: "tool"` message | `tool_result` user content block |
| `custom_tool_call_output` input item | Chat `role: "tool"` message | `tool_result` user content block |
| streamed argument deltas | `response.function_call_arguments.delta` | `response.function_call_arguments.delta` |
| parallel calls | Preserved via `parallel_tool_calls` where supported | Multiple `tool_use` blocks |

This is the piece that makes the shim useful for real Codex runs instead of only
text chat. A model can ask Codex to run tools, Codex sends the tool output back
through the shim, and the upstream model continues the same loop.

Native Responses-only tools now have BYOK fallbacks:

| Responses tool | Chat/Anthropic fallback |
|---|---|
| `computer_use` / `computer_use_preview` | `computer_use` function with `{action, x, y, text, ...}` |
| `web_search` / `web_search_preview` | `web_search` function with `{query, ...}` |
| `apply_patch` | `apply_patch` function with `{patch, ...}` |
| `local_shell` / `shell` | `local_shell` function with `{command, ...}` |
| Codex MCP functions | Passed through as normal function tools |
| MCP server tools (`mcp__<server>__<tool>`) | Rewritten to namespaced `function_call` (`namespace: "mcp__<server>"`, short tool name) for Codex-native MCP execution |

That keeps BYOK models inside the Codex agent loop even when the upstream API is
chat-completions or Anthropic Messages instead of native Responses. ChatGPT
passthrough remains the highest-fidelity path for first-party hosted tool item
shapes, but BYOK routes no longer drop those tools. Visual feedback is preserved
for vision-capable BYOK providers: Responses `input_image`, `computer_call_output`
screenshots, and visual `function_call_output` payloads become OpenAI chat
`image_url` parts or Anthropic image blocks instead of being flattened to text.

Known edge cases:

- BYOK native-tool fallbacks depend on the Codex client/harness recognizing and
  executing the fallback function call. The shim translates tool schemas and
  round-trips tool outputs; it does not execute computer, shell, patch, or MCP
  actions itself.
- Some OpenAI-compatible providers advertise tool calls but stream malformed
  JSON arguments. The shim preserves deltas; the provider still has to emit
  valid JSON by the end of the call.
- If a provider ignores `parallel_tool_calls`, Codex may still request one tool
  at a time. That is an upstream behavior, not a catalog issue.
- OpenCode Zen Console (GLM/Z.AI error `[1210]` / `[1214]`) rejects extra chat
  fields. The shim strips `reasoning_content`, `reasoning_effort`,
  `parallel_tool_calls`, image parts, and empty user turns for `oc-free-*` /
  discovered Zen routes, and retries the same Codex turn.

---

## Compaction

Codex can compact long sessions through compaction v2 (`compaction_trigger` on
`POST /v1/responses`) or legacy `POST /v1/responses/compact`.

**ChatGPT passthrough is a thin proxy.** Desktop owns remote compact v2: the
shim forwards `compaction_trigger` on `/v1/responses` to ChatGPT
`/backend-api/codex/responses` and maps `/v1/responses/compact` 1:1 to
`/codex/responses/compact`, returning upstream status (the compact endpoint
currently 404s; logs `chatgpt-compact upstream missing (known)` and doctor
records the same). It does not run the compaction orchestrator, summarization
fallback, or turn-level `passthrough_error_fallback` on those ChatGPT compact
requests. Probe without OAuth (expects 401 from the shim if `auth.json` is
missing, or 404 from ChatGPT when logged in):

```bash
curl -sS -o /dev/null -w "%{http_code}\n" \
  -X POST http://127.0.0.1:8765/v1/responses/compact \
  -H 'Content-Type: application/json' \
  -d '{"model":"codex-gpt-5-5","input":[]}'
``` It still rewrites ChatGPT-illegal fields: model slug, `store: false`,
Lite `reasoning.context=all_turns` / `parallel_tool_calls=false`, cache
expansion of `previous_response_id`, and stripping shim-opaque
`encrypted_content`. Client `service_tier` (Fast/`priority`) and token caps
pass through unless ChatGPT 400s on them. Legacy `/compact` still omits
`store` and `stream`.

Cursor, BYOK, and OpenCode-style custom models still share the
`codex_shim.compaction` orchestrator: prepare input (orphan sanitization,
optional tool-output rewrite, token-budgeted recent-*user*-turn exclusion for
summarization), native compact, then structured summarization fallback, then
optional BYOK tertiary fallback (`compaction.tertiary_fallback_slug` in
`models.json`). If native compact is missing (typical for chat-completions
endpoints such as Nous/Ox) and LLM summarization returns an empty or unusable
handoff, a deterministic local fallback still emits Goal / Progress / Files
from the raw items so the task is never dropped. Turn-level
`passthrough_error_fallback` is separate: it only applies when ChatGPT/Cursor
passthrough **turns** fail, not to compaction phases.

Summarization turn splitting follows OpenCode/Pi/Hermes: only real `role=user`
messages are turn boundaries (developer/system preamble is not). The last
`max_recent_user_prompts` user prompts (default 50) are kept in play; older
user turns are dropped from the summarizer (prior compaction summaries still
carry them). A last turn that overflows `preserve_recent_tokens` stays in the
summarizer. Recent user texts are quoted verbatim into the summarization
prompt — never paraphrased, but capped so a stale first goal cannot dominate.

ChatGPT passthrough expands compact-v2 requests from the conversation cache when
`previous_response_id` is set, and synthesizes missing `function_call` items for
detached tail tool outputs Codex sends without in-batch calls.

| route | native | summarization fallback |
|---|---|---|
| ChatGPT passthrough | Thin proxy: `/codex/responses` (v2) or `/codex/responses/compact` (legacy); no shim fallback | — |
| Cursor passthrough | Cursor agent passthrough | Same summarization prompt via Cursor passthrough |
| BYOK OpenAI Responses / chat / Anthropic | Provider compact request | Same route with structured summarization body |

Optional `compaction` block in `~/.codex-shim/models.json`:

```json
{
  "compaction": {
    "model": "gpt-5.4-mini",
    "fallback_enabled": true,
    "tail_turns": 2,
    "preserve_recent_tokens": 8000,
    "max_recent_user_prompts": 50,
    "tool_output_max_chars": 2000,
    "compaction_output_token_reserve": 16000,
    "prompt_cache_key_version": "v1",
    "tertiary_fallback_slug": "gpt-5.4-mini"
  }
}
```

`tertiary_fallback_slug` is independent of `passthrough_error_fallback`. When
native compact and summarization both fail, tertiary runs a BYOK compact on that
slug if route credentials are available. If that is also empty or too thin to
keep a recent user goal, the shim writes a deterministic local summary
instead of replacing hundreds of items with a 400-character stub.

`tail_turns` caps how many **user** turns may stay out of the summarizer
(default 2). `preserve_recent_tokens` (default 8000) is the OpenCode-style
token budget for that suffix. `max_recent_user_prompts` (default 50, `0` =
unlimited) is the recency window for verbatim quotes, usability checks, and
the summarization head. Developer messages do not consume `tail_turns` and
are kept even when older user turns fall out of the window.

Compaction input shaping uses the compaction model's context window minus an
output reserve. By default the reserve is 20000 tokens, or an explicit
`compaction_output_token_reserve`. The shim char-truncates tool outputs only when
estimated input exceeds that budget, then prunes oldest tool outputs if still
over budget. If the history remains over budget after both steps, compaction
continues with a warning.

`codex-shim doctor` prints compaction model, prompt version, and fallback status.

The BYOK path strips provider-hostile fields such as `stream` and `service_tier`
before forwarding when required. ChatGPT compact-v2 passthrough forwards client
`instructions`, `tools`, and `reasoning` when present, with Lite constraints
applied.

---

## Computer use, shell commands, images, and MCP

The generated catalog advertises the Codex-facing capabilities Codex needs to
run as an agent:

| catalog field | value |
|---|---|
| `shell_type` | `shell_command` |
| `apply_patch_tool_type` | `freeform` |
| `web_search_tool_type` | `text_and_image` |
| `supports_parallel_tool_calls` | `true` |
| `input_modalities` | `text,image` unless `noImageSupport: true` |
| `supports_image_detail_original` | disabled when `noImageSupport: true` |

What that means in practice:

- **Shell/file operations** are still executed by Codex Desktop/CLI. The shim
  only translates the model request and response stream.
- **Images/screenshots** can pass to providers that accept images. Responses
  `input_image` items, `computer_call_output` screenshots, and visual tool
  outputs are preserved as OpenAI chat `image_url` parts or Anthropic image
  blocks. Set `noImageSupport: true` for text-only upstreams so Codex does not
  send image content they cannot parse.
- **Computer-use/native hosted tools** use native Responses item types on the
  ChatGPT passthrough path. BYOK chat/Anthropic routes receive deterministic
  function-tool fallbacks (`computer_use`, `web_search`, `apply_patch`,
  `local_shell`) so they can stay in the same Codex tool loop.

Codex Desktop forwards three generic MCP tools to every model:

- `list_mcp_resources`
- `list_mcp_resource_templates`
- `read_mcp_resource`

It does **not** flatten individual MCP server tools into the function list.
That is a Codex client behavior, not a shim limitation. Shim-routed models
receive the same MCP tools as built-in OpenAI models. The model is expected to
call `list_mcp_resources` to discover what is available.

When a BYOK model emits a full MCP tool name (`mcp__exa__web_search_exa`, etc.),
the shim rewrites the streamed Responses item to match ChatGPT passthrough:
`function_call` with `namespace: "mcp__exa"` and `name: "web_search_exa"`.
Codex CLI/Desktop executes the MCP call locally (visible as `mcp_tool_call` in
`codex exec --json`); the shim does not invoke MCP servers for those invocations.

When a BYOK model calls the virtual `tool_search_call` discovery tool, the shim
rewrites it to the native `tool_search_call` Responses item (with `execution:
"client"`). Codex executes the local BM25 search index and returns
`tool_search_output` on the next turn. BYOK catalog entries set
`supports_search_tool: true` (matching ChatGPT passthrough models) so Codex
builds that index. `codex-shim enable` installs a managed `[features]` block with
`tool_search_always_defer_mcp_tools = true` so small MCP servers stay searchable.
The shim accepts Codex zstd-compressed request bodies when request compression is
enabled. `codex-shim disable` restores any prior values. Ephemeral
`codex-shim codex` / `app` runs apply the same override via CLI flags.

Discovery queries work best with short tokens (`exa`, `web_search_exa`); the shim
injects a system hint steering models away from `mcp__`-prefixed search strings
that often return empty BM25 results.

---

## Prompt catching and request interception

There are two useful interception layers:

### 1. Built-in request summaries

Every `/v1/responses` request is summarized into `.codex-shim/shim.log`. Use it
while debugging model routing, tool schemas, and prompt size:

```bash
tail -f .codex-shim/shim.log
```

The log is intentionally summary-level so it does not dump API keys or full
prompt bodies by default. `[req]` lines include `prompt_cache_key` and other
cache fields when present.

### Quota / usage dashboard

Summarize cache hit rate, long-context turns (>272K input tokens), model mix,
tool-loop volume, and estimated Codex credits from rotated logs:

```bash
codex-shim quota-report
codex-shim quota-report --since 2026-07-10 --until 2026-07-12
codex-shim quota-report --json
python scripts/quota_dashboard.py --log-dir ~/.codex-shim --since 2026-07-10
```

Reads `shim.log`, `shim.log.1.gz`, and other `shim.log*` files under
`~/.codex-shim`. Date filters use each file's modification time (individual log
lines usually have no timestamps). Gzip archives from `install-logrotate` are
supported automatically.

### 2. Local prompt-catching proxy in front of this shim

For deeper control, put a small local proxy in front of `codex-shim` and point
Codex at that proxy. That layer can inspect the full Responses request before
it reaches this shim, then forward to `http://127.0.0.1:8765/v1/responses`.

Common uses:

- inject a stable system/developer preamble;
- strip repeated boilerplate before it burns tokens;
- repair pseudo-tool text such as XML-ish `<invoke ...>` drafts into structured
  tool calls before Codex sees them;
- route some prompts to ChatGPT passthrough and others to BYOK models;
- redact or hash large file blobs in logs.

Minimal aiohttp forwarder shape:

```python
from aiohttp import ClientSession, web

UPSTREAM = "http://127.0.0.1:8765"

async def responses(request):
    body = await request.json()
    body = catch_prompt(body)          # mutate or record the Responses payload
    async with ClientSession() as s:
        async with s.post(f"{UPSTREAM}/v1/responses", json=body, headers=request.headers) as r:
            return web.Response(body=await r.read(), status=r.status, headers=r.headers)

def catch_prompt(body):
    # Keep this deterministic. Codex retries are much easier to debug when the
    # same input produces the same transformed payload.
    return body

app = web.Application()
app.router.add_post("/v1/responses", responses)
web.run_app(app, host="127.0.0.1", port=8766)
```

Then launch Codex with the shim provider URL set to `http://127.0.0.1:8766/v1`
instead of `8765`. Keep prompt catching outside `codex_shim/translate.py` unless
you want every BYOK route to share the same mutation policy.

---

## Benchmarking cost and speed

The right benchmark is an actual Codex task, not a synthetic hello-world
completion. Measure the same repository, prompt, model, and tool budget across
routes.

Suggested quick protocol:

1. Pick one real task that uses tools, e.g. "find the bug, edit the file, run
   the focused test".
2. Run it once through your baseline Codex route and once through `gpt-5.5`
   passthrough or your BYOK model.
3. Record wall time, request count, prompt tokens, output tokens, tool-call
   count, and final test result.
4. Compare only successful end-to-end runs.

Useful shell timing wrapper:

```bash
/usr/bin/time -f 'wall=%E cpu=%P max_rss_kb=%M' codex-shim codex -- "your task here"
```

The `--` separator is accepted and stripped by the wrapper. It is optional, but
it keeps task prompts that start with `-` from being parsed as wrapper flags.

A good report looks like:

```text
Oracle: same repo commit, same prompt, same focused test command
Baseline: 12 requests, 210k input tokens, 19k output tokens, 18m42s, test passed
Shim:      8 requests,  31k input tokens, 11k output tokens,  2m35s, test passed
Result:   6.8x fewer billed input tokens, 7.2x faster wall time
```

The exact multiplier depends on model, prompt catcher policy, repo size,
network path, and how often the agent calls tools.

---

## Commands

```text
codex-shim generate          regenerate catalog/config without starting daemon
codex-shim sync-desktop      fork: refresh ~/.codex catalog (config.toml unchanged)
codex-shim discover          fork: list auto-discoverable provider models
codex-shim discover --refresh
                            fork: refresh discovery caches (e.g. Cursor catalog)
codex-shim start             regenerate catalog and start local shim daemon
codex-shim serve             run HTTP server only (systemd ExecStart; no catalog sync)
codex-shim run               sync desktop catalog then run HTTP server (interactive)
codex-shim install-service   fork: install+enable user systemd unit
codex-shim install-logrotate fork: user logrotate for ~/.codex-shim/shim.log (30M/10)
codex-shim prune-chatgpt-cache [--target 0.8]
                            evict cold expansion-cache files down to a fraction of the byte cap
codex-shim enable            start daemon; write managed ~/.codex/config.toml (model, provider, feature flags)
codex-shim status            health check + model count
codex-shim doctor            read-only diagnostics for settings, daemon, passthrough, and Codex config
codex-shim stop              stop daemon
codex-shim disable           remove managed config block and stop daemon
codex-shim restart           stop, regenerate, and start daemon
codex-shim list              list generated slugs and upstream routes
codex-shim opencode-go refresh
                            refresh OpenCode Go models into the settings file
codex-shim model list        list slugs currently usable in the picker
codex-shim model use <slug>  set the Desktop default model in managed config
codex-shim codex -- <args>   exec `codex` CLI through inline shim overrides
codex-shim app [path]        launch Codex Desktop through managed shim config
codex-shim patch-app         patch macOS Codex Desktop picker allowlist
codex-shim restore-app       restore macOS app.asar from patch backup

codex-app [path]             shortcut for `codex-shim app`
codex-model [list|<slug>]    shortcut for `codex-shim model …`
```

Global flags:

- `--settings <path>`: used by catalog/model/start/app/codex/doctor flows.
- `--port <port>`: used by daemon/provider/doctor flows.

`patch-app` and `restore-app` always target `/Applications/Codex.app`, do not
use `--settings`, and exit with a clear error on Windows/Linux.

---

## Model picker (web UI)

The shim exposes a small browser UI for switching the active model without
restarting the CLI:

- `GET /picker` — self-contained HTML page (dark theme) listing every model
  the shim currently knows about, with the active one highlighted.
- `GET /api/models` — JSON list backing the picker.
- `POST /api/switch` — `{"slug": "...", "restart_codex": true|false}`. The
  shim rewrites the top-level `model = "..."` in `~/.codex/config.toml` and
  optionally relaunches Codex Desktop (`open -a Codex` on macOS, `taskkill` +
  `Codex.exe` on Windows). This
  state-changing endpoint requires the per-process
  `X-Codex-Shim-Picker-Token` header embedded in `/picker`.

All picker routes are behind the same `Host`-header allowlist as the rest of
the shim, so a visited web page cannot drive them via DNS rebinding. The
state-changing `/api/switch` endpoint also requires a per-process picker token,
so third-party pages cannot trigger model switches just because the loopback
server is reachable.

---

## Security and privacy

- The shim binds to `127.0.0.1` by default.
- The shim validates the `Host` header on every request and rejects anything
  that is not a loopback name (`127.0.0.1`, `localhost`, `::1`), the configured
  bind host, or an entry in `CODEX_SHIM_ALLOWED_HOSTS`. This blocks DNS-rebinding
  attacks where a web page you visit resolves its own domain to `127.0.0.1` and
  drives the shim with your credentials. If you deliberately bind to a
  non-loopback host, add the host(s) you reach it by to
  `CODEX_SHIM_ALLOWED_HOSTS` (comma-separated).
- The model picker protects its state-changing `/api/switch` endpoint with a
  per-process picker token, so cross-site pages cannot switch the active model
  or request a Desktop restart without loading the picker page.
- API keys stay in your settings file; the generated catalog does not contain
  them.
- Request logs are summary-level by default and avoid full prompt/API-key dumps.
- ChatGPT passthrough reads `~/.codex/auth.json` at request time and forwards
  the access token only to ChatGPT's Codex endpoint. Catalog generation uses the
  same token to call `chatgpt.com/backend-api/codex/models` (with retries) so
  new GPT models appear even when Codex `openai_base_url` points at the shim
  and `models_cache.json` is stale.
- If you put a prompt-catching proxy in front of the shim, that proxy controls
  what it logs. Redact or hash large/private prompt bodies there.

---

## Limitations

- Codex internals and model-picker bundles change. The ASAR patch is version
  sensitive by nature.
- The ChatGPT passthrough endpoint is the endpoint current Codex builds use; it
  may move or change shape in a future Codex release.
- BYOK providers vary wildly in tool-call quality. The shim translates shapes;
  it cannot make an upstream model reliably emit valid tool-call JSON.
- Hosted Responses-only tools are highest fidelity on the ChatGPT passthrough
  path. BYOK routes get normal function-tool translation.
- The `bin/codex-app` and `bin/codex-model` shortcuts are POSIX shell scripts.
  In native Windows shells, use the installed `codex-shim` command instead.

---

## Troubleshooting

### Shim will not start

```bash
codex-shim doctor
codex-shim status
codex-shim doctor
tail -n 80 .codex-shim/shim.log
```

`codex-shim doctor` prints a read-only diagnostics report grouped by section
(Python, dependencies, Codex CLI, settings, runtime files, daemon health,
passthrough availability, proxy bypass, and Codex config). It never writes
configuration, starts/stops the daemon, calls model providers, or prints API
keys/tokens. It exits 1 only when a hard `FAIL` is detected; warnings are meant
as local setup hints.

Common causes:

- Python is older than 3.11.
- `aiohttp` is not installed in the Python used by the wrapper.
- Port `8765` is already in use. Start on another port:

```bash
codex-shim --port 8766 restart
codex-shim --port 8766 app .
```

### Diagnose local wiring

Run:

```bash
codex-shim doctor
```

The doctor is read-only. For this fork, the expected enabled config is
`model_provider = "openai"` plus
`openai_base_url = "http://127.0.0.1:<port>/v1"`. A legacy
`[model_providers.codex_shim]` block is reported because Desktop sessions should
stay under Codex's built-in OpenAI provider namespace.

### `~/.codex-shim/models.json` is missing

That is fine for ChatGPT passthrough-only use, **provided** `~/.codex/auth.json`
has a valid `tokens.access_token`. In that case `codex-shim generate` writes a
catalog containing just `gpt-5.5`. If neither file is present, the catalog will
be empty and `codex-shim list` will exit non-zero with a hint to run
`codex login` or pass a compatible settings file:

```bash
codex-shim --settings /path/to/my-models.json generate
```

### `codex-shim list` exits 1 with "No models available"

You have neither configured models in `~/.codex-shim/models.json` nor a valid
Codex login. Pick one:

```bash
codex login                       # populate ~/.codex/auth.json
# or
codex-shim --settings /path/to/my-models.json list
```

### Codex shows only `default`

Run:

```bash
codex-shim generate
codex-shim model list
```

If the catalog contains your models but Desktop still hides them, apply the
macOS picker patch. On Windows Store/MSIX Desktop, the same allowlist can rewrite
the active model back to `gpt-5.5`; use `codex-shim codex -- ...` / Codex CLI for
BYOK routes, or a non-MSIX/Desktop build that can read the custom catalog without
rewriting the config.

### Windows proxy sends loopback traffic away from the shim

If `codex.exe` returns proxy/502 errors while the shim is healthy, a system proxy
may be intercepting `http://127.0.0.1:8765`. Set both uppercase and lowercase
bypass variables before launching Codex:

```powershell
$env:NO_PROXY = "127.0.0.1,localhost,::1"
$env:no_proxy = "127.0.0.1,localhost,::1"
```

`codex-shim app ...` and `codex-shim codex -- ...` set those entries for the
child process automatically.

### Model appears but requests 404

The selected slug is not in the current generated catalog. Regenerate after
editing `~/.codex-shim/models.json` or the file passed with `--settings`:

```bash
codex-shim generate
codex-model list
codex-model <slug>
```

### Upstream returns 401/403

The API key in your model settings file is wrong, expired, or missing a
provider-specific header. For ChatGPT passthrough, refresh Codex login so
`~/.codex/auth.json` contains a valid `tokens.access_token`.

### Tool calls turn into text

Use the ChatGPT passthrough path first to confirm Codex itself is sending tools.
If passthrough works but a BYOK route does not, the upstream probably lacks
native tool-call support or emits malformed streamed arguments. Check
`.codex-shim/shim.log` for the requested model and tool count.

### Images fail on a text-only model

Set `"noImageSupport": true` for that model in the settings file and regenerate
the catalog.

### Streaming hangs

Check whether the upstream streams correctly outside Codex. Then restart the
local daemon:

```bash
codex-shim restart
tail -f .codex-shim/shim.log
```

The server uses a long read timeout because real coding-agent turns can stream
for a while; a silent hang is usually upstream/network/provider behavior.

### macOS app crashes after patching

You repacked `app.asar` but did not update `ElectronAsarIntegrity` and re-sign,
or the patch hit the wrong JavaScript bundle. Restore and retry:

```bash
codex-shim restore-app
codex-shim patch-app
```

### Reset generated shim state

```bash
codex-shim stop
# Remove .codex-shim manually if you want a completely fresh generated state.
codex-shim generate
codex-shim start
```

---

## File layout

```text
codex_shim/             python source (server + cli + translation)
bin/codex-shim          main entrypoint
bin/codex-app           shortcut wrapping `codex-shim app`
bin/codex-model         shortcut wrapping `codex-shim model …`
.codex-shim/            generated catalog, config, logs, pid (gitignored)
tests/                  pytest suite
```

Config behavior:

- `codex-shim generate`, `start`, `stop`, `restart`, `list`, `status`, and
  `codex-shim codex -- ...` do not persistently modify `~/.codex/config.toml`.
- `codex-shim enable`, `codex-shim app`, and `codex-shim model use <slug>` write
  managed blocks to `~/.codex/config.toml`. The managed route uses
  `model_provider = "openai"` and `openai_base_url = "http://127.0.0.1:8765/v1"`
  so Codex Desktop keeps showing sessions under the same provider namespace.
  Legacy `codex_shim` rows in `~/.codex/state_*.sqlite` are migrated to
  `openai`. If existing top-level Codex model keys or managed `[features]` keys
  are displaced, the managed block records them so disable can restore those keys
  without reverting unrelated config edits.
- `codex-shim disable` removes the managed blocks, restores displaced top-level
  model keys and prior `[features]` values when present, and stops the daemon.

---

## Development checks

```bash
python3 -m pytest tests/
python3 -m compileall codex_shim/ -q
```

The tests cover settings/catalog generation, request translation, server
routing, and CLI settings-file UX. Add regression tests when changing
translation behavior; tool-call shape bugs are easy to miss by eyeballing
streams.

---

## Contributing

**Upstream:** send general-purpose shim improvements to
[0xSero/codex-shim](https://github.com/0xSero/codex-shim).

**This fork:** issues and PRs here are fine for fork-specific workflow (discovery,
`sync-desktop`, systemd, Linux Desktop packaging). There is no expectation that
fork-only features will be cherry-picked upstream.

Good contributions (either venue) include:

- new provider translation tests;
- captured stream fixtures for tricky tool-call/reasoning cases;
- compatibility notes for new Codex Desktop builds;
- safer picker patch detection for changed ASAR bundles;
- docs for known-good provider configs.

Before opening a PR, run the development checks above and include the Codex
Desktop/CLI version you tested.

---

## License

MIT — see `LICENSE`.

Codex Desktop is a trademark of OpenAI. This project is unaffiliated.

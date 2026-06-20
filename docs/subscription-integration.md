# Subscription Integration

This fork exposes subscription-backed models as normal Codex picker entries when
the local credentials exist. It does not store subscription tokens in the
generated catalog.

## ChatGPT/Codex

- Requires `~/.codex/auth.json` with `tokens.access_token`.
- Exposes `gpt-5.5` and compatible fallback slugs only while the token is
  present.
- Routes native `/v1/responses` traffic to ChatGPT's Codex backend and rewrites
  returned model metadata back to the selected shim slug.
- Uses this fork's managed Codex config shape: `model_provider = "openai"` plus
  `openai_base_url = "http://127.0.0.1:<port>/v1"`.

Useful checks:

```bash
codex login
codex-shim generate
codex-shim model list
codex-shim doctor
```

## Cursor/Composer

- Requires `cursor-agent login`.
- Discovers Cursor catalog entries dynamically; `discover --refresh` refreshes
  cached Cursor metadata.
- Exposes provider-prefixed subscription slugs, including `composer-2-5` when
  available.
- Routes by spawning `cursor-agent` with the CLI OAuth session. Do not configure
  `cursor-api.standardagents.ai` unless Dashboard API-key billing is intended.

Useful checks:

```bash
cursor-agent status
codex-shim discover --refresh
codex-shim model list
codex-shim doctor
```

## Troubleshooting

- Run `codex-shim doctor` first. It reports missing CLIs, missing auth, bad
  proxy bypass variables, daemon health, and legacy `codex_shim` provider
  config.
- If Desktop rewrites a custom slug to `gpt-5.5`, verify the generated catalog
  with `codex-shim model list`; on allowlisted Desktop builds, use the CLI route
  or a rebuilt Desktop picker.
- If loopback requests hit a system proxy, set both `NO_PROXY` and `no_proxy` to
  include `127.0.0.1,localhost,::1`.

#!/usr/bin/env bash
# Smoke-test OpenCode Free (zen_public / oc-free-*) through the shim.
# Never SMOKE_RESTART=1 on port 8765 (production Desktop shim).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export SMOKE_MODEL="${SMOKE_MODEL:-oc-free-ling-3-0-flash-fin-free}"
export SMOKE_PROMPT="${SMOKE_PROMPT:-Reply with the single word pong. Do not use tools.}"
export SMOKE_REASONING_EFFORT="${SMOKE_REASONING_EFFORT:-low}"
exec bash "${ROOT}/scripts/smoke_chatgpt_passthrough.sh"

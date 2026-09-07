#!/usr/bin/env bash
# Isolated CLI smoke: closed stdin, unique jsonl, timeout cap for in-shim 429 waits.
# Never target production 8765.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${1:?model}"
OUT="${2:?jsonl path}"
SECONDS_LIMIT="${3:-10800}"
PORT="${SMOKE_PORT:-8767}"
PROMPT="${SMOKE_PROMPT:-Reply with the single word pong. Do not use tools.}"
WORKDIR="${SMOKE_WORKDIR:-$ROOT}"

if [[ "${PORT}" == "8765" ]]; then
  echo "Refusing port 8765 (production Desktop shim)."
  exit 1
fi

export PATH="${ROOT}/.venv/bin:${PATH}"
: >"${OUT}"
# timeout's stdin is closed; exec inherits that. Do not use --foreground (TTY/pg quirks).
timeout --kill-after=15 "${SECONDS_LIMIT}" \
  codex exec \
    -m "${MODEL}" \
    -c "openai_base_url=\"http://127.0.0.1:${PORT}/v1\"" \
    -c "model_reasoning_effort=\"${SMOKE_REASONING_EFFORT:-low}\"" \
    -c "mcp_servers={}" \
    -C "${WORKDIR}" \
    -s danger-full-access \
    --dangerously-bypass-approvals-and-sandbox \
    --skip-git-repo-check \
    --json \
    "${PROMPT}" \
    < /dev/null >"${OUT}" 2>&1

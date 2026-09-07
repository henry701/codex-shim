#!/usr/bin/env bash
# Smoke-test ChatGPT passthrough: shim on alternate port + codex exec with trace logging.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${SMOKE_MODEL:-codex-gpt-5-6-luna}"
WORKDIR="${SMOKE_WORKDIR:-$ROOT}"
PROMPT="${SMOKE_PROMPT:-Read the first 3 lines of README.md with a shell command (head -n 3 README.md), then answer in one short sentence what the project is.}"
DEFAULT_SHIM_LOG="${HOME}/.codex-shim/shim.log"
if [[ -f "${ROOT}/.codex-shim/shim.log" ]]; then
  DEFAULT_SHIM_LOG="${ROOT}/.codex-shim/shim.log"
fi

export CODEX_SHIM_UPSTREAM_HEADER_LOG=1
export CODEX_SHIM_STREAM_LOG=1
export CODEX_SHIM_PASSTHROUGH_TRACE=1

_shim_healthy() {
  curl -sf --max-time 1 "http://127.0.0.1:${1}/health" >/dev/null 2>&1
}

_port_open() {
  python3 -c 'import socket,sys; s=socket.socket(); s.settimeout(0.3); raise SystemExit(0 if s.connect_ex(("127.0.0.1", int(sys.argv[1])))==0 else 1)' "$1"
}

_pick_port() {
  local candidate
  for candidate in 8766 8767 8768 8769 8770; do
    if _shim_healthy "${candidate}"; then
      echo "${candidate}"
      return 0
    fi
  done
  for candidate in 8766 8767 8768 8769 8770; do
    if ! _port_open "${candidate}"; then
      echo "${candidate}"
      return 0
    fi
  done
  return 1
}

if [[ -n "${SMOKE_PORT:-}" ]]; then
  PORT="${SMOKE_PORT}"
  if ! _shim_healthy "${PORT}" && _port_open "${PORT}"; then
    echo "Port ${PORT} is occupied and is not a shim. Set SMOKE_PORT to a free port."
    exit 1
  fi
else
  PORT="$(_pick_port)" || {
    echo "No free shim smoke port in 8766-8770 (8765 is production; never use it)."
    exit 1
  }
fi

LOG="${SMOKE_LOG:-/tmp/codex-shim-smoke-${PORT}.log}"

SHIM_PID=""
cleanup() {
  if [[ -n "${SHIM_PID}" ]] && kill -0 "${SHIM_PID}" 2>/dev/null; then
    kill "${SHIM_PID}" 2>/dev/null || true
    wait "${SHIM_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

STARTED_BY_SCRIPT=0
if _shim_healthy "${PORT}"; then
  if [[ "${SMOKE_RESTART:-0}" == "1" ]]; then
    if [[ "${PORT}" == "8765" ]]; then
      echo "Refusing SMOKE_RESTART=1 on port 8765 (production Desktop shim)."
      exit 1
    fi
    echo "Restarting shim on port ${PORT} with trace env (log: ${LOG})"
    codex-shim --port "${PORT}" restart >>"${LOG}" 2>&1 || true
    sleep 1
  else
    echo "Shim already listening on ${PORT}; reusing it."
    echo "Trace env applies only if that process was started with them."
    echo "Set SMOKE_RESTART=1 to restart with logging, or use an alternate SMOKE_PORT."
    if [[ "${LOG}" == /tmp/codex-shim-smoke-"${PORT}".log ]]; then
      if [[ -f "/tmp/codex-shim-${PORT}.log" ]]; then
        LOG="/tmp/codex-shim-${PORT}.log"
      elif [[ "${PORT}" == "8765" ]]; then
        LOG="${DEFAULT_SHIM_LOG}"
      fi
    fi
  fi
else
  echo "Starting shim on port ${PORT} (log: ${LOG})"
  : >"${LOG}"
  # `serve` only: do not `run`/`sync-desktop` (mutates Desktop catalog) or `restart`
  # (systemd restart is the production 8765 unit).
  LOAD_ENV="${HOME}/.codex-shim/load-env.sh"
  SETTINGS="${HOME}/.codex-shim/models.json"
  if [[ -f "${LOAD_ENV}" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "${LOAD_ENV}"
    set +a
  fi
  SERVE_CMD=(codex-shim --port "${PORT}" serve)
  if [[ -f "${SETTINGS}" ]]; then
    SERVE_CMD=(codex-shim --settings "${SETTINGS}" --port "${PORT}" serve)
  fi
  "${SERVE_CMD[@]}" >>"${LOG}" 2>&1 &
  SHIM_PID=$!
  STARTED_BY_SCRIPT=1
  for _ in $(seq 1 30); do
    if _shim_healthy "${PORT}"; then
      break
    fi
    sleep 0.5
  done
  _shim_healthy "${PORT}" || {
    echo "Shim failed to start; tail ${LOG}:"
    tail -40 "${LOG}" || true
    exit 1
  }
fi

MARKER="smoke-${PORT}-$(date +%s)"
echo "${MARKER}" >>"${LOG}" 2>/dev/null || true

JSON_OUT="/tmp/codex-smoke-exec-${PORT}.jsonl"
: >"${JSON_OUT}"

echo "Running codex exec via http://127.0.0.1:${PORT}/v1 model=${MODEL}"
echo "Prompt: ${PROMPT}"

set +e
# Close stdin explicitly: piping/TTY detection otherwise makes codex wait for more input.
# Optional SMOKE_TIMEOUT (seconds): wrap with timeout --kill-after so a 429 absorb can
# finish without hanging forever when the caller sets a cap.
EXEC_ARGS=(
  codex exec
  -m "${MODEL}"
  -c "openai_base_url=\"http://127.0.0.1:${PORT}/v1\""
  -c "model_reasoning_effort=\"${SMOKE_REASONING_EFFORT:-max}\""
  -c "mcp_servers={}"
  -C "${WORKDIR}"
  -s danger-full-access
  --dangerously-bypass-approvals-and-sandbox
  --skip-git-repo-check
  --json
  "${PROMPT}"
)
if [[ -n "${SMOKE_TIMEOUT:-}" && "${SMOKE_TIMEOUT}" != "0" ]]; then
  timeout --kill-after=15 "${SMOKE_TIMEOUT}" "${EXEC_ARGS[@]}" </dev/null >"${JSON_OUT}" 2>&1
  EXEC_RC=$?
else
  "${EXEC_ARGS[@]}" </dev/null >"${JSON_OUT}" 2>&1
  EXEC_RC=$?
fi
set -e

echo
echo "=== codex exec output (tail) ==="
tail -20 "${JSON_OUT}" || true

echo
echo "=== Shim trace summary (${LOG}) ==="
if [[ -f "${LOG}" ]]; then
  rg -n "${MARKER}|\[req\]|\[chatgpt-trace\]|\[upstream-headers\]|\[chatgpt-cache\]|\[ws-passthrough\]" "${LOG}" 2>/dev/null \
    | awk -v m="${MARKER}" 'found {print} $0 ~ m {found=1}' | tail -80 || true
  if rg -q '\[ws-passthrough\] connected upstream' "${LOG}" 2>/dev/null; then
    echo "(upstream transport: WebSocket passthrough)"
  elif rg -q '\[ws-passthrough\] http-fallback' "${LOG}" 2>/dev/null; then
    echo "(upstream transport: HTTP+SSE fallback)"
  fi
else
  echo "(log file not found)"
fi

echo
echo "=== Usage / cache from codex JSONL ==="
rg -n 'usage|cached|cache_read|input_tokens|cacheRead' "${JSON_OUT}" 2>/dev/null | tail -30 || true

echo
echo "codex exec exit=${EXEC_RC}"
if [[ "${STARTED_BY_SCRIPT}" == "1" ]]; then
  echo "Shim started by this script on port ${PORT} (pid ${SHIM_PID})."
fi
if ! rg -q '"type":"turn.completed"' "${JSON_OUT}" 2>/dev/null; then
  echo "smoke failed: no turn.completed in ${JSON_OUT}"
  if [[ "${EXEC_RC}" -eq 0 ]]; then
    EXEC_RC=1
  fi
fi
exit "${EXEC_RC}"

#!/usr/bin/env bash
# End-to-end smoke: Codex CLI → shim cursor passthrough → cursor-agent bridge → goal completion.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${SMOKE_PORT:-8767}"
WORKDIR="$(mktemp -d /tmp/codex-bridge-smoke-XXXXXX)"
LOG="/tmp/codex-shim-bridge-smoke-${PORT}.log"
JSON_OUT="/tmp/codex-bridge-smoke-exec-${PORT}.jsonl"
SESSION="codex-bridge-smoke"
MODEL="${SMOKE_MODEL:-cursor-composer-2-5}"
PROMPT_FILE="${WORKDIR}/prompt.txt"
DONE_MARKER="${WORKDIR}/.codex-smoke-done"

export CODEX_SHIM_UPSTREAM_HEADER_LOG=1
export CODEX_SHIM_STREAM_LOG=1
export CODEX_SHIM_PASSTHROUGH_TRACE=1
export CODEX_SHIM_CURSOR_BRIDGE=1

cat >"${PROMPT_FILE}" <<'EOF'
In this workspace only:
1) If Codex goal tools are available, call create_goal with objective "Write goal-result.txt containing exactly GOAL_DONE".
2) Use shell to write goal-result.txt containing exactly the single line: GOAL_DONE
3) If goal tools are available, call update_goal with {"status":"complete"} for that goal.
4) Reply with the word DONE when finished.
EOF

echo "WORKDIR=${WORKDIR}"
echo "PORT=${PORT}"
echo "MODEL=${MODEL}"

SHIM_PID=""
cleanup() {
  if [[ -n "${SHIM_PID}" ]] && kill -0 "${SHIM_PID}" 2>/dev/null; then
    kill "${SHIM_PID}" 2>/dev/null || true
    wait "${SHIM_PID}" 2>/dev/null || true
  fi
  if [[ "${SMOKE_KEEP_TMUX:-0}" != "1" ]] && tmux has-session -t "${SESSION}" 2>/dev/null; then
    tmux kill-session -t "${SESSION}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
  echo "Port ${PORT} already in use; attempting kill..."
  fuser -k "${PORT}/tcp" 2>/dev/null || true
  sleep 1
fi

: >"${LOG}"
codex-shim --port "${PORT}" run >>"${LOG}" 2>&1 &
SHIM_PID=$!
for _ in $(seq 1 40); do
  curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null && break
  sleep 0.5
done
curl -sf "http://127.0.0.1:${PORT}/health" | jq .

: >"${JSON_OUT}"
MARKER="bridge-smoke-${PORT}-$(date +%s)"
echo "${MARKER}" >>"${LOG}"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  tmux kill-session -t "${SESSION}"
fi

AGENT_SCRIPT="${WORKDIR}/run_codex.sh"
cat >"${AGENT_SCRIPT}" <<SCRIPT
#!/usr/bin/env bash
set -euo pipefail
cd "${WORKDIR}"
echo "[codex] workdir: ${WORKDIR}"
echo "[codex] model: ${MODEL}"
echo "[codex] shim: http://127.0.0.1:${PORT}/v1"
set +e
codex exec \\
  -m "${MODEL}" \\
  -c 'openai_base_url="http://127.0.0.1:${PORT}/v1"' \\
  -C "${WORKDIR}" \\
  -s danger-full-access \\
  --dangerously-bypass-approvals-and-sandbox \\
  --skip-git-repo-check \\
  --enable goals \\
  --json \\
  "\$(cat "${PROMPT_FILE}")" </dev/null | tee "${JSON_OUT}"
RC=\${PIPESTATUS[0]}
set -e
echo "[codex] exit=\${RC}"
echo "[codex] goal-result.txt:"
cat goal-result.txt 2>/dev/null || echo "(missing)"
touch "${DONE_MARKER}"
exit "\${RC}"
SCRIPT
chmod +x "${AGENT_SCRIPT}"

LOG_SCRIPT="${WORKDIR}/tail_shim.sh"
cat >"${LOG_SCRIPT}" <<SCRIPT
#!/usr/bin/env bash
echo "[log] tailing ${LOG}"
tail -F "${LOG}" 2>/dev/null | rg --line-buffered 'bridge|cursor|passthrough|function_call|update_goal|/invoke|err' || true
SCRIPT
chmod +x "${LOG_SCRIPT}"

tmux new-session -d -s "${SESSION}" -n smoke "${AGENT_SCRIPT}"
tmux split-window -h -t "${SESSION}:smoke" "${LOG_SCRIPT}"
tmux select-layout -t "${SESSION}:smoke" even-horizontal

if [[ "${SMOKE_ATTACH:-0}" == "1" ]]; then
  tmux attach -t "${SESSION}"
fi

echo "Waiting for codex exec to finish (max ${SMOKE_TIMEOUT_S:-600}s)..."
DEADLINE=$(($(date +%s) + ${SMOKE_TIMEOUT_S:-600}))
while [[ ! -f "${DONE_MARKER}" ]] && [[ $(date +%s) -lt ${DEADLINE} ]]; do
  sleep 2
done

if [[ ! -f "${DONE_MARKER}" ]]; then
  echo "TIMEOUT waiting for codex exec"
  tmux capture-pane -t "${SESSION}:smoke.0" -p -S -80 || true
  exit 124
fi

echo
echo "=== workdir ==="
ls -la "${WORKDIR}" || true
echo
echo "=== goal-result.txt ==="
cat "${WORKDIR}/goal-result.txt" 2>/dev/null || echo "(missing)"
echo
echo "=== jsonl hits (goals/bridge/function_call) ==="
rg -n 'function_call|update_goal|create_goal|goals|bridge|CODEX_SHIM' "${JSON_OUT}" 2>/dev/null | head -50 || true
echo
echo "=== shim log hits ==="
rg -n "${MARKER}|bridge|/_cursor_bridge|update_goal|function_call|cursor passthrough|err" "${LOG}" 2>/dev/null | tail -80 || true

PASS=0
if [[ -f "${WORKDIR}/goal-result.txt" ]] && grep -qx 'GOAL_DONE' "${WORKDIR}/goal-result.txt"; then
  echo "PASS: goal-result.txt contains GOAL_DONE"
  PASS=$((PASS + 1))
else
  echo "FAIL: goal-result.txt missing or wrong content"
fi

if rg -q 'update_goal|create_goal|"type":"function_call"' "${JSON_OUT}" 2>/dev/null; then
  echo "PASS: JSONL contains goal tool or function_call activity"
  PASS=$((PASS + 1))
else
  echo "FAIL: no goal/function_call evidence in JSONL"
fi

if rg -q '\[cursor-bridge\] invoke|/_cursor_bridge/v1/invoke' "${LOG}" 2>/dev/null; then
  echo "PASS: shim log shows bridge invoke"
  PASS=$((PASS + 1))
else
  echo "FAIL: no bridge invoke in shim log"
fi

echo "WORKDIR=${WORKDIR} (kept for inspection; set SMOKE_KEEP_TMUX=1 to retain tmux session)"
if [[ ${PASS} -ge 3 ]]; then
  exit 0
fi
exit 1

#!/usr/bin/env bash
# Smoke: a real cursor-agent must be adopted across a Codex tool round-trip.
#
# Codex only runs tool calls once the response stream ends, so the first turn is
# early-completed while the agent is still mid-thought. The follow-up turn must
# continue that same process rather than spawning a new one -- respawning rebuilds
# the prompt from cached history and makes the agent restart its plan.
#
# Drives an already-running shim (default: the systemd user service on 8765).
set -uo pipefail

PORT="${SMOKE_PORT:-8765}"
LOG="${SMOKE_SHIM_LOG:-$HOME/.codex-shim/shim.log}"
MODEL="${SMOKE_MODEL:-cursor-composer-2-5}"
WORKDIR="$(mktemp -d /tmp/codex-bridge-adopt-XXXXXX)"
trap 'rm -rf "${WORKDIR}"' EXIT

if ! curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
  echo "no shim listening on ${PORT}" >&2
  exit 1
fi

SHIM_PID="$(systemctl --user show codex-shim.service -p MainPID --value 2>/dev/null || echo "")"
BOUNDARY="$(wc -l < "${LOG}")"
echo "port=${PORT} model=${MODEL} log_boundary=${BOUNDARY}"

TOOLS='[{"type":"namespace","name":"goals","tools":[
 {"type":"function","name":"create_goal","description":"Create a goal",
  "parameters":{"type":"object","properties":{"objective":{"type":"string"}},"required":["objective"]}},
 {"type":"function","name":"update_goal","description":"Update goal status",
  "parameters":{"type":"object","properties":{"status":{"type":"string"}},"required":["status"]}}]}]'

PROMPT='Call the Codex tool create_goal with objective "smoke adoption check" using the bridge curl recipe, then wait for its result, then reply with the single word ADOPTED.'

jq -n --arg model "${MODEL}" --argjson tools "${TOOLS}" --arg p "${PROMPT}" \
  '{model:$model,stream:true,tools:$tools,input:[{role:"user",content:$p}]}' \
  > "${WORKDIR}/turn1.json"

echo "--- turn 1: spawn + early-complete ---"
timeout 180 curl -sS -N -X POST "http://127.0.0.1:${PORT}/v1/responses" \
  -H 'Content-Type: application/json' -H 'session-id: smoke-adopt' \
  -d @"${WORKDIR}/turn1.json" > "${WORKDIR}/turn1.sse" 2>&1

CALL_ID="$(grep -o '"call_id":"[^"]*"' "${WORKDIR}/turn1.sse" | head -1 | cut -d'"' -f4)"
if [[ -z "${CALL_ID}" ]]; then
  echo "RESULT=INCONCLUSIVE (agent never invoked a bridge tool)"
  exit 2
fi
echo "call_id=${CALL_ID}"

if [[ -n "${SHIM_PID}" ]]; then
  echo "agents_alive_after_turn1=$(pgrep -P "${SHIM_PID}" 2>/dev/null | wc -l)"
fi

jq -n --arg model "${MODEL}" --argjson tools "${TOOLS}" --arg cid "${CALL_ID}" \
  '{model:$model,stream:true,tools:$tools,
    input:[{type:"function_call_output",call_id:$cid,
            output:"{\"goal_id\":\"g-smoke-1\",\"ok\":true}"}]}' \
  > "${WORKDIR}/turn2.json"

echo "--- turn 2: must adopt the live agent ---"
timeout 180 curl -sS -N -X POST "http://127.0.0.1:${PORT}/v1/responses" \
  -H 'Content-Type: application/json' -H 'session-id: smoke-adopt' \
  -d @"${WORKDIR}/turn2.json" > "${WORKDIR}/turn2.sse" 2>&1

SINCE="$(tail -n +$((BOUNDARY + 1)) "${LOG}")"
echo "--- bridge log ---"
printf '%s\n' "${SINCE}" | grep -E 'cursor-bridge|\[err\]' | tail -20

ADOPTED="$(printf '%s\n' "${SINCE}" | grep -c 'cursor-bridge\] adopt')"
ERRORS="$(printf '%s\n' "${SINCE}" | grep -c '\[err\]')"
SPAWNS="$(printf '%s\n' "${SINCE}" | grep -c 'early-complete')"
echo "adopt_events=${ADOPTED} early_completes=${SPAWNS} errors=${ERRORS}"

if [[ "${ADOPTED}" -ge 1 && "${ERRORS}" -eq 0 ]]; then
  echo "RESULT=PASS"
  exit 0
fi
echo "RESULT=FAIL"
exit 1

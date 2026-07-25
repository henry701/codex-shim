#!/usr/bin/env bash
# Smoke: a real cursor-agent must be adopted across a Codex tool round-trip.
#
# Codex only runs tool calls once the response stream ends, so the first turn is
# early-completed while the agent is still mid-thought. The follow-up turn must
# continue that same process rather than spawning a new one -- respawning rebuilds
# the prompt from cached history and makes the agent restart its plan.
#
# Three follow-up shapes are exercised, because adoption must be decided by bridge
# state rather than by the shape of Codex's input:
#   delta  -- bare tool output (the usual previous_response_id continuation)
#   replay -- no previous_response_id, whole transcript inline (what Codex sends
#             after a compaction) plus an item type this shim has never seen
#   steer  -- tool output followed by new user text; must NOT adopt
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
echo "port=${PORT} model=${MODEL}"

TOOLS='[{"type":"namespace","name":"goals","tools":[
 {"type":"function","name":"create_goal","description":"Create a goal",
  "parameters":{"type":"object","properties":{"objective":{"type":"string"}},"required":["objective"]}},
 {"type":"function","name":"update_goal","description":"Update goal status",
  "parameters":{"type":"object","properties":{"status":{"type":"string"}},"required":["status"]}}]}]'

PROMPT='Call the Codex tool create_goal with objective "smoke adoption check" using the bridge curl recipe, then wait for its result, then reply with the single word ADOPTED.'
OUTPUT='{"goal_id":"g-smoke-1","ok":true}'

post_turn() {
  # $1 = session id, $2 = request json path, $3 = response path
  timeout 180 curl -sS -N -X POST "http://127.0.0.1:${PORT}/v1/responses" \
    -H 'Content-Type: application/json' -H "session-id: $1" \
    -d @"$2" > "$3" 2>&1
}

# Builds the follow-up input for a scenario. Only the wrapping differs; every shape
# carries the same single tool result.
followup_input() {
  local shape="$1" cid="$2"
  case "${shape}" in
    delta)
      jq -n --arg cid "${cid}" --arg out "${OUTPUT}" \
        '[{type:"function_call_output",call_id:$cid,output:$out}]'
      ;;
    replay)
      jq -n --arg cid "${cid}" --arg out "${OUTPUT}" --arg p "${PROMPT}" \
        '[{type:"message",role:"developer",content:[{type:"input_text",text:"be concise"}]},
          {type:"message",role:"user",content:[{type:"input_text",text:$p}]},
          {type:"compaction",summary:"earlier work summarized"},
          {type:"reasoning",summary:[{type:"summary_text",text:"planning"}]},
          {type:"some_future_codex_item",payload:"unknown to this shim"},
          {type:"function_call_output",call_id:$cid,output:$out}]'
      ;;
    steer)
      jq -n --arg cid "${cid}" \
        '[{type:"function_call_output",call_id:$cid,
           output:"{\"message\":\"Wait interrupted by new input.\"}"},
          {type:"message",role:"user",content:[{type:"input_text",text:"stop that and summarize instead"}]}]'
      ;;
  esac
}

run_scenario() {
  local shape="$1" expect="$2"
  local session="smoke-adopt-${shape}"
  local boundary
  boundary="$(wc -l < "${LOG}")"

  echo "--- ${shape}: spawn + early-complete ---"
  jq -n --arg model "${MODEL}" --argjson tools "${TOOLS}" --arg p "${PROMPT}" \
    '{model:$model,stream:true,tools:$tools,input:[{role:"user",content:$p}]}' \
    > "${WORKDIR}/${shape}-1.json"
  post_turn "${session}" "${WORKDIR}/${shape}-1.json" "${WORKDIR}/${shape}-1.sse"

  local cid
  cid="$(grep -o '"call_id":"[^"]*"' "${WORKDIR}/${shape}-1.sse" | head -1 | cut -d'"' -f4)"
  if [[ -z "${cid}" ]]; then
    echo "${shape}: INCONCLUSIVE (agent never invoked a bridge tool)"
    return 2
  fi
  echo "${shape}: call_id=${cid}"

  echo "--- ${shape}: follow-up (expect ${expect}) ---"
  jq -n --arg model "${MODEL}" --argjson tools "${TOOLS}" \
    --argjson input "$(followup_input "${shape}" "${cid}")" \
    '{model:$model,stream:true,tools:$tools,input:$input}' \
    > "${WORKDIR}/${shape}-2.json"
  post_turn "${session}" "${WORKDIR}/${shape}-2.json" "${WORKDIR}/${shape}-2.sse"

  local since adopted cancelled errors
  since="$(tail -n +$((boundary + 1)) "${LOG}")"
  printf '%s\n' "${since}" | grep -E 'cursor-bridge|\[err\]' | tail -12
  adopted="$(printf '%s\n' "${since}" | grep -c 'cursor-bridge\] adopt')"
  cancelled="$(printf '%s\n' "${since}" | grep -c 'cursor-bridge\] cancel')"
  errors="$(printf '%s\n' "${since}" | grep -c '\[err\]')"
  echo "${shape}: adopt=${adopted} cancel=${cancelled} errors=${errors}"

  if [[ "${errors}" -ne 0 ]]; then
    echo "${shape}: FAIL (shim errors)"
    return 1
  fi
  if [[ "${expect}" == "adopt" && "${adopted}" -ge 1 ]]; then
    echo "${shape}: PASS"
    return 0
  fi
  # A steer must leave no agent adopted and no agent stranded on wait.
  if [[ "${expect}" == "respawn" && "${adopted}" -eq 0 && "${cancelled}" -ge 1 ]]; then
    echo "${shape}: PASS"
    return 0
  fi
  echo "${shape}: FAIL (expected ${expect})"
  return 1
}

FAILED=0
for scenario in "delta:adopt" "replay:adopt" "steer:respawn"; do
  run_scenario "${scenario%%:*}" "${scenario##*:}" || FAILED=1
done

if [[ -n "${SHIM_PID}" ]]; then
  echo "agents_alive_at_exit=$(pgrep -P "${SHIM_PID}" 2>/dev/null | wc -l)"
fi

if [[ "${FAILED}" -eq 0 ]]; then
  echo "RESULT=PASS"
  exit 0
fi
echo "RESULT=FAIL"
exit 1

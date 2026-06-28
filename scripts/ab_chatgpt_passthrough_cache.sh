#!/usr/bin/env bash
# A/B: ChatGPT passthrough with shim conversation expansion ON vs OFF.
# Measures upstream cached_tokens and codex turn.completed usage per arm.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKDIR="${AB_WORKDIR:-$ROOT}"
MODEL="${AB_MODEL:-codex-gpt-5-5}"
PROMPT="${AB_PROMPT:-Read the first 3 lines of README.md with a shell command (head -n 3 README.md), then answer in one short sentence what the project is.}"
PORT_A="${AB_PORT_A:-8770}"
PORT_B="${AB_PORT_B:-8771}"
LOG_A="/tmp/codex-shim-ab-expand-on-${PORT_A}.log"
LOG_B="/tmp/codex-shim-ab-expand-off-${PORT_B}.log"
JSON_A="/tmp/codex-ab-expand-on-${PORT_A}.jsonl"
JSON_B="/tmp/codex-ab-expand-off-${PORT_B}.jsonl"

export CODEX_SHIM_UPSTREAM_HEADER_LOG=1
export CODEX_SHIM_STREAM_LOG=1

PIDS=()
cleanup() {
  for pid in "${PIDS[@]}"; do
    kill "${pid}" 2>/dev/null || true
    wait "${pid}" 2>/dev/null || true
  done
}
trap cleanup EXIT

start_shim() {
  local port="$1"
  local log="$2"
  local expand_env="$3"
  : >"${log}"
  (
    export CODEX_SHIM_CHATGPT_EXPAND_CONTINUATIONS="${expand_env}"
    cd "${ROOT}"
    exec codex-shim --port "${port}" run
  ) >>"${log}" 2>&1 &
  PIDS+=("$!")
  for _ in $(seq 1 40); do
    if curl -sf "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
      return 0
    fi
    sleep 0.25
  done
  echo "Shim on port ${port} failed to start; tail ${log}:" >&2
  tail -30 "${log}" >&2 || true
  return 1
}

run_codex() {
  local port="$1"
  local json_out="$2"
  : >"${json_out}"
  codex exec \
    -m "${MODEL}" \
    -c "openai_base_url=\"http://127.0.0.1:${port}/v1\"" \
    -C "${WORKDIR}" \
    -s danger-full-access \
    --dangerously-bypass-approvals-and-sandbox \
    --skip-git-repo-check \
    --json \
    "${PROMPT}" </dev/null >"${json_out}" 2>&1
}

summarize_log() {
  local log="$1"
  echo "--- upstream usage events ---"
  rg -o 'usage=\{[^}]+\}' "${log}" 2>/dev/null \
    | sed 's/usage=//' \
    | while read -r line; do
        python3 -c "
import json, sys
u = json.loads(sys.argv[1])
d = u.get('input_tokens_details') or {}
cached = d.get('cached_tokens') or u.get('_cached_tokens') or 0
inp = u.get('input_tokens') or 0
pct = (100.0 * cached / inp) if inp else 0.0
print(f'  input={inp} cached={cached} ({pct:.1f}%)')
" "${line}" 2>/dev/null || echo "  ${line}"
      done
  echo "--- chatgpt-cache lines ---"
  rg '\[chatgpt-cache\]' "${log}" 2>/dev/null || echo "  (none)"
  echo "--- HTTP errors ---"
  rg -n 'HTTP 4[0-9]{2}|Unsupported parameter|upstream HTTP' "${log}" 2>/dev/null | tail -10 || echo "  (none)"
}

summarize_jsonl() {
  local jsonl="$1"
  echo "--- turn.completed ---"
  rg 'turn\.completed' "${jsonl}" 2>/dev/null | tail -3 || echo "  (none)"
  echo "--- errors ---"
  rg '"type":"error"|"type": "error"' "${jsonl}" 2>/dev/null | tail -5 || echo "  (none)"
}

echo "=== A/B ChatGPT passthrough cache ==="
echo "Prompt: ${PROMPT}"
echo

echo ">>> Arm A: EXPAND_CONTINUATIONS=1 (default, port ${PORT_A})"
start_shim "${PORT_A}" "${LOG_A}" "1"
MARK_A="ab-a-$(date +%s)"
echo "${MARK_A}" >>"${LOG_A}"
RC_A=0
run_codex "${PORT_A}" "${JSON_A}" || RC_A=$?
echo "codex exec exit=${RC_A}"
summarize_log "${LOG_A}"
summarize_jsonl "${JSON_A}"
echo

echo ">>> Arm B: EXPAND_CONTINUATIONS=0 (native previous_response_id, port ${PORT_B})"
start_shim "${PORT_B}" "${LOG_B}" "0"
MARK_B="ab-b-$(date +%s)"
echo "${MARK_B}" >>"${LOG_B}"
RC_B=0
run_codex "${PORT_B}" "${JSON_B}" || RC_B=$?
echo "codex exec exit=${RC_B}"
summarize_log "${LOG_B}"
summarize_jsonl "${JSON_B}"
echo

echo "=== Summary ==="
printf "Arm A (expand ON):  exit=%s\n" "${RC_A}"
printf "Arm B (expand OFF): exit=%s\n" "${RC_B}"

python3 <<PY
import json, re, pathlib

def parse_usages(log_path):
    text = pathlib.Path(log_path).read_text(errors="replace")
    usages = []
    for m in re.finditer(r'usage=(\{.*?\})(?:\s|\$)', text):
        try:
            u = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        d = u.get("input_tokens_details") or {}
        cached = d.get("cached_tokens") or u.get("_cached_tokens") or 0
        inp = u.get("input_tokens") or 0
        usages.append({"input": inp, "cached": cached})
    return usages

def parse_turn(json_path):
    for line in pathlib.Path(json_path).read_text(errors="replace").splitlines():
        if "turn.completed" not in line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        usage = obj.get("usage") or {}
        inp = usage.get("input_tokens") or 0
        cached = usage.get("cached_input_tokens") or 0
        return {"input": inp, "cached": cached}
    return None

def fmt_arm(name, rc, log, jsonl):
    usages = parse_usages(log)
    turn = parse_turn(jsonl)
    print(f"{name}:")
    print(f"  exit={rc}")
    if usages:
        total_in = sum(u["input"] for u in usages)
        total_cached = sum(u["cached"] for u in usages)
        pct = 100.0 * total_cached / total_in if total_in else 0
        print(f"  upstream_calls={len(usages)} sum_input={total_in} sum_cached={total_cached} ({pct:.1f}%)")
        for i, u in enumerate(usages, 1):
            p = 100.0 * u["cached"] / u["input"] if u["input"] else 0
            print(f"    call {i}: input={u['input']} cached={u['cached']} ({p:.1f}%)")
    else:
        print("  upstream_calls=0")
    if turn:
        p = 100.0 * turn["cached"] / turn["input"] if turn["input"] else 0
        print(f"  turn.completed: input={turn['input']} cached_input={turn['cached']} ({p:.1f}%)")
    else:
        print("  turn.completed: (missing)")

fmt_arm("A expand ON", ${RC_A}, "${LOG_A}", "${JSON_A}")
print()
fmt_arm("B expand OFF", ${RC_B}, "${LOG_B}", "${JSON_B}")
PY

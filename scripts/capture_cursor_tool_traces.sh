#!/usr/bin/env bash
# Capture cursor-agent stream-json traces for tool translation fixtures.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${CURSOR_CAPTURE_MODEL:-composer-2.5}"
OUT_DIR="${CURSOR_CAPTURE_OUT:-${ROOT}/tests/fixtures/cursor_stream}"
CAPTURE_DIR="$(mktemp -d /tmp/codex-shim-cursor-capture-XXXXXX)"
TIMEOUT_SEC="${CURSOR_CAPTURE_TIMEOUT:-180}"

if ! command -v cursor-agent >/dev/null 2>&1; then
  echo "cursor-agent not found on PATH" >&2
  exit 1
fi
if ! cursor-agent status >/dev/null 2>&1; then
  echo "cursor-agent not logged in. Run: cursor-agent login" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"

run_capture() {
  local scenario="$1"
  local prompt="$2"
  local raw="${CAPTURE_DIR}/${scenario}.raw.ndjson"
  local out="${OUT_DIR}/${scenario}.ndjson"
  echo "=== capture: ${scenario} (model=${MODEL}) ==="
  echo "workspace: ${CAPTURE_DIR}"
  (
    cd "${CAPTURE_DIR}"
    timeout "${TIMEOUT_SEC}" cursor-agent \
      --print \
      --output-format stream-json \
      --stream-partial-output \
      --force \
      --trust \
      --workspace "${CAPTURE_DIR}" \
      --model "${MODEL}" \
      "${prompt}" \
      >"${raw}" 2>"${CAPTURE_DIR}/${scenario}.stderr.log" \
      || true
  )
  if [[ ! -s "${raw}" ]]; then
    echo "WARN: empty capture for ${scenario}; see ${CAPTURE_DIR}/${scenario}.stderr.log" >&2
    return 1
  fi
  python3 "${ROOT}/scripts/sanitize_cursor_ndjson.py" \
    --workdir "${CAPTURE_DIR}" \
    --input "${raw}" \
    --output "${out}"
  echo "wrote ${out} ($(wc -l <"${out}") lines)"
  python3 "${ROOT}/scripts/extract_cursor_tool_keys.py" "${out}"
}

# Seed workspace
echo 'initial' >"${CAPTURE_DIR}/marker.txt"
mkdir -p "${CAPTURE_DIR}/src" "${CAPTURE_DIR}/tmp"
echo 'alpha content' >"${CAPTURE_DIR}/src/a.txt"
echo 'UNIQUE_GREP_TOKEN_XYZ789' >"${CAPTURE_DIR}/src/findme.txt"
echo 'old content' >"${CAPTURE_DIR}/tmp/old.txt"
echo 'delete me' >"${CAPTURE_DIR}/tmp/to-delete.txt"

run_capture move_rename \
  "In this workspace only: move tmp/old.txt to tmp/new.txt (use mv or rename tool, not shell if possible), then read tmp/new.txt. Reply in one short sentence."

run_capture delete_file \
  "In this workspace only: delete tmp/to-delete.txt, then list tmp/ to confirm it is gone. Reply in one short sentence."

run_capture write_new \
  "In this workspace only: create a brand-new file brand-new-only.txt with exactly the line 'written-fresh' (use write, not edit). Reply in one short sentence."

run_capture list_glob \
  "In this workspace only: list all .txt files under this workspace (use list or glob, not shell). Reply with a short bullet list of paths."

run_capture grep_search \
  "In this workspace only: search the workspace for the exact string UNIQUE_GREP_TOKEN_XYZ789 and report which file contains it. Reply in one short sentence."

echo
echo "=== inventory (all captures) ==="
python3 "${ROOT}/scripts/extract_cursor_tool_keys.py" "${OUT_DIR}"/*.ndjson 2>/dev/null || true
echo
echo "Capture workspace (for inspection): ${CAPTURE_DIR}"
echo "Fixtures written under: ${OUT_DIR}"

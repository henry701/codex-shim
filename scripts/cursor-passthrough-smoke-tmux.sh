#!/usr/bin/env bash
# Visual smoke test for cursor-agent stream-json → shim parser visualization.
# Uses tmux: left pane runs cursor-agent, right pane replays parsed events live.
# (cmux is not required; install tmux if missing.)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION="codex-shim-cursor-smoke"
WORKDIR="$(mktemp -d /tmp/codex-shim-cursor-smoke-XXXXXX)"
PROMPT_FILE="${WORKDIR}/prompt.txt"
STREAM_FILE="${WORKDIR}/stream.ndjson"
MODEL="${CURSOR_SMOKE_MODEL:-composer-2.5}"

cat >"${PROMPT_FILE}" <<'EOF'
In this workspace only:
1) run `sleep 1`
2) append the line done-after-sleep to marker.txt via shell
3) write patch-note.txt with content patched
4) read marker.txt
Reply in one short sentence when finished.
EOF
echo 'initial' >"${WORKDIR}/marker.txt"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "Killing existing tmux session ${SESSION}"
  tmux kill-session -t "${SESSION}"
fi

tmux new-session -d -s "${SESSION}" -n smoke "bash -lc '
  set -euo pipefail
  cd \"${WORKDIR}\"
  echo \"[agent] workspace: ${WORKDIR}\"
  echo \"[agent] model: ${MODEL}\"
  cursor-agent --print --output-format stream-json --stream-partial-output \\
    --force --trust --workspace \"${WORKDIR}\" --model \"${MODEL}\" \\
    \"\$(cat \"${PROMPT_FILE}\")\" 2>stderr.log | tee \"${STREAM_FILE}\"
  echo \"[agent] done. lines: \$(wc -l <\"${STREAM_FILE}\")\"
  echo \"[agent] marker.txt:\"
  cat marker.txt || true
  read -r -p \"Press enter to close agent pane...\"
'"

tmux split-window -h -t "${SESSION}:smoke" "bash -lc '
  set -euo pipefail
  echo \"[viz] waiting for ${STREAM_FILE}...\"
  until [[ -s \"${STREAM_FILE}\" ]]; do sleep 0.5; done
  echo \"[viz] replaying with codex-shim cursor_stream_visualizer\"
  cd \"${ROOT}\"
  python3 -m codex_shim.cursor_stream_visualizer \"${STREAM_FILE}\" \\
    --workdir \"${WORKDIR}\" --delay 0.05
  echo
  echo \"[viz] running pytest fixture replay...\"
  uv run pytest tests/test_cursor_passthrough_smoke.py -q
  read -r -p \"Press enter to close viz pane...\"
'"

tmux select-layout -t "${SESSION}:smoke" even-horizontal
tmux attach -t "${SESSION}"

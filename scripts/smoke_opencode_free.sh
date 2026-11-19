#!/usr/bin/env bash
# Smoke-test OpenCode Free (zen_public / oc-free-*) through the shim.
# Never SMOKE_RESTART=1 on port 8765 (production Desktop shim).
# SMOKE_MODEL pins a slug manually; otherwise a free model is picked live
# from models.dev (provider `opencode`, cost.input == 0, not deprecated),
# preferring fast/flash slugs. No hardcoded default: free slugs rotate fast.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export SMOKE_PROMPT="${SMOKE_PROMPT:-Reply with the single word pong. Do not use tools.}"
export SMOKE_REASONING_EFFORT="${SMOKE_REASONING_EFFORT:-low}"

if [[ -z "${SMOKE_MODEL:-}" ]]; then
  AUTO="$(python3 - <<'PY'
import json
import re
import urllib.request

request = urllib.request.Request(
    "https://models.dev/api.json",
    headers={"User-Agent": "codex-shim-smoke", "Accept": "application/json"},
)
payload = json.load(urllib.request.urlopen(request, timeout=20))
models = (payload.get("opencode") or {}).get("models") or {}
free = [
    key
    for key, row in models.items()
    if isinstance(row, dict) and row.get("status") != "deprecated"
    and (row.get("cost") or {}).get("input") == 0
]
def rank(slug):
    lowered = slug.lower()
    return (0 if "flash" in lowered else 1 if "mini" in lowered else 2, lowered)
free.sort(key=rank)
if free:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", free[0]).strip("-").lower()
    print(f"oc-free-{slug}")
PY
  )" || AUTO=""
  if [[ -z "${AUTO}" ]]; then
    echo "Could not pick a free OpenCode model from models.dev."
    echo "Pass one manually, e.g. SMOKE_MODEL=oc-free-big-pickle $0"
    exit 1
  fi
  export SMOKE_MODEL="${AUTO}"
  echo "Auto-picked SMOKE_MODEL from models.dev: ${SMOKE_MODEL}"
fi

exec bash "${ROOT}/scripts/smoke_chatgpt_passthrough.sh"

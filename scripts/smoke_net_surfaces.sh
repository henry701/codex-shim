#!/usr/bin/env bash
# Smoke Codex (ChatGPT Luna) and OpenCode Free after networking/keepalive changes.
# Never SMOKE_RESTART=1 on port 8765 (production Desktop shim).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${SMOKE_PORT:-8766}"
if [[ "${SMOKE_RESTART:-0}" == "1" && "${PORT}" == "8765" ]]; then
  echo "Refusing SMOKE_RESTART=1 on port 8765 (production Desktop shim)."
  exit 1
fi

SURFACES="${SMOKE_SURFACES:-chatgpt,opencode}"
FAILED=0
IFS=',' read -r -a ITEMS <<<"${SURFACES}"
for surface in "${ITEMS[@]}"; do
  surface="${surface// /}"
  case "${surface}" in
    chatgpt|codex|luna)
      echo "=== surface: Codex ChatGPT (${SMOKE_CHATGPT_MODEL:-codex-gpt-5-6-luna}) ==="
      SMOKE_MODEL="${SMOKE_CHATGPT_MODEL:-codex-gpt-5-6-luna}" \
        bash "${ROOT}/scripts/smoke_chatgpt_passthrough.sh" || FAILED=1
      ;;
    opencode|oc-free|zen_public)
      echo "=== surface: OpenCode Free (${SMOKE_OPENCODE_MODEL:-auto from models.dev}) ==="
      SMOKE_MODEL="${SMOKE_OPENCODE_MODEL:-}" \
        bash "${ROOT}/scripts/smoke_opencode_free.sh" || FAILED=1
      ;;
    nvidia|nemotron)
      echo "=== surface: NVIDIA Nemotron (${SMOKE_NVIDIA_MODEL:-nvidia-nvidia-nemotron-3-5-lightning-30b-a3b}) ==="
      SMOKE_MODEL="${SMOKE_NVIDIA_MODEL:-nvidia-nvidia-nemotron-3-5-lightning-30b-a3b}" \
        bash "${ROOT}/scripts/smoke_chatgpt_passthrough.sh" || FAILED=1
      ;;
    openrouter)
      echo "=== surface: OpenRouter free (${SMOKE_OPENROUTER_MODEL:-or-nvidia-nemotron-3-5-lightning-free}) ==="
      SMOKE_MODEL="${SMOKE_OPENROUTER_MODEL:-or-nvidia-nemotron-3-5-lightning-free}" \
        bash "${ROOT}/scripts/smoke_chatgpt_passthrough.sh" || FAILED=1
      ;;
    muse)
      # Muse Spark left the models.dev catalog; the surface now runs whatever
      # free OpenCode model models.dev serves (or SMOKE_MUSE_MODEL when pinned).
      echo "=== surface: OpenCode Free via Muse slot (${SMOKE_MUSE_MODEL:-auto from models.dev}) ==="
      SMOKE_MODEL="${SMOKE_MUSE_MODEL:-}" \
        bash "${ROOT}/scripts/smoke_opencode_free.sh" || FAILED=1
      ;;
    zai|z.ai)
      echo "=== surface: z.ai on OpenCode ==="
      if [[ -z "${OPENCODE_API_KEY:-}" ]]; then
        echo "z.ai GLM on OpenCode Zen is paid (glm-5.3-flash); OPENCODE_API_KEY is unset and oc-free has no glm."
        FAILED=1
      else
        SMOKE_MODEL="${SMOKE_ZAI_MODEL:-zen-glm-5-3-flash}" \
          bash "${ROOT}/scripts/smoke_chatgpt_passthrough.sh" || FAILED=1
      fi
      ;;
    "")
      ;;
    *)
      echo "Unknown smoke surface: ${surface} (chatgpt|opencode|nvidia|openrouter|muse|zai)"
      exit 1
      ;;
  esac
done

exit "${FAILED}"

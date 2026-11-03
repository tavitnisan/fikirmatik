#!/usr/bin/env bash
# ComfyUI on the box's 127.0.0.1:8188; reach it from your Mac through `./gpu.sh tunnel`.
set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh
cd "$WORK/ComfyUI"
exec "$(local_venv comfy)/bin/python" main.py \
  --listen 127.0.0.1 --port "${COMFY_PORT:-8188}" \
  --output-directory "$WORK/images" \
  --disable-auto-launch \
  ${COMFY_ARGS:-}

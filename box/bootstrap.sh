#!/usr/bin/env bash
# Runs ON the GPU box. Safe to re-run: finished steps are skipped or are no-ops.
#   lab venv  : torch + transformers + tuning tools (lab.py, tune.sh)
#   vllm venv : vLLM (serving), kept separate because it pins its own torch
#   model     : downloaded into $WORK/hf so it survives pod restarts
set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh
mkdir -p "$WORK"/{venvs,runs,models,hf}

echo "== GPU";  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "== disk"; df -h "$WORK" | tail -1
echo "== model: $MODEL  (work dir: $WORK)"

if [ "$(id -u)" = 0 ] && command -v apt-get >/dev/null && ! command -v tmux >/dev/null; then
  apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq tmux >/dev/null
fi

ensure_uv

echo "== lab venv (torch, transformers, tuning tools)"
[ -x "$WORK/venvs/lab/bin/python" ] || uv venv -q --python 3.12 "$WORK/venvs/lab"
# torchvision + pillow: tune.sh loads the image processor of multimodal models (Qwen3.5+).
uv pip install -q --python "$WORK/venvs/lab/bin/python" --torch-backend=auto torch torchvision
# flash-linear-attention: fast kernels for the linear-attention layers in hybrid
# models (Qwen3.5/3.6/3.8, Qwen3-Next); without it transformers uses a slow fallback.
# The tuning tool isn't installed here: tune.sh adds it the first time it runs.
uv pip install -q --python "$WORK/venvs/lab/bin/python" --torch-backend=auto \
  transformers accelerate datasets "huggingface_hub>=1.7" "flash-linear-attention[cuda]" pillow

if [ "${SKIP_VLLM:-0}" != 1 ]; then
  echo "== vllm venv"
  # vLLM's PyPI wheels are compiled against CUDA 13, so torch must be the cu130 build
  # (driver with CUDA >= 13.0). `auto` can pick an older build that loads but can't start.
  if ! "$WORK/venvs/vllm/bin/python" -c "import vllm.platforms.cuda" 2>/dev/null; then
    rm -rf "$WORK/venvs/vllm"
    uv venv -q --python 3.12 "$WORK/venvs/vllm"
    uv pip install -q --python "$WORK/venvs/vllm/bin/python" --torch-backend="${VLLM_TORCH_BACKEND:-cu130}" vllm
    rm -rf "$WORK/venvs/vllm.pack"
  fi
  [ -f "$WORK/venvs/vllm.pack/done" ] || pack_venv vllm  # serve.sh starts from this local copy
fi

echo "== downloading $MODEL"
"$WORK/venvs/lab/bin/hf" download "$MODEL" --exclude "original/*" --exclude "*.gguf" >/dev/null

echo "== checks"
"$WORK/venvs/lab/bin/python" -c "import torch, transformers; print('lab : torch', torch.__version__, '| transformers', transformers.__version__, '| cuda', torch.cuda.is_available(), torch.cuda.device_count(), 'GPU(s)')"
[ "${SKIP_VLLM:-0}" = 1 ] || "$WORK/venvs/vllm/bin/python" -c "import vllm, torch, vllm.platforms.cuda; print('vllm:', vllm.__version__, '| torch', torch.__version__, '| kernels load OK')"
du -sh "$HF_HOME" | sed 's/^/hf cache: /'
echo "== bootstrap done"

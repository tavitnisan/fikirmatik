#!/usr/bin/env bash
# Runs ON the GPU box. Installs ai-toolkit (LoRA training) in its own venv; it
# pins torch and transformers versions that differ from the other venvs.
# Run image_setup.sh first: training reuses the MOSS-Image files it downloaded.
set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh
AITK_REF="${AITK_REF:-14f8f9509068}"   # ai-toolkit main on 2026-10-22 (MOSS-Image support landed 2026-10-16)
AITK="$WORK/ai-toolkit"

ensure_uv

echo "== ai-toolkit $AITK_REF"
if [ ! -d "$AITK/.git" ]; then
  git clone -q https://github.com/ostris/ai-toolkit.git "$AITK"
fi
git -C "$AITK" fetch -q origin
git -C "$AITK" checkout -q "$AITK_REF"
git -C "$AITK" submodule update -q --init --recursive

[ -x "$WORK/venvs/aitk/bin/python" ] || uv venv -q --python 3.12 "$WORK/venvs/aitk"
# Versions from ai-toolkit's README; the CUDA build is picked to match the driver.
uv pip install -q --python "$WORK/venvs/aitk/bin/python" --torch-backend=auto \
  torch==2.13.0 torchvision==0.28.0 torchaudio==2.11.0
uv pip install -q --python "$WORK/venvs/aitk/bin/python" --torch-backend=auto -r "$AITK/requirements.txt"

mkdir -p "$WORK/datasets" "$WORK/loras"
"$WORK/venvs/aitk/bin/python" -c "import torch, diffusers, transformers; print('aitk: torch', torch.__version__, '| diffusers', diffusers.__version__, '| transformers', transformers.__version__, '| cuda', torch.cuda.is_available())"
echo "== training setup done"

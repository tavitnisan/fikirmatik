#!/usr/bin/env bash
# Runs ON the GPU box. Self-hosted image generation: ComfyUI + MOSS-Image
# (Apache-2.0). Safe to re-run.
#   comfy venv : ComfyUI (pinned release with native MOSS-Image support)
#   models     : downloaded into the shared HF cache, symlinked into ComfyUI
set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh
COMFY_REF="${COMFY_REF:-v0.42.4}"
COMFY="$WORK/ComfyUI"
REPO=Comfy-Org/MOSS-Image
FILES=(
  diffusion_models/moss_image_sft_bf16.safetensors   # tuned for prompt following and layout
  diffusion_models/moss_image_base_bf16.safetensors  # rawer, and the one to train LoRAs on
  text_encoders/qwen3.5_9b_bf16.safetensors
  vae/flux2-vae.safetensors
)

ensure_uv

echo "== ComfyUI $COMFY_REF"
if [ ! -d "$COMFY/.git" ]; then
  git clone -q https://github.com/comfyanonymous/ComfyUI.git "$COMFY"
fi
git -C "$COMFY" fetch -q --tags origin
git -C "$COMFY" checkout -q "$COMFY_REF"
[ -x "$WORK/venvs/comfy/bin/python" ] || uv venv -q --python 3.12 "$WORK/venvs/comfy"
before=$(uv pip freeze --python "$WORK/venvs/comfy/bin/python" 2>/dev/null | md5sum)
uv pip install -q --python "$WORK/venvs/comfy/bin/python" --torch-backend=auto torch torchvision torchaudio
uv pip install -q --python "$WORK/venvs/comfy/bin/python" --torch-backend=auto -r "$COMFY/requirements.txt" "huggingface_hub>=1.7"
[ "$(uv pip freeze --python "$WORK/venvs/comfy/bin/python" 2>/dev/null | md5sum)" = "$before" ] || rm -rf "$WORK/venvs/comfy.pack"
[ -f "$WORK/venvs/comfy.pack/done" ] || pack_venv comfy  # comfy.sh starts from this local copy

echo "== downloading $REPO (~40 GB)"
# Ask the library for the snapshot folder: the CLI's printed format differs between TTYs and pipes.
snap=$("$WORK/venvs/comfy/bin/python" -c 'import sys; from huggingface_hub import snapshot_download; print(snapshot_download(sys.argv[1], allow_patterns=sys.argv[2:]))' "$REPO" "${FILES[@]}" | tail -1)
[ -d "$snap" ] || { echo "download failed: $snap"; exit 1; }
for f in "${FILES[@]}"; do
  mkdir -p "$COMFY/models/$(dirname "$f")"
  ln -sf "$snap/$f" "$COMFY/models/$f"
done

mkdir -p "$WORK/loras" "$WORK/images"
# Trained LoRAs and generated images live outside the ComfyUI checkout.
[ -L "$COMFY/models/loras/trained" ] || ln -s "$WORK/loras" "$COMFY/models/loras/trained"

echo "== checks"
"$WORK/venvs/comfy/bin/python" -c "import torch; print('comfy: torch', torch.__version__, '| cuda', torch.cuda.is_available())"
ls -lL "$COMFY"/models/{diffusion_models,text_encoders,vae}/*.safetensors | awk '{n = split($9, p, "/"); printf "%6.1f GB  %s\n", $5/1e9, p[n]}'
echo "== image setup done. Start it with ./gpu.sh comfy"

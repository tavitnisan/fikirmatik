#!/usr/bin/env bash
# train.sh NAME [STEPS] [TRIGGER]
# Trains a MOSS-Image LoRA on $WORK/datasets/NAME (images + same-named .txt captions).
# Checkpoints every 250 steps go to $WORK/loras/NAME/ with sample images, so you can
# pick the best step instead of the last one. Re-running resumes from the last save.
set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh
name=$1
steps="${2:-2000}"
trigger="${3:-$name}"
dataset="$WORK/datasets/$name"
[[ "$name$trigger" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "name/trigger: letters, digits, _ and - only"; exit 1; }
[ -d "$dataset" ] || { echo "no dataset at $dataset; upload one with ./gpu.sh dataset <folder> $name"; exit 1; }
n_img=$(( $(find "$dataset" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.webp' \) | wc -l) ))
n_txt=$(( $(find "$dataset" -maxdepth 1 -type f -name '*.txt' | wc -l) ))
echo "dataset $dataset: $n_img images, $n_txt captions"
[ "$n_img" -gt 0 ] || { echo "no images; push some with ./gpu.sh dataset <folder> $name"; exit 1; }
[ "$n_txt" -ge "$n_img" ] || echo "warning: some images have no caption (./gpu.sh caption $name)"

cfg="$WORK/loras/$name.yaml"
mkdir -p "$WORK/loras"
sed -e "s|__NAME__|$name|" -e "s|__OUT__|$WORK/loras|" -e "s|__TRIGGER__|$trigger|" \
    -e "s|__DATASET__|$dataset|" -e "s|__STEPS__|$steps|" lora_moss.yaml > "$cfg"
cd "$WORK/ai-toolkit"
"$WORK/venvs/aitk/bin/python" run.py "$cfg"

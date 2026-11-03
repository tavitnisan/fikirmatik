#!/usr/bin/env bash
# serve.sh [MODEL_OR_FOLDER] [NAME]
# OpenAI-compatible API on 127.0.0.1:8000 of the box. It is deliberately not
# exposed publicly; reach it from your Mac through `./gpu.sh tunnel`.
# 128K context by default: Qwen3.8 only keeps KV cache for 1 layer in 4, so a 27B in BF16 on
# one 80 GB GPU still has room for ~230K cached tokens, and agent work fills context fast.
# MAX_SEQS caps concurrent requests. vLLM's default on Blackwell (1024) needs one linear-attention
# cache block per request and doesn't fit next to 128K of context; a personal server needs a handful.
set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh
target="${1:-$MODEL}"
name="${2:-$(basename "$target")}"
VENV=$(local_venv vllm)
export PATH="$VENV/bin:$PATH"
# vLLM sizes its own memory pool; the expandable-segments setting from env.sh fights it.
unset PYTORCH_CUDA_ALLOC_CONF
# FlashInfer's sampler is compiled on first use and needs a CUDA 13 nvcc the image lacks;
# vLLM's PyTorch sampler needs nothing extra.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
# Tool calling + separated reasoning, needed by agent harnesses. These parsers are
# for the Qwen3.x family; set TOOL_ARGS="" (or other parsers) for other models.
TOOL_ARGS="${TOOL_ARGS---reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder}"
exec "$VENV/bin/vllm" serve "$target" \
  --served-model-name "$name" \
  --host 127.0.0.1 --port "${PORT:-8000}" \
  --tensor-parallel-size "$(nvidia-smi -L | wc -l)" \
  --max-model-len "${MAX_LEN:-131072}" \
  --max-num-seqs "${MAX_SEQS:-64}" \
  --gpu-memory-utilization "${GPU_UTIL:-0.92}" \
  $TOOL_ARGS ${VLLM_ARGS:-}

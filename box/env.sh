# Sourced by every script on the GPU box. /workspace is the persistent volume on
# RunPod and Vast; anything outside it disappears when the instance is recreated.
WORK="${WORK:-/workspace}"
{ [ -d "$WORK" ] && [ -w "$WORK" ]; } || WORK="$HOME/work"
export WORK
export HF_HOME="$WORK/hf"
# The package cache is disposable; keep it on the container disk, not the billed volume.
export UV_CACHE_DIR="${UV_CACHE_DIR:-$HOME/.cache/uv}"
export UV_LINK_MODE=copy
export MODEL="${MODEL:-Qwen/Qwen3.8-27B}"
export PATH="$WORK/venvs/lab/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Network volumes are slow per file, and starting vLLM or ComfyUI opens tens of thousands of them.
# pack_venv NAME stores $WORK/venvs/NAME as a few archives in $WORK/venvs/NAME.pack; local_venv NAME
# unpacks that onto the container disk once per boot (or after a re-pack) and prints the directory
# to run from. With no pack it prints the venv on the volume, which works as before, just slower.
LOCAL_VENVS=/root/venvs
pack_venv() {
  local v=$1 src="$WORK/venvs/$1" pack="$WORK/venvs/$1.pack"
  [ -x "$src/bin/python" ] || return 0
  echo "== packing the $v venv for fast starts"
  rm -rf "$pack.new" && mkdir -p "$pack.new"
  (cd "$src" && find . -mindepth 1 | LC_ALL=C sort > "$pack.new/list")
  split -n l/16 -d "$pack.new/list" "$pack.new/list."
  local l
  for l in "$pack.new"/list.??; do  # parallel readers: the volume's latency, not its bandwidth, is the limit
    (cd "$src" && tar -c --no-recursion -T "$l" | zstd -q -3 -T1 > "$pack.new/part${l##*.}.tar.zst") &
  done
  wait
  rm -f "$pack.new"/list* && touch "$pack.new/done" && rm -rf "$pack" && mv "$pack.new" "$pack"
}
local_venv() {
  local v=$1 pack="$WORK/venvs/$1.pack" dst="$LOCAL_VENVS/$1"
  [ -f "$pack/done" ] || { echo "$WORK/venvs/$v"; return; }
  if [ ! -f "$dst/.unpacked" ] || [ "$pack/done" -nt "$dst/.unpacked" ]; then
    echo "unpacking the $v venv to local disk" >&2
    rm -rf "$dst" && mkdir -p "$dst"
    ls "$pack"/part*.tar.zst | xargs -P 8 -I{} sh -c "zstd -dcq '{}' | tar -x -C '$dst'" >&2
    # Entry-point scripts and activate name the volume path; point them at the local copy.
    { grep -rlIZ "$WORK/venvs/$v" "$dst/bin" || true; } | xargs -0 -r sed -i "s|$WORK/venvs/$v|$dst|g"
    touch "$dst/.unpacked"
  fi
  echo "$dst"
}

# Base images often ship an old uv that predates the CUDA 13 (cu130) torch builds.
ensure_uv() {
  if ! uv --version 2>/dev/null | awk '{split($2, v, "."); exit !(v[1] > 0 || v[2] >= 10)}'; then
    # The system interpreter: the venvs on PATH have no pip.
    local py
    py=$(PATH=/usr/local/bin:/usr/bin:/bin command -v python3)
    "$py" -m pip install -q -U "uv>=0.10" 2>/dev/null || "$py" -m pip install -q -U --break-system-packages "uv>=0.10"
    PATH="$(dirname "$("$py" -c 'import uv; print(uv.find_uv_bin())')"):$PATH"
    hash -r
  fi
  uv --version
}

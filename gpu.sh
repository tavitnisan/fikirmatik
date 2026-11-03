#!/usr/bin/env bash
# Drive a rented GPU box from your Mac. Works with any provider that hands you
# `ssh -p PORT root@HOST` (RunPod, Vast.ai, Lambda, ...). Settings live in .env.
set -euo pipefail
KIT="$(cd "$(dirname "$0")" && pwd)"
[ -f "$KIT/.env" ] && { set -a; . "$KIT/.env"; set +a; }
SSH_KEY="${SSH_KEY:-$HOME/.ssh/gpu_rental_ed25519}"
POD_USER="${POD_USER:-root}"
POD_PORT="${POD_PORT:-22}"
MODEL="${MODEL:-tavitnisanyan/fikirmatik}"
KNOWN_HOSTS="$HOME/.ssh/known_hosts_gpu"

usage() {
  cat <<EOF
usage: ./gpu.sh <command>

  keygen              create the SSH key to paste into your provider (once)
  setup               push the kit, install everything, download \$MODEL  (~15-30 min)
  tune [model]        automatic tuning run (attach to pick and save a result)
  lab <args>          run box/lab.py on the GPU, e.g. ./gpu.sh lab collect
  serve [path] [name] start vLLM for a model id or a saved folder
  tunnel              forward vLLM (:8000) and ComfyUI (:8188) to this Mac
  chat                chat with whatever is being served (needs the tunnel)
  ui                  fikirmatik.ai in your browser: tunnel, serve and chat without a terminal

  image-setup         install ComfyUI + MOSS-Image (~40 GB)    (~15 min)
  comfy               start ComfyUI; open http://127.0.0.1:8188 with the tunnel up
  imagine "prompt"    generate from the terminal; images land in ./images
  train-setup         install ai-toolkit for LoRA training
  dataset <dir> <name> upload a folder of images (+ .txt captions) for training
  caption <name>      caption a dataset with the served vision model (needs serve)
  train <name> [steps] [trigger]   train a LoRA; it shows up in ComfyUI/imagine

  attach [session]    attach to a tmux session; Ctrl-b d detaches
  logs [session]      follow a session's log
  stop <session>      stop a session (e.g. stop serve to free the GPU for images)
  status              GPU, sessions, disk
  stop-pod            stop the RunPod pod itself: GPU billing ends, /workspace is kept
  pull <path> [dest]  copy a file/folder from the box's work dir to ./pulled
  ssh                 plain shell on the box
  forget              forget the box's host key (after re-renting the same IP:port)
EOF
}

need_host() {
  [ -n "${POD_HOST:-}" ] || { echo "Set POD_HOST and POD_PORT in $KIT/.env (copy .env.example)."; exit 1; }
}

SSH_OPTS=(-i "$SSH_KEY" -p "$POD_PORT" -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KNOWN_HOSTS" -o ServerAliveInterval=30 -o ConnectTimeout=15)
# Remote prefix: resolve the work dir the same way box/env.sh does.
PRE='W=/workspace; { [ -d $W ] && [ -w $W ]; } || W=$HOME/work; mkdir -p $W/lab && cd $W/lab;'

rsh()  { need_host; ssh "${SSH_OPTS[@]}" "$POD_USER@$POD_HOST" "$PRE $*"; }
rtty() { need_host; ssh -t "${SSH_OPTS[@]}" "$POD_USER@$POD_HOST" "$PRE $*"; }
q()    { printf '%q ' "$@"; }

push() {
  # --no-same-owner: network volumes refuse to chown to the Mac's uid, and root owns everything there anyway.
  COPYFILE_DISABLE=1 tar -C "$KIT/box" --no-xattrs -czf - . | rsh 'tar xzf - --no-same-owner && chmod +x *.sh'
}

session() {  # session NAME CMD...
  local name=$1 v fwd; shift
  # Settings the box scripts read; session.sh writes them into the tmux job, since tmux doesn't pass them on.
  fwd="MODEL=$(q "$MODEL")"
  for v in MAX_LEN MAX_SEQS VLLM_ARGS GPU_UTIL TOOL_ARGS; do [ -z "${!v+x}" ] || fwd+=" $v=$(q "${!v}")"; done
  rsh "command -v tmux >/dev/null || { apt-get update -qq && apt-get install -y -qq tmux >/dev/null; }; $fwd ./session.sh $(q "$name" "$@")"
}

# NO_ATTACH=1 starts jobs without opening tmux (for scripts and agents); follow them with `logs`.
attach() {
  if [ -n "${NO_ATTACH:-}" ]; then echo "running in tmux session '${1:-lab}'; ./gpu.sh logs ${1:-lab} to follow"; return; fi
  rtty "tmux attach -t $(q "${1:-lab}")"
}

cmd="${1:-help}"; shift || true
case "$cmd" in
  keygen)
    if [ ! -f "$SSH_KEY" ]; then
      mkdir -p "$(dirname "$SSH_KEY")" && chmod 700 "$(dirname "$SSH_KEY")"
      ssh-keygen -t ed25519 -N "" -C "gpu-rental" -f "$SSH_KEY" >/dev/null
      echo "created $SSH_KEY"
    fi
    pbcopy < "$SSH_KEY.pub" 2>/dev/null && echo "public key copied to clipboard:" || echo "public key:"
    cat "$SSH_KEY.pub"
    ;;
  setup)
    push
    session setup bash bootstrap.sh
    attach setup
    ;;
  tune)
    push
    session tune bash tune.sh "${1:-$MODEL}"
    attach tune
    ;;
  lab)
    push
    session lab python lab.py "$@"
    attach lab
    ;;
  serve)
    push
    session serve bash serve.sh "$@"
    echo "loading takes a few minutes; './gpu.sh logs serve' to watch, then './gpu.sh tunnel'"
    ;;
  tunnel)
    need_host
    echo "vLLM    http://127.0.0.1:8000  -> box:8000"
    echo "ComfyUI http://127.0.0.1:8188  -> box:8188   (Ctrl-C to stop)"
    ssh -N -L "8000:127.0.0.1:8000" -L "8188:127.0.0.1:8188" "${SSH_OPTS[@]}" "$POD_USER@$POD_HOST"
    ;;
  chat)     python3 "$KIT/chat_client.py" "$@" ;;
  ui)       python3 "$KIT/ui.py" "$@" ;;
  image-setup)
    push
    session image-setup bash image_setup.sh
    attach image-setup
    ;;
  comfy)
    push
    session comfy bash comfy.sh
    echo "starting; './gpu.sh logs comfy' to watch, then open http://127.0.0.1:8188 with './gpu.sh tunnel' running"
    ;;
  imagine)  python3 "$KIT/imagine.py" "$@" ;;
  train-setup)
    push
    session train-setup bash train_setup.sh
    attach train-setup
    ;;
  dataset)
    [ -d "${1:-}" ] && [ -n "${2:-}" ] || { echo "usage: ./gpu.sh dataset <local folder> <name>"; exit 1; }
    COPYFILE_DISABLE=1 tar -C "$1" --no-xattrs -czf - . | rsh "mkdir -p \$W/datasets/$(q "$2") && tar xzf - --no-same-owner -C \$W/datasets/$(q "$2") && ls \$W/datasets/$(q "$2") | wc -l | xargs echo files on box:"
    ;;
  caption)
    push
    session caption python caption.py "${1:?usage: ./gpu.sh caption <name> [--trigger WORD]}" "${@:2}"
    attach caption
    ;;
  train)
    push
    session train bash train.sh "$@"
    attach train
    ;;
  stop)     rsh "tmux kill-session -t $(q "${1:?usage: ./gpu.sh stop <session>}") && echo stopped $(q "$1")" ;;
  attach)   attach "${1:-lab}" ;;
  logs)     rsh "tail -n 80 $([ -n "${NO_FOLLOW:-}" ] || echo -f) \$W/runs/$(q "${1:-lab}.log")" ;;  # NO_FOLLOW=1 prints and returns
  stop-pod)  # RunPod only: the pod stops itself, using the pod-scoped key RunPod puts in /etc/rp_environment
    rsh '[ -f /etc/rp_environment ] && . /etc/rp_environment; { [ -n "${RUNPOD_POD_ID:-}" ] && command -v runpodctl >/dev/null; } || { echo "not a RunPod pod (no runpodctl or RUNPOD_POD_ID): stop it from the provider console"; exit 1; }; runpodctl stop pod "$RUNPOD_POD_ID" 2>&1 || runpodctl pod stop "$RUNPOD_POD_ID"'
    ;;
  status)   rsh 'nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader; echo; tmux ls 2>/dev/null || echo "no tmux sessions"; echo; df -h $W | tail -1; du -sh $W/hf $W/models $W/ComfyUI/models 2>/dev/null; true' ;;
  pull)
    mkdir -p "${2:-$KIT/pulled}"
    rsh "tar czf - -C \$W $(q "$1")" | tar xzf - -C "${2:-$KIT/pulled}"
    echo "copied to ${2:-$KIT/pulled}/$1"
    ;;
  ssh)      rtty 'exec bash -l' ;;
  forget)   need_host; ssh-keygen -R "[$POD_HOST]:$POD_PORT" -f "$KNOWN_HOSTS" ;;
  *)        usage ;;
esac

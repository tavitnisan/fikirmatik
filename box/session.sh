#!/usr/bin/env bash
# session.sh NAME CMD [ARGS...]
# Runs CMD in a detached tmux session so it survives SSH disconnects.
# Output is also appended to $WORK/runs/NAME.log. Re-using a NAME replaces that session.
set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh
name=$1; shift
mkdir -p "$WORK/runs"
runner="$WORK/runs/.$name.sh"
{
  # A tmux server that is already running gives new sessions its own environment, not ours.
  for v in MODEL MAX_LEN MAX_SEQS VLLM_ARGS GPU_UTIL TOOL_ARGS; do [ -z "${!v+x}" ] || printf 'export %s=%q\n' "$v" "${!v}"; done
  echo "cd $(printf %q "$PWD") && source ./env.sh"
  printf '%q ' "$@"; echo
  echo 'rc=$?; echo; echo "[exited with $rc] Ctrl-b then d to detach"; exec bash'
} > "$runner"
tmux kill-session -t "$name" 2>/dev/null || true
tmux new-session -d -s "$name" -x 200 -y 50 "bash $runner"
tmux pipe-pane -t "$name" -o "cat >> $WORK/runs/$name.log"
echo "started tmux session '$name' (log: $WORK/runs/$name.log)"

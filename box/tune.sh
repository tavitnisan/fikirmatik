#!/usr/bin/env bash
# tune.sh [MODEL]
# Each model gets its own run folder, so an interrupted run resumes from its
# Optuna checkpoint when started again.
set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh
# Installed on first use rather than by setup, so it isn't on the box unless you tune.
command -v heretic >/dev/null || { ensure_uv >/dev/null; uv pip install -q --python "$WORK/venvs/lab/bin/python" "heretic-llm==1.4.0"; }
model="${1:-$MODEL}"
run="$WORK/runs/tune-${model//\//__}"
mkdir -p "$run"
cp tune.toml "$run/config.toml"
cd "$run"
echo "run folder: $run"
echo "when it finishes, save the model to $WORK/models/<name> (merge strategy is preset)"
heretic --model "$model"

#!/usr/bin/env bash
# Run a single stage, optionally for one model only.
#   bash run/run_one_stage.sh generate configs/full.yaml qwen7b
# Stages in order: prepare generate judge sample_scores split probe metaqa probe_scores evaluate
set -euo pipefail
cd "$(dirname "$0")/.."
STAGE="$1"; CONFIG="${2:-configs/full.yaml}"; MODEL="${3:-}"
if [ -n "$MODEL" ]; then
  python -m pressure_probe.cli stage "$STAGE" --config "$CONFIG" --model "$MODEL"
else
  python -m pressure_probe.cli stage "$STAGE" --config "$CONFIG"
fi

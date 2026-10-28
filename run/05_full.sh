#!/usr/bin/env bash
# Step 5. Full experiment (about 4 to 10 GPU hours). Safe to rerun after a disconnect:
# finished work is skipped. To run one stage at a time instead, use run_one_stage.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m pressure_probe.cli run --config configs/full.yaml

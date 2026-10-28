#!/usr/bin/env bash
# Step 3. Smoke test: every stage on 24 questions with two tiny models (about 15 minutes on a T4).
# Needs a GPU and HF_TOKEN. Output: results/smoke/ and a throwaway paper build.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m pressure_probe.cli run --config configs/smoke.yaml
python -m pressure_probe.cli report --config configs/smoke.yaml
echo "Smoke test finished. Check results/smoke/summary.json and paper/validation_report.json."

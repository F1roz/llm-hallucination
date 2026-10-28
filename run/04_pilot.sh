#!/usr/bin/env bash
# Step 4. Pilot: 200 questions on the real 7B/8B models (about 1 to 2 hours on a T4).
# Go on to the full run only if each model has at least 15 CW answers and the CPS AUROC
# is above 0.6 for at least one model (both printed at the end of this script).
set -euo pipefail
cd "$(dirname "$0")/.."
python -m pressure_probe.cli run --config configs/pilot.yaml
python - <<'PY'
import json
summary = json.load(open("results/pilot/summary.json"))
for key, model in summary["models"].items():
    cw = model["counts"]["CW"]
    cps = model["methods"]["cps"]["auroc_matched"]["value"]
    ready = cw >= 15 and cps is not None and cps > 0.6
    print(f"{model['display_name']}: CW = {cw}, CPS AUROC = {cps}, meets go rule: {ready}")
print("Go to the full run if at least one model meets the go rule and both have CW >= 15.")
PY

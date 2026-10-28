#!/usr/bin/env bash
# Step 6. Make the human labelling sheet (50 CW + 50 CR answers) at results/full/human_labels.csv.
# Two authors fill rater1 and rater2 with 1 (correct) or 0 (wrong), independently.
set -euo pipefail
cd "$(dirname "$0")/.."
python scripts/make_label_sheet.py --config configs/full.yaml

#!/usr/bin/env bash
# Step 2. Run the offline tests on a simulated model. No GPU, no downloads, about one minute.
# All tests must pass before you spend GPU time.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m pytest -q

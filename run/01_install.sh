#!/usr/bin/env bash
# Step 1. Install the project and its dependencies.
# GPU machine (Colab, Kaggle): installs torch, transformers, the OpenAI client and test tools.
# vLLM is optional and makes generation faster; the code falls back to transformers without it.
set -euo pipefail
cd "$(dirname "$0")/.."
pip install -q -e ".[gpu,judge,dev]"
pip install -q vllm || echo "vLLM not installed; the slower transformers backend will be used."
python -c "import pressure_probe; print('pressure_probe installed')"

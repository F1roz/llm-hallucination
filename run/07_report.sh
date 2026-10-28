#!/usr/bin/env bash
# Step 7. Put the results into the paper: figures, tables, text, PDF, Word file, ZIP.
# Run again whenever you change paper/main.template.tex, authors.yaml or the human labels.
# Needs LaTeX (latexmk) and pandoc. On Colab: apt-get install texlive-latex-extra texlive-science latexmk pandoc
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIG="${1:-configs/full.yaml}"
python -m pressure_probe.cli report --config "$CONFIG"
cat paper/validation_report.json

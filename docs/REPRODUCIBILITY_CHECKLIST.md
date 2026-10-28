# Reproducibility Checklist

| Item | Where it is recorded | Status |
| --- | --- | --- |
| Code version | git commit of the repository | set when you push |
| Configuration | `configs/<mode>.yaml`, copied to `results/<run>/config_used.json` with a fingerprint | automatic |
| Random seeds | `seed` in config; per dataset sample seed; per request generation seeds; bootstrap seed | automatic |
| Dataset versions | pinned commit hashes in `src/pressure_probe/data.py` | done |
| Sampled items | `results/<run>/questions.jsonl` | automatic |
| Model versions | pinned revisions in the config | done |
| Precision and backend | `results/<run>/<model>/backend.json`, printed in the paper | automatic |
| Hardware | `results/<run>/environment.json` (GPU name, memory, CUDA) | automatic |
| Package versions | `results/<run>/environment.txt` (pip freeze at run time) | automatic |
| Prompts | `src/pressure_probe/prompts.py`; challenge prompts printed in the paper's prompt table from the same code | done |
| Raw outputs | `generations.jsonl`, `probes.jsonl`, `metaqa.jsonl` per model | automatic |
| Intermediate scores | `judgments.jsonl`, `sample_scores.jsonl`, `split.jsonl`, `probe_scores.jsonl`, `scores.csv` | automatic |
| All reported numbers | `results/<run>/summary.json`, the only input to the paper | automatic |
| Human labels | `results/<run>/human_labels.csv` | team fills |
| Wall clock and generations | `results/<run>/<model>/timings.json` | automatic |
| Logs | `results/<run>/logs/pipeline.log` | automatic |
| Paper build | `paper/validation_report.json`, `paper/generated/values.json` | automatic |
| Tests | `python -m pytest -q`; results in `docs/TEST_REPORT.md` | done |

## Known sources of variation

- **GPU kernels.** Greedy decoding is deterministic up to floating point differences between GPU types and between vLLM and transformers. Rerun on the same hardware and backend for exact matches.
- **Quantization.** 4-bit weights on GPUs under 20 GB can change answers. The paper states the precision used.
- **API judge.** GPT-4o-mini is called with temperature 0 and a seed, but OpenAI does not guarantee identical outputs over time. Judgments are stored, so analyses rerun from stored labels are exact.
- **Bootstrap.** Seeded; identical across reruns of `evaluate`.

## Rerun from stored outputs

`python -m pressure_probe.cli stage evaluate --config configs/full.yaml` followed by
`python -m pressure_probe.cli report --config configs/full.yaml` recomputes every
statistic, figure and table from the stored files without a GPU.

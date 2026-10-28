# Agreement Is Not Truth

Code, protocol and paper for a study of **consistent hallucinations**: answers a large
language model gets wrong the same way every time. Detectors that sample a model and
look for disagreement cannot see them. This project tests whether pushing back on an
answer can.

The main score is the **doubt flip rate (DFR)**: after answering, the model is asked
"Are you sure?" in three phrasings, each in a separate conversation, and DFR is the
share of replies that abandon the answer. It needs only generated text, no training
and 4 generations per question. The contrastive variant **CPS** (counterclaim flip rate
minus doubt flip rate) was the original main score; the pilot showed that its
subtraction cancels the signal, so it is reported as a tested variant
(`docs/EXPERIMENTAL_PROTOCOL.md`, change log).

## What is in the repository

| Path | Contents |
| --- | --- |
| `src/pressure_probe/` | the pipeline: data, generation backends, NLI, judging, split, baselines, MetaQA, DFR and CPS, statistics, reporting |
| `configs/` | `smoke`, `pilot`, `full` (Colab) and `test` (offline, simulated model) |
| `notebooks/` | three Colab notebooks: smoke test, pilot, full experiment |
| `paper/` | IEEE template with yellow `[todo]` placeholders, bibliography, generated tables and figures |
| `docs/` | protocol, dataset notes, Colab guide, checklists, literature review, methodology guide, SOP |
| `tests/` | unit and integration tests on a simulated model |
| `tools/` | writing style checker |
| `run/` | numbered scripts for each step, from install to report |
| `scripts/` | label sheet and notebook builders |

## Quick start (offline, no GPU)

```bash
pip install -e .[dev]
python -m pytest -q                                   # all tests on the simulated model
python -m pressure_probe.cli report --config configs/full.yaml   # placeholder paper
```

## Running the experiment

Start with `CODE_GUIDE.md`: it lists the numbered run scripts in `run/` and explains
every file. The Colab walkthrough is `docs/COLAB_EXECUTION_GUIDE.md`. In short:

1. `notebooks/smoke_test.ipynb` checks every stage on tiny models.
2. `notebooks/pilot_experiment.ipynb` ran 200 questions (done on 27 October 2026; its questions are excluded from the full run).
3. `notebooks/full_experiment.ipynb` runs the full study and writes the results into the paper.

Command line equivalent:

```bash
python -m pressure_probe.cli run --config configs/full.yaml      # all stages, resumable
python -m pressure_probe.cli report --config configs/full.yaml   # figures, tables, PDF, Word, ZIP
```

## How results reach the paper

`results/<run>/summary.json` is the only source of numbers. The report step renders
`paper/main.tex` from `paper/main.template.tex`, replacing `\RES{...}`,
`\RESTXT{...}`, `\RESTABLE{...}` and `\RESFIG{...}` tokens. Anything without a
measured value stays as a yellow `[todo]`. Edit prose in the template, never numbers.

## Models and data

Qwen2.5-7B-Instruct and Llama-3.1-8B-Instruct (revisions pinned); TruthfulQA,
TriviaQA and SciQ (revisions pinned). See `docs/DATASET_DOCUMENTATION.md` for licenses.
SciQ is CC BY-NC 3.0, so released question text is for non commercial use.

## License

MIT for the code. See `LICENSE` for third party files in `paper/`.

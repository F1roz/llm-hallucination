# Code Guide

This guide explains how to run the code and what each file does, in the order the
data flows through it. Read sections 1 and 2 first. Section 3 is a reference for when
you want to understand or change something.

# 1. How to run it

Every step is a numbered script in `run/`. On Colab, prefix each with `!`, for example
`!bash run/03_smoke.sh`. The notebooks in `notebooks/` run the same commands cell by cell.

| Step | Command | Needs | Time (estimate) |
| --- | --- | --- | --- |
| 1 | `bash run/01_install.sh` | internet | 5 min |
| 2 | `bash run/02_test_offline.sh` | nothing | 1 min |
| 3 | `bash run/03_smoke.sh` | GPU, `HF_TOKEN` | 15 min |
| 4 | `bash run/04_pilot.sh` | GPU, `HF_TOKEN` | 1 to 2 h |
| 5 | `bash run/05_full.sh` | GPU, `HF_TOKEN` | 4 to 10 h |
| 6 | `bash run/06_label_sheet.sh`, then two authors fill the sheet | results of step 5 | 2 h reading |
| 7 | `bash run/07_report.sh` | LaTeX, pandoc | 2 min |

`OPENAI_API_KEY` is optional in steps 3 to 5. Without it, correctness is judged by
alias matching and a local NLI model.

Set the keys before running:

```bash
export HF_TOKEN=hf_...          # Colab: add as a secret instead
export OPENAI_API_KEY=sk-...    # optional
```

**If a run stops** (Colab disconnects, out of memory): run the same command again.
Each stage skips the questions it already finished.

**To run one stage only**, for one model or both:

```bash
bash run/run_one_stage.sh generate configs/full.yaml qwen7b
bash run/run_one_stage.sh evaluate configs/full.yaml
```

# 2. The big picture

```text
configs/full.yaml
      |
      v
prepare -> generate -> judge -> sample_scores -> split -> probe -> metaqa -> probe_scores -> evaluate -> report
 (data)   (answers)  (right/   (baselines       (CW/CR/  (pressure  (MetaQA    (which replies   (statistics   (paper)
                      wrong)    from samples)    INC)     questions) baseline)  flipped; DFR)   summary.json)
```

- Stages from `generate` to `probe_scores` run once per model and write to
  `results/<run>/<model>/`.
- `evaluate` reads both models and writes `results/<run>/summary.json`, the only source
  of numbers for the paper.
- `report` turns `summary.json` into tables, figures and text inside `paper/main.tex`,
  then builds the PDF and Word files.

# 3. Every file, in the order it is used

## Settings

| File | What it does |
| --- | --- |
| `configs/smoke.yaml` | Tiny models (Qwen2.5 0.5B and 1.5B), 8 questions per dataset. Checks the code runs. |
| `configs/pilot.yaml` | Real models, 100 TruthfulQA + 50 TriviaQA + 50 SciQ questions. Checks there is a signal. |
| `configs/full.yaml` | Real models, 717 TruthfulQA + 200 TriviaQA + 200 SciQ questions, pilot questions excluded, at most 120 CW per model in the matched set. The paper's run. |
| `configs/test.yaml` | Simulated model and synthetic questions for the offline tests. Never used for results. |
| `src/pressure_probe/config.py` | Loads a YAML file into a `Config` object and checks the values (for example, that the threshold is between 0 and 1). |

Settings you are most likely to change in a config:

| Setting | Meaning |
| --- | --- |
| `datasets` | how many questions per dataset (`null` means all) |
| `n_samples`, `sample_temperature` | sampled answers per question and their randomness (10, 0.5) |
| `consistency_threshold` | share of samples that must agree with the answer for it to count as consistent (0.9) |
| `n_pressure` | number of counterclaims and number of doubts per question (3) |
| `run_metaqa`, `run_random_alternative` | switch the MetaQA baseline and the random alternative ablation on or off |
| `batch_size` | lower it if the GPU runs out of memory |
| `judge.backend` | `auto` uses OpenAI if a key is set, otherwise alias + NLI |

## Shared building blocks

| File | What it does |
| --- | --- |
| `io_utils.py` | Reads and writes `.jsonl` files one row per question, so finished rows survive a crash. Sets up logging. |
| `text.py` | Cleans answers (`clean_answer`), normalises them for comparison, detects refusals, matches against reference aliases. |
| `data.py` | Downloads the three datasets at fixed versions, samples questions with a fixed seed, converts them to one `Question` format. |
| `prompts.py` | Every prompt in the study: the answer prompt, the 3 counterclaim and 3 doubt templates, P(True), verbal confidence, alternative answer, MetaQA, judge. Change wording here only. |
| `nli.py` | The DeBERTa NLI model. `EquivalenceJudge` decides whether two answers mean the same (entailment both ways, with the question in front). Results are cached. |
| `methods.py` | The list of methods in the tables: name, access type (text or logits), generations per question. |

## Model backends (`src/pressure_probe/backends/`)

All backends take the same request and return the same result, so the stages do not
care which one runs.

| File | What it does |
| --- | --- |
| `base.py` | `GenRequest` (messages, temperature, number of samples) and `GenResult` (texts, token log probabilities). |
| `__init__.py` | `load_backend` picks one: vLLM if installed and working, else transformers. Uses 4-bit models when GPU memory is under 20 GB. |
| `vllm_backend.py` | Fast batched generation with vLLM. |
| `hf_backend.py` | Generation with plain Hugging Face transformers. Slower, but works everywhere. |
| `openai_backend.py` | Calls GPT-4o-mini for the correctness judge. |
| `mock_backend.py` | A fake model for tests. Never used for results. |

## The stages (`src/pressure_probe/stages/`), in run order

| Order | File | Reads | Writes (in `results/<run>/<model>/`) | What it does |
| --- | --- | --- | --- | --- |
| 1 | `prepare.py` | datasets | `../questions.jsonl`, `../config_used.json`, `../environment.json` | Samples the questions and records the software environment. |
| 2 | `generate.py` | questions | `generations.jsonl` | Greedy answer, 10 samples, P(True) and verbal confidence for every question. |
| 3 | `judge.py` | generations | `judgments.jsonl`, `judge_methods.json` | Decides if each answer is correct: refusal check, then alias match, then OpenAI or NLI. |
| 4 | `sample_scores.py` | generations | `sample_scores.jsonl` | Compares samples with the answer: agreement rate, Semantic Entropy, SelfCheckGPT, cross family disagreement. |
| 5 | `split.py` | judgments, sample scores | `split.jsonl`, `matched_ids.json` | Labels each answer CW, CR, INC or EXCL and draws the matched set (all CW, equal CR per dataset). |
| 6 | `probe.py` | matched set | `probes.jsonl` | Asks for an alternative answer, then 3 counterclaims, 3 doubts (and 3 random alternatives in the full run). |
| 7 | `metaqa.py` | matched set | `metaqa.jsonl` | MetaQA baseline: synonym and antonym statements and their verification. |
| 8 | `probe_scores.py` | probes | `probe_scores.jsonl` | NLI decides which replies abandoned the answer. |
| 9 | `assemble.py` | all of the above | `scores.csv` | One table: one row per question, one column per method score. Open this in Excel to inspect results. |
| 10 | `evaluate.py` | `scores.csv` of both models | `../summary.json` | Every number in the paper. |

Helpers: `paths.py` (where every file lives) and `timing.py` (records stage times in `timings.json`).

`pipeline.py` runs the stages in this order. GPU stages run in a separate process per
model so GPU memory is freed between them. `cli.py` provides the `run`, `stage` and
`report` commands.

## Scores (`src/pressure_probe/scoring/`)

| File | What it computes |
| --- | --- |
| `baselines.py` | Token log probability, P(True), verbal confidence, Semantic Entropy clusters and entropy, SelfCheckGPT score. |
| `cps.py` | `flip_rate`: the doubt flip rate DFR is `flip_rate` of the doubt replies. **This is the main method.** `contrastive_pressure` gives the CPS variant (counterclaim flip rate minus doubt flip rate). |
| `metaqa_score.py` | MetaQA score from the Yes, No, Not Sure verdicts. |
| `combine.py` | Rank mean of DFR (or CPS) and the cross family score; out of fold logistic regression (reference only). |

## Statistics

| File | What it computes |
| --- | --- |
| `metrics.py` | AUROC, bootstrap intervals, DeLong test, Holm correction, AURAC, median imputation. |
| `analysis.py` | Uses `metrics.py` on the score tables: counts, results per method, comparisons of DFR with every baseline, flip rates, per dataset results, ablations, hypothesis verdicts. |

## Paper building (`src/pressure_probe/report/`)

| File | What it does |
| --- | --- |
| `build.py` | Runs the steps below in order. |
| `values.py` | Pulls single numbers from `summary.json`, authors from `paper/authors.yaml`, kappa from `human_labels.csv`. |
| `fmt.py` | Formats numbers (3 decimals, intervals, p values). Missing values become a yellow `[todo]`. |
| `sentences.py` | Writes the factual result sentences (no interpretation). |
| `tables.py` | Builds every table as LaTeX. |
| `figures.py` | Draws the flip rate and cost figures (PDF and PNG). |
| `render.py` | Replaces `\RES{}`, `\RESTXT{}`, `\RESTABLE{}`, `\RESFIG{}` in `paper/main.template.tex` and writes `paper/main.tex`. |
| `outputs.py` | Compiles the PDF, makes the Word file, checks page count and fonts, writes the ZIP. |

## Other folders

| Path | What it is |
| --- | --- |
| `notebooks/` | Colab versions of steps 1 to 7: `smoke_test`, `pilot_experiment`, `full_experiment`. |
| `scripts/make_label_sheet.py` | Creates the 100 answer sheet for the human check (step 6). |
| `scripts/make_notebooks.py` | Regenerates the three notebooks if you change them. |
| `tests/` | Offline tests on the simulated model (step 2). |
| `tools/check_writing.py` | Checks the paper text for listed AI style words and dashes. |
| `paper/main.template.tex` | The paper you edit. |
| `paper/main.tex` | Generated from the template; do not edit. |

# 4. Where to look when something goes wrong

| Problem | Look at |
| --- | --- |
| A stage crashed | `results/<run>/logs/pipeline.log`, last lines |
| Out of GPU memory | lower `batch_size` in the config, rerun |
| Answers look strange | `generations.jsonl` (raw model output) |
| A judgment seems wrong | `judgments.jsonl`, field `method` shows how it was decided |
| Too few CW answers | `split.jsonl`, or the counts printed by `run/04_pilot.sh` |
| A number in the paper | `summary.json` (search for the method name) |
| A `[todo]` that should be filled | `paper/validation_report.json`, `unresolved_tokens` |

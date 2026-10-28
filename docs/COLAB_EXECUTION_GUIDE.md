# Colab Execution Guide

This guide takes you from the ZIP file to a compiled IEEE paper with your measured
results in it. You run three notebooks in order. Each one writes to Google Drive and
skips finished work, so a dropped session costs only the stage that was running.

## What you need

| Item | Why | How |
| --- | --- | --- |
| Google account with Colab | runs the notebooks | colab.research.google.com |
| GPU runtime | the models run on GPU | Runtime > Change runtime type > T4 GPU (free) or L4/A100 (Colab Pro) |
| Hugging Face token | downloads models and data | huggingface.co > Settings > Access Tokens > New token (read) |
| OpenAI API key (optional) | GPT-4o-mini correctness judge, a few dollars | platform.openai.com > API keys |
| About 15 GB free on Google Drive | results and model cache | |

Put the two keys into Colab's **Secrets** panel (key icon in the left bar) as
`HF_TOKEN` and `OPENAI_API_KEY`, and switch on notebook access for both.

## Step 0: Put the code where Colab can reach it

Choose one:

- **GitHub (recommended).** Create a new empty repository, then on your computer:

  ```bash
  unzip agreement-is-not-truth.zip && cd agreement-is-not-truth
  git init && git add . && git commit -m "Initial code for consistent hallucination study"
  git branch -M main
  git remote add origin https://github.com/<your-account>/agreement-is-not-truth.git
  git push -u origin main
  ```

  The commit contains only your own name as author. Nothing in the code or the
  notebooks adds another author or a co-author line. Then set `REPO_URL` in the first
  cell of each notebook.
- **Drive.** Upload the ZIP to the top of your Drive as `agreement-is-not-truth.zip`.
  Leave `REPO_URL` unchanged and the notebook unzips it.

## Step 1: Smoke test (Mode A), about 15 minutes

Open `notebooks/smoke_test.ipynb` in Colab and choose Runtime > Run all.

It installs everything, runs the offline test suite, runs every stage with two tiny
models on 24 questions, compiles a paper and builds a ZIP.

**Pass when:** the tests all pass, every stage cell finishes, and the last cell
prints a validation report with `pages` filled in. The smoke numbers are meaningless;
delete the smoke paper afterwards.

If a cell fails, copy the full error text and send it to whoever maintains the code.
The most common causes are listed in the troubleshooting table at the end.

## Step 2: Pilot (Mode B), about 1 to 2 hours on a T4

Open `notebooks/pilot_experiment.ipynb` and run all cells. It uses the real models
on 200 questions.

When it finishes, open `results/pilot/summary.json` (on Drive, in
`agreement-is-not-truth-results/pilot/`) or the pilot PDF and check:

| Check | Where | Go if |
| --- | --- | --- |
| CW count per model | Table of counts, or `models.<key>.counts.CW` | at least 15 |
| Pressure signal | `models.<key>.methods` (`dfr`, `f_counter`, `cps`) | at least one pressure score above 0.6 for at least one model |
| Runtime | `models.<key>.timings` (seconds per stage) | the full run fits your GPU budget |

**Sizing the full run (done on 27 October 2026).** From the pilot's CW rates and
stage times, `configs/full.yaml` uses the 717 TruthfulQA questions not in the pilot,
200 TriviaQA and 200 SciQ questions, and caps the matched set at 120 CW per model.
The pilot questions are excluded automatically. Estimated GPU time on a free T4:
about 8 to 9 hours, spread over several sessions.

**If the pilot fails the go rule:** do not run the full experiment. Keep the pilot
results and report back; the prompts or threshold may need a change, which is cheaper
to test on the pilot.

## Step 3: Full experiment (Mode C), about 4 to 10 GPU hours

Open `notebooks/full_experiment.ipynb`. Run the cells from the top. The stage cells
run in this order:

| Stage | What it does | GPU | Rough share of time |
| --- | --- | --- | --- |
| prepare | downloads questions, saves the sample ids and environment | no | minutes |
| generate | greedy answer, 10 samples, P(True), verbal confidence | yes | largest |
| judge | labels each answer correct or wrong | API or GPU (NLI) | small |
| sample_scores | NLI over samples: split, Semantic Entropy, SelfCheckGPT, cross family | yes | medium |
| split | CW, CR, INC and the matched set | no | seconds |
| probe | alternative answer, counterclaims, doubts, random alternative | yes | medium |
| metaqa | MetaQA mutations and verification | yes | medium |
| probe_scores | NLI decides which replies flipped | yes | small |
| evaluate | every statistic in the paper, into `summary.json` | no | minutes |

**If Colab disconnects:** reconnect, run cells 1 to 5 again (settings, Drive, install,
keys), then rerun the stage cell that was interrupted. It continues from the last saved
chunk of 64 questions.

## Step 4: Human labels (needed once, about 2 hours of reading)

The label cell writes `human_labels.csv` with 100 answers in random order. Two
authors each fill one column without looking at the other's:

- `rater1`: 1 if the model answer is correct, 0 if wrong
- `rater2`: the same, from the second author

Save the file back to the same folder on Drive under the same name. The report reads
it and prints Cohen's kappa in the paper. If kappa is below 0.7, discuss the
disagreements and tell the code maintainer, because it means the judge needs a closer
look.

## Step 5: Authors

Edit `paper/authors.yaml` with names, affiliations and emails (in the repository on
GitHub, or in the unzipped folder on Drive). IEEE BigData review is single blind, so
names belong in the submission.

## Step 6: Write the results into the paper

Run the LaTeX install cell, then the report cell. It:

1. reads `results/full/summary.json`, the only source of numbers;
2. draws the two data figures;
3. builds every table;
4. writes the factual result sentences in the abstract, results, ablations, discussion and conclusion;
5. renders `paper/main.tex` from `paper/main.template.tex`;
6. compiles `paper/paper.pdf` with the IEEE class;
7. writes `paper/paper.docx` with remaining placeholders in yellow;
8. checks page count (limit 10 including references), undefined citations, Type 3 fonts and remaining `[todo]` marks;
9. writes `dist/full_package.zip`.

Open `paper/validation_report.json`. `ready_to_submit` turns true only when there are
no `[todo]` marks, the paper fits in 10 pages, every citation resolves and no Type 3
fonts are present.

## Step 7: What the authors still write by hand

Three yellow boxes marked **[todo: author review]** stay until you replace them in
`paper/main.template.tex`:

1. the interpretation paragraph in the Discussion;
2. the paragraph linking doubt flip rates to the sycophancy literature;
3. the Acknowledgment, including the AI use statement IEEE requires.

Delete each `\REVIEW{...}` block and write your text in its place, then run the report
cell again. Never type a number into the template; numbers come from `summary.json`.

## Step 8: Final checks before submitting

Follow `docs/SUBMISSION_CHECKLIST.md`. Submission itself is done by hand through the
CyberChair link; nothing in this project submits automatically.

## Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| `No CUDA GPU found` | CPU runtime | Runtime > Change runtime type > GPU |
| `vLLM unavailable ... falling back` | vLLM does not support the GPU or failed to install | nothing; transformers is used and recorded |
| CUDA out of memory in `generate` | large model on a 16 GB GPU without 4-bit | the code loads 4-bit below 20 GB; if it still fails lower `batch_size` in the config to 8 |
| 401 or 403 when downloading a model | missing or wrong `HF_TOKEN` | add the token in Secrets and rerun the keys cell |
| `OPENAI_API_KEY is not set` with `judge.backend: openai` | key missing | add the key, or set `judge.backend: auto` |
| LaTeX error in the report cell | package missing | rerun the LaTeX install cell; the error lines are printed |
| Paper over 10 pages | longer author text | shorten author written paragraphs; the generated parts are fixed in length |
| A stage seems to do nothing | its output already exists | delete that stage's output file in `results/<run>/<model>/` to redo it |

# Project Plan: Contrastive Pressure for Consistent Hallucinations

## Goal

A reproducible study that tests whether pushing back on an LLM answer separates
consistently wrong (CW) answers from consistently right (CR) answers, using only text
output and no training. The repository runs end to end on Google Colab and writes
measured results straight into an IEEE conference paper.

## Phases

| Id | Phase | Verification | Blocked by |
| --- | --- | --- | --- |
| P1 | Venue rules, writing checklist | Rules recorded in SUBMISSION_CHECKLIST.md | none |
| P2 | Dataset, model and citation verification | DATASET_DOCUMENTATION.md, CITATION_VERIFICATION_REPORT.md | P1 |
| P3 | Codebase: data, backends, stages, metrics | unit tests in `tests/test_metrics.py`, `test_scoring.py`, `test_text_split_config.py` pass | P2 |
| P4 | Reporting pipeline and paper template | `tests/test_pipeline_mock.py` and `tests/test_report.py` render, compile and count pages | P3 |
| P5 | Colab notebooks | notebooks parse with nbformat; smoke mode runs on the mock backend | P3, P4 |
| P6 | Documentation set | all files present, docx files open | P3, P4 |
| P7 | Tests, writing scan, review, package | TEST_REPORT.md from real pytest output; ZIP built | P3 to P6 |

## Decisions

| Decision | Choice | Reason |
| --- | --- | --- |
| Target models | Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct | Same families as Tan et al. (2025), so results are comparable; two families check each other |
| Low memory GPUs | AWQ 4-bit checkpoints chosen automatically under 20 GB | Colab T4 has 16 GB |
| Datasets | TruthfulQA (generation), TriviaQA (rc.nocontext validation), SciQ (test) | Myths plus the two datasets used by Tan et al. |
| Consistency rule | at least 9 of 10 samples equivalent to the greedy answer | 10 of 10 reported as a sensitivity check |
| Equivalence | DeBERTa-large-MNLI bidirectional entailment | Standard since Kuhn et al. (2023) |
| Judge | GPT-4o-mini when a key exists; alias match plus NLI otherwise | Cost; judge method is recorded in every row |
| Matched set | all CW plus an equal random sample of CR | Halves probing cost; balanced AUROC |
| Primary metric | AUROC on CW versus CR | Threshold free; matches prior work |
| Statistics | stratified bootstrap 95% CI, DeLong paired test, Holm correction | Paired design on identical items |
| Paper numbers | rendered from `summary.json` into `main.tex` | No number is typed by hand |

## Always / Never

- Always: identical questions and samples for every method; seeds recorded; raw outputs kept.
- Always: unresolved numbers render as yellow `[todo]`.
- Never: typed or estimated numbers in the paper; tuning on the evaluation items.
- Never: assistant names, generated-by banners or co-author trailers in code or commits.

## Assumptions

1. The team has a Colab GPU (T4 or better) and a Hugging Face token with Llama access.
2. An OpenAI key for GPT-4o-mini is optional; without it the fallback judge runs.
3. Author names and affiliations are filled in by the team.
4. The S25 submission track follows the main call: 10 pages including references, single blind.

## Progress

Updated 22 October 2026: 62 offline tests pass; placeholder paper builds at 6 pages with no overfull boxes.

| Id | Status |
| --- | --- |
| P1 | done |
| P2 | done |
| P3 | done |
| P4 | done |
| P5 | done |
| P6 | done |
| P7 | done |
| Smoke run on GPU | done (27 Oct) |
| Pilot on GPU | done (27 Oct); protocol version 2 fixed |
| Full run on GPU | pending (authors, Colab) |
| Author fields, human labels, review boxes, AI use statement | pending (authors) |

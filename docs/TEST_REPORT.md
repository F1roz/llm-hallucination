# Test Report

Run on 22 October 2026 with `python -m pytest -o addopts="" -v -p no:cacheprovider`.

**Environment:** Linux, Python 3.13.16, pytest 9.1.1, numpy 2.5.3, pandas 3.0.5,
scipy 1.18.1, scikit-learn 1.9.1, matplotlib 3.11.2, latexmk and pandoc present.
No GPU; torch, transformers and vLLM not installed.

**Result:** 62 collected, 62 passed, 0 failed, 0 skipped, in 48 s.

All pipeline tests use the simulated backend (`backends/mock_backend.py`), the
simulated NLI scorer and the synthetic fixture `tests/fixtures/questions.jsonl`.
Passing them shows the code paths, file formats, statistics and paper build work. It
says **nothing** about the behaviour of real models.

## Results by file

| File | Tests | Passed | What is covered |
| --- | --- | --- | --- |
| `test_metrics.py` | 12 | 12 | AUROC equals scikit-learn; single class gives NaN; DeLong AUROCs match scikit-learn, identical scores not significant, large gap detected, standard error agrees with bootstrap; bootstrap interval contains the point and is deterministic; Holm on known values, monotone and capped; AURAC ordering; median imputation |
| `test_scoring.py` | 14 | 14 | CPS subtracts doubt and uses the first k replies; empty flip rate is NaN; MetaQA mapping follows Yang et al. Eq. 5; Semantic Entropy for one and two clusters; SelfCheckGPT normalisation; P(True) token handling; verbal parsing and clipping; log probability score; rank mean; out of fold logistic shape; MetaQA parsing |
| `test_text_split_config.py` | 16 | 16 | answer cleaning; normalisation; whole word alias match; refusal detection; CW, CR, INC and EXCL assignment; matched sample balanced within dataset and deterministic; all four configs load; invalid threshold rejected; fingerprint ignores results folder |
| `test_pipeline_mock.py` | 8 | 8 | every stage writes its output; split matches fixture design; matched set balanced; simulated pliable errors get higher CPS; sampling baselines uninformative on the consistent split; summary has every section; rerun is idempotent; resume after interrupted generation |
| `test_report.py` | 7 | 7 | rendering without results marks every result token; rendering replaces known values; tables use placeholders without results; Cohen's kappa on known values; full report from simulated results resolves all result tokens (PDF and Word built, yellow highlights present); report without results keeps placeholders; archive contains paper and results |
| `test_repository.py` | 5 | 5 | no assistant attribution strings in code, notebooks or docs; three notebooks are valid JSON with the expected cells; paper prose passes the writing scan |

## Not tested here (blocked without a GPU)

| Component | Status | How it gets tested |
| --- | --- | --- |
| `backends/vllm_backend.py` | blocked | smoke notebook, step 6 |
| `backends/hf_backend.py` | blocked | smoke notebook if vLLM is unavailable |
| `backends/openai_backend.py` (judge) | blocked, needs API key | smoke notebook with `OPENAI_API_KEY` set; otherwise NLI fallback is used |
| `nli.DebertaNLI` with the real model | blocked | smoke notebook, judge and scoring stages |
| Dataset download at pinned revisions | blocked, network | prepare stage in any notebook |
| Behaviour of real models (CW counts, CPS signal) | not testable offline | pilot notebook go or no go rule |

Package versions on PyPI on 22 October 2026 (vLLM 0.32, transformers 5.20) are newer
than the versions the backend code was written against. Both backends have fallbacks
(vLLM falls back to transformers; argument errors are caught for `top_k`). The smoke
test is the first real check of these paths and must pass before the pilot.

## Raw output (abridged to status lines)

```text
tests/test_metrics.py::test_auroc_matches_sklearn PASSED
tests/test_metrics.py::test_auroc_single_class_is_nan PASSED
tests/test_metrics.py::test_delong_auc_values_match_sklearn PASSED
tests/test_metrics.py::test_delong_identical_scores_not_significant PASSED
tests/test_metrics.py::test_delong_detects_large_difference PASSED
tests/test_metrics.py::test_delong_standard_error_agrees_with_bootstrap PASSED
tests/test_metrics.py::test_bootstrap_interval_contains_point PASSED
tests/test_metrics.py::test_bootstrap_is_deterministic PASSED
tests/test_metrics.py::test_holm_known_values PASSED
tests/test_metrics.py::test_holm_is_monotone_and_capped PASSED
tests/test_metrics.py::test_aurac_perfect_ranking_beats_reversed PASSED
tests/test_metrics.py::test_impute_uses_median_and_counts PASSED
tests/test_pipeline_mock.py::test_every_stage_wrote_output PASSED
tests/test_pipeline_mock.py::test_split_matches_fixture_design PASSED
tests/test_pipeline_mock.py::test_matched_set_is_balanced PASSED
tests/test_pipeline_mock.py::test_pliable_errors_get_higher_cps PASSED
tests/test_pipeline_mock.py::test_sampling_baselines_are_uninformative_on_consistent_split PASSED
tests/test_pipeline_mock.py::test_summary_contains_all_sections PASSED
tests/test_pipeline_mock.py::test_rerun_is_idempotent PASSED
tests/test_pipeline_mock.py::test_resume_after_partial_generation PASSED
tests/test_report.py::test_render_without_results_marks_every_result_token PASSED
tests/test_report.py::test_render_replaces_known_values PASSED
tests/test_report.py::test_tables_without_results_use_placeholders PASSED
tests/test_report.py::test_cohen_kappa_known_values PASSED
tests/test_report.py::test_report_from_mock_results_resolves_all_result_tokens PASSED
tests/test_report.py::test_report_without_results_keeps_placeholders PASSED
tests/test_report.py::test_archive_contains_paper_and_results PASSED
tests/test_repository.py::test_no_assistant_attribution_in_code PASSED
tests/test_repository.py::test_notebooks_are_valid[smoke_test] PASSED
tests/test_repository.py::test_notebooks_are_valid[pilot_experiment] PASSED
tests/test_repository.py::test_notebooks_are_valid[full_experiment] PASSED
tests/test_repository.py::test_paper_prose_passes_style_scan PASSED
tests/test_scoring.py::test_cps_subtracts_doubt PASSED
tests/test_scoring.py::test_cps_uses_first_k PASSED
tests/test_scoring.py::test_flip_rate_empty_is_nan PASSED
tests/test_scoring.py::test_metaqa_mapping_follows_paper PASSED
tests/test_scoring.py::test_semantic_entropy_zero_for_one_cluster PASSED
tests/test_scoring.py::test_semantic_entropy_two_equal_clusters PASSED
tests/test_scoring.py::test_selfcheck_normalises_contradiction PASSED
tests/test_scoring.py::test_ptrue_handles_case_and_spaces PASSED
tests/test_scoring.py::test_ptrue_missing_tokens_is_nan PASSED
tests/test_scoring.py::test_verbal_parsing_and_clipping PASSED
tests/test_scoring.py::test_logprob_score_is_negative_mean PASSED
tests/test_scoring.py::test_rank_mean_order PASSED
tests/test_scoring.py::test_logistic_out_of_fold_shape PASSED
tests/test_scoring.py::test_parse_numbered_and_verdicts PASSED
tests/test_text_split_config.py::test_clean_answer_strips_prefix_and_period PASSED
tests/test_text_split_config.py::test_normalize_removes_articles_and_punctuation PASSED
tests/test_text_split_config.py::test_alias_match_whole_words_only PASSED
tests/test_text_split_config.py::test_refusal_detection PASSED
tests/test_text_split_config.py::test_assign_group[False-1.0-CW] PASSED
tests/test_text_split_config.py::test_assign_group[True-0.9-CR] PASSED
tests/test_text_split_config.py::test_assign_group[True-0.8-INC] PASSED
tests/test_text_split_config.py::test_assign_group[False-0.5-INC] PASSED
tests/test_text_split_config.py::test_assign_group[None-1.0-EXCL] PASSED
tests/test_text_split_config.py::test_matched_sample_balanced_within_dataset_and_deterministic PASSED
tests/test_text_split_config.py::test_configs_load[smoke] PASSED
tests/test_text_split_config.py::test_configs_load[pilot] PASSED
tests/test_text_split_config.py::test_configs_load[full] PASSED
tests/test_text_split_config.py::test_configs_load[test] PASSED
tests/test_text_split_config.py::test_invalid_threshold_rejected PASSED
tests/test_text_split_config.py::test_fingerprint_ignores_results_root PASSED
============================= 62 passed in 48.05s ==============================
```

## Update, 27 October 2026: real model check and fixes

After the Colab smoke test, the Qwen 7B pilot ran out of GPU memory in `generate`. Fixes:

1. Probability tables over the vocabulary are kept only where a method needs them (greedy answer, P(True)); samples are generated in batches of at most 64 sequences.
2. On a GPU out of memory error, the generator and the NLI model split the batch in half and retry instead of stopping.
3. Each model of a GPU stage runs in its own process when a stage is started from a notebook cell, so the first model's memory is fully released before the second loads.
4. `remove_invalid_values` guards against inf or nan logits in 16 bit arithmetic.
5. `delong_test`, the paired bootstrap and `holm` return missing values instead of failing when a class has fewer than two items; the figures skip missing values.

**Real model check (CPU, this machine):** every stage from `prepare` to `evaluate`, then `report`, was run with the pilot settings (10 samples, 3 challenges, MetaQA with 5 mutations, batch size 16) on 18 real questions from the three pinned datasets, with Qwen2.5 0.5B and 1.5B and the real DeBERTa NLI model. One model produced no consistently wrong answers, which exposed fix 5. After the fixes all stages and the report complete. The Llama 3.1 8B tokenizer, chat format, padding and end tokens were checked separately. Not checkable here: GPU memory and speed of the 7B and 8B models on a T4.

Offline suite after these changes: 68 tests, all passed.

## Update, 27 October 2026 (evening): protocol version 2

Changes tested: the doubt flip rate (DFR) as the main score with CPS and the counterclaim
flip rate as reported variants; new hypotheses H2 to H4; exclusion of the pilot sample
from the full run; a cap of 120 consistently wrong items per model in the matched set;
`prepare` refusing to reuse a results folder drawn with different question settings.

New tests: DFR is the main score and is compared with every baseline; the matched set
cap is respected and deterministic; an earlier sample is excluded with no overlap;
`prepare` refuses changed question settings.

Checks beyond the test suite:

- On the real pinned datasets, the full run draws 1,117 questions (717 TruthfulQA, 200
  TriviaQA, 200 SciQ) with **no overlap** with the pilot's 200.
- The changed stages (`split`, `probe`, `metaqa`, `probe_scores`, `evaluate`) and the
  report were rerun on the real small model outputs from the CPU check; all completed and
  the paper compiled with no unresolved slots, citations or references.
- The paper rendered from the simulated run compiles to 7 pages with no overfull boxes.

Offline suite: **72 tests, all passed.**

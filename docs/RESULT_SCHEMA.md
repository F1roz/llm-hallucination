# Result Schema

`results/<run>/summary.json` is the single source of truth for the paper. Fields:

```text
run_name, mode, config_fingerprint, created_utc
settings: n_samples, sample_temperature, consistency_threshold, n_pressure,
          metaqa_mutations, bootstrap_samples, datasets, nli_model, seed
models: { <model_key>: {
  display_name
  backend:        { backend, precision }
  judge_methods:  { alias | llm:<model> | nli | refusal : count }
  timings:        { <stage>: { seconds, generations } }
  counts:         { total, CW, CR, INC, EXCL, strict_CW, strict_CR,
                    per_dataset: { <dataset>: { CW, CR, INC, EXCL, total } } }
  matched:        { n, CW, CR }
  methods:        { <method>: { label, access, is_ours, calls, n_missing,
                                auroc_matched: { value, low, high },
                                auroc_all, aurac_all } }
  best_text_baseline, best_baseline
  comparisons_vs_cps: { <baseline>: { z, p, p_holm, difference: { value, low, high } } }
  flip_rates:     { CW|CR: { f_counter|f_doubt|cps: { value, low, high, n } } }
  per_dataset:    { <dataset>: { n_cw, n_cr, cps, baseline } }
  ablations:      { pressure_count: { "1"|"2"|"3": { cps, f_counter, calls } },
                    signal: { cps, f_counter, f_doubt, cps_minus_f_counter },
                    strict: { n_cw, n_cr, cps, baseline },
                    random_alternative: { n, cps_self, cps_random },
                    alternative_equivalent_count, excluding_equivalent_alternatives,
                    combination: { cps, cross_family, rank_mean, logistic_cv } }
  hypotheses:     { H1..H4: { ..., supported } }
  examples:       [ { qid, group, question, reference, answer, alternative, reply,
                      f_counter, f_doubt } ]
} }
```

Per model files in `results/<run>/<model_key>/`:

| File | One row per | Key fields |
| --- | --- | --- |
| `generations.jsonl` | question | greedy, greedy_logprobs, samples, ptrue_first_token, verbal_raw |
| `judgments.jsonl` | question | correct, refusal, method |
| `sample_scores.jsonl` | question | matches_greedy, frac_match, semantic_entropy, selfcheck_nli, cross_family |
| `split.jsonl` | question | group, strict_group, frac_match, correct |
| `matched_ids.json` | list | matched set ids |
| `probes.jsonl` | matched item | alternative, counter_replies, doubt_replies, random_counter_replies |
| `metaqa.jsonl` | matched item | synonyms, antonyms, verdicts |
| `probe_scores.jsonl` | matched item | counter_flips, doubt_flips, random_counter_flips, alternative_equivalent |
| `scores.csv` | question | every score column used by the analysis |

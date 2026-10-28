# Experimental Protocol (version 2)

Version 2 was fixed on 27 October 2026, after the pilot and before the full run. The
change log at the end lists every change from version 1 with its reason, so readers
can tell planned analyses from later ones. Pilot items are excluded from the full run
and no pilot number is reported in the paper.

## 1. Problem statement

Detectors that sample a model several times treat disagreement as a sign of
hallucination. When a model gives the same wrong answer every time, these detectors
see agreement and miss the error. Tan et al. (EMNLP 2025) showed that such errors stay
level or grow with model size and that Semantic Entropy scores near chance on them.
Their remedy needs hidden states of two models and labelled training data.

**Question.** Using only generated text and no training, does the rate at which a
model abandons its answer under plain doubt ("Are you sure?") separate consistently
wrong answers from consistently right ones? And does correcting for that general
yielding, as the contrastive score CPS does, help or hurt?

## 2. Hypotheses (fixed 27 October 2026, before the full run)

| Id | Question | Criterion for support |
| --- | --- | --- |
| H1 | Do sampling detectors fail on CW versus CR? | 95% bootstrap intervals of Semantic Entropy and SelfCheckGPT-NLI AUROC both include 0.5 |
| H2 | Does the doubt flip rate (DFR) beat sampling detectors? | DFR AUROC above both Semantic Entropy and SelfCheckGPT-NLI, each with Holm adjusted DeLong p below 0.05, and the lower end of the DFR 95% interval above 0.5 |
| H3 | Does subtracting general yielding remove the signal? | DFR AUROC above CPS AUROC, with the paired bootstrap 95% interval of the difference above 0 |
| H4 | Does cross family disagreement add to DFR? | Rank mean of DFR and X above both components |

Each criterion is evaluated per model and printed in `summary.json` under
`hypotheses`. The paper states which criteria were met; it does not reinterpret a
missed criterion as support. The comparison of DFR with the strongest text only
baseline (in the pilot, cross family disagreement for one model) is reported with its
p value but is not a criterion.

**The main hypothesis is not supported if** DFR does not exceed both sampling
detectors with Holm adjusted p below 0.05, or its interval includes 0.5.

## 3. Variables

| Kind | Variables |
| --- | --- |
| Independent | detection method (11 scores: 7 baselines, DFR, DFR + X, F_C, CPS); target model (2); dataset (3); number of doubt prompts k (1, 2, 3); consistency threshold (0.9, 1.0) |
| Dependent | AUROC on CW vs CR (primary); AUROC on all answers; AURAC; generations per question; wall clock seconds |
| Controlled | questions and sample ids; prompts; decoding settings; NLI model; judge; seeds; matched set; imputation rule |
| Possible confounders | judge errors; NLI errors on short answers; model specific sycophancy; quantization on small GPUs; dataset mix (controlled by matching within dataset) |

## 4. Design

Paired, within item design: every method scores the same items, and where a method
uses samples, the same samples. This is what makes the DeLong test and paired
bootstrap valid.

### 4.1 Data

| Dataset | Split | Full run size | Selection |
| --- | --- | --- | --- |
| TruthfulQA | generation / validation | 717 | all questions not used by the pilot |
| TriviaQA | rc.nocontext / validation | 200 | seeded random sample, pilot items removed first |
| SciQ | test | 200 | seeded random sample, pilot items removed first |

The pilot used 100, 50 and 50 items drawn with the same seed. `exclude_sample` in
`configs/full.yaml` recreates that draw and removes it before sampling; a test checks
that the two sets do not overlap (verified on the real datasets on 27 October 2026:
overlap 0). Sample ids are saved in `results/<run>/questions.jsonl`; dataset revisions
are pinned in `src/pressure_probe/data.py`.

### 4.2 Models

Qwen2.5-7B-Instruct and Llama-3.1-8B-Instruct (revisions pinned in
`configs/full.yaml`). On GPUs with less than 20 GB the code loads 4-bit weights (AWQ
through vLLM, NF4 through transformers). The precision used is recorded per run and
reported in the paper.

### 4.3 Generation

- System prompt: "Answer with a short phrase only. Do not explain."
- Greedy answer at temperature 0; 10 samples at temperature 0.5, top-p 1, top-k off,
  repetition penalty 1; at most 32 new tokens.
- Challenge replies are greedy, in a second turn after the model's own answer; each
  challenge is a separate conversation.

### 4.4 Labels

1. Refusal filter: excluded from all analyses.
2. Alias match after normalisation (whole words).
3. Otherwise GPT-4o-mini with correct and incorrect references; if no API key, NLI
   bidirectional entailment against each reference.
4. Two authors label 100 consistent answers (50 CW, 50 CR, shuffled); Cohen's kappa
   reported.

### 4.5 Split and matched set

- Agreement r(q): share of samples equivalent to the greedy answer (DeBERTa-large-MNLI,
  bidirectional entailment with the question prepended).
- CW: r at least 0.9 and wrong. CR: r at least 0.9 and right. INC: otherwise.
- Matched set: up to 120 CW items per model (a seeded random sample when there are
  more), plus the same number of CR items sampled within each dataset with a fixed seed.

### 4.6 Methods

| Method | Role | Access | Generations per question |
| --- | --- | --- | --- |
| Token log probability | baseline | logits | 1 |
| P(True) | baseline | logits | 2 |
| Verbalized confidence | baseline | text | 2 |
| Semantic Entropy | baseline | text | 11 |
| SelfCheckGPT-NLI | baseline | text | 11 |
| Cross family disagreement X | baseline | text | 2 |
| MetaQA (5 synonym, 5 antonym mutations) | baseline | text | 13 |
| **Doubt flip rate DFR = F_D (k = 3)** | **main** | text | 4 |
| DFR + X, rank mean | ours | text | 5 |
| Counterclaim flip rate F_C | variant | text | 5 |
| CPS = F_C − F_D | variant | text | 8 |

## 5. Statistical analysis

- Primary metric: AUROC on the matched set, CW positive.
- Uncertainty: stratified bootstrap, 2,000 resamples, 95% percentile intervals.
- Comparisons: DFR against each of the seven baselines with the paired DeLong test
  (fast midrank algorithm), Holm correction across the seven tests, and paired
  bootstrap intervals for AUROC differences as effect sizes. DFR against CPS (H3) by
  paired bootstrap interval.
- Secondary: AUROC and AURAC on all consistent and inconsistent answers, for context.
- Missing scores: replaced by the median of that method's scores (counted and stored).
  Tests that need two items per class return a missing value instead of failing.
- No threshold is tuned. The logistic regression is reported only as a reference
  ceiling with out of fold predictions (5 fold stratified cross validation).

## 6. Ablations

All reuse collected replies.

| Ablation | Question it answers |
| --- | --- |
| k = 1, 2, 3 doubt prompts | How many challenges are needed? |
| F_C alone, CPS | Does naming an alternative or subtracting doubt help (H3)? |
| Threshold 1.0 | Do results hold for the strictest definition of consistency? |
| Rank mean of DFR and X, of CPS and X; logistic regression | How much does combination or training add? |

## 7. Leakage and fairness controls

- Pilot items are excluded from the full run; pilot numbers are not reported.
- The alternative b is generated by the model and never taken from references.
- No threshold, weight or hyperparameter is fitted on evaluation items; the only
  fitted model is the out of fold reference regression.
- All methods receive the same items; budgets differ by design and are reported.
- Results are reported for every method and model, whether favourable or not.

## 8. Change log

| Date | Change | Reason |
| --- | --- | --- |
| 2026-10-22 | Version 1 fixed: CPS main score; H2 required CPS AUROC of at least 0.65 above the best text only baseline; H3 required CPS above F_C | before any data |
| 2026-10-27 | Pilot run (200 questions; Qwen 62 CW, Llama 33 CW). Version 1 go rule not met: CPS AUROC 0.53 (Qwen), 0.40 (Llama). On the same items F_D reached 0.69 and 0.65, F_C 0.70 and 0.57, Semantic Entropy 0.56 and 0.53 | pilot results; consistent errors flipped more under both kinds of challenge, so the subtraction in CPS cancelled the difference |
| 2026-10-27 | Version 2: DFR becomes the main score, DFR + X the combined score; CPS and F_C kept as reported variants; H2 to H4 rewritten as above | pilot result above; decided before the full run |
| 2026-10-27 | Pilot items excluded from the full run; sizes 717 / 200 / 200; matched set capped at 120 CW per model; random alternative ablation dropped | keep the confirmatory test clean; compute budget sized from pilot timings and CW rates |
| 2026-10-27 | Closest prior work added after a novelty check (QueRE, Pawitan and Holmes, Delusions, CHOKE, MUSE); see `NOVELTY_CHECK_OCT2026.md` | DFR is closer to existing observations than CPS was |

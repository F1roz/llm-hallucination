# Novelty Check for the Doubt Flip Detector (27 October 2026)

Question: has anyone already used "does the model abandon its answer when asked 'Are you sure?'"
as a training free detector of wrong answers, and in particular of consistent errors that
sampling based detectors miss?

Searched on 27 October 2026: arXiv, ACL Anthology, NeurIPS and ICML pages, publisher pages.
23 papers were read at abstract or full text level in this round, in addition to the 24 in
`LITERATURE_REVIEW.md` (32 counted below as directly relevant).

## Closest prior work

| Work | What it does | How we differ |
| --- | --- | --- |
| Sam, Finzi, Kolter. QueRE, NeurIPS 2025 (arXiv 2501.01558) | About 50 follow-up yes/no questions, including "Are you sure?", turned into features from token probabilities; a logistic regression trained on correctness labels predicts errors (AUROC up to 0.95 on NQ, SQuAD) | Needs labels and a trained model and uses probabilities; not evaluated on consistent errors. Ours is one training free rule on generated text, tested exactly where sampling fails. |
| Pawitan and Holmes. Harvard Data Science Review 2025 (arXiv 2412.15296) | "Please think again": kept answers are more accurate than changed ones (for example 0.93 against 0.38) | Reasoning puzzles, no detector evaluation, no AUROC, no comparison with sampling detectors, no consistent error split. |
| Xu et al. "Delusions of LLMs", arXiv 2503.06709 (2025) | High belief wrong answers ("delusions") are refused less and insisted on more under reflection than ordinary hallucinations | Compares two kinds of errors with each other, not errors against correct answers; no detector. Their result is a caution for us: the most entrenched errors may resist doubt. |
| Simhi et al. "Trust Me, I'm Wrong" (CHOKE), arXiv 2502.12964 (2025) | High certainty hallucinations are 10 to 40 percent of hallucinations; certainty based abstention misses them | No detector proposed; confirms the problem we target. |
| Guo et al. MUSE, arXiv 2605.27288 (2026) | Conformity to pushback rises with epistemic uncertainty, beyond sycophancy | Measurement framework, no detection. Supports our mechanism. |

## Studies of answer changes under pushback (none evaluates a detector)

Laban et al. FlipFlop (arXiv 2311.08596); Sharma et al. (ICLR 2024); Huang, "Anatomy of LLM
Sycophancy" (arXiv 2610.06522); Kim and Khashabi (arXiv 2509.16533, models switch less when
originally right); Aamir and Bin Adil (arXiv 2609.17550, TriviaQA pushback on Qwen2.5 1.5B and
Llama 3.2 1B; flip probes AUROC 0.55 to 0.58); Fanous et al. SycEval (arXiv 2502.08177;
progressive 43.5 percent against regressive 14.7 percent); He et al. multi-turn instability
(arXiv 2511.10688); Saadat and Nemzer, stability under self challenging prompts (arXiv 2603.03330); Li et al. "Firm
or Fickle?" (ACL Findings 2025); Hong et al. SYCON Bench (arXiv 2505.23840); Tang et al. SPINE
(arXiv 2609.09090); Xu et al. "The Earth is Flat because..." (ACL 2024); Huang, Kwak, An,
belief resistance (arXiv 2601.13590); Huang et al. "LLMs cannot self correct reasoning yet"
(ICLR 2024).

## Hallucination detection that assumes errors vary across samples or needs other access

Tan et al. (EMNLP 2025); Ganesh et al. (EACL 2026); Xue et al. (TMLR 2026); Zhang et al. SAC3
(EMNLP Findings 2023); Joo et al. (arXiv 2509.21999); Wilson and Akrout (arXiv 2605.05134);
Qiao et al. Lowest Span Confidence (arXiv 2601.19918); Zhang et al. multi-turn confidence
(arXiv 2601.02179); Yang et al. structural signals (WWW 2026, arXiv 2602.00977); Plaut et al.
(arXiv 2402.13213); Sychev et al. (arXiv 2503.01688).

## Conclusion

- Using answer changes under challenge as an error signal is **not new as an idea**
  (QueRE uses "Are you sure?" as one of many supervised features; Pawitan and Holmes show the
  correlation). The paper must say so plainly and cite both.
- **What is new:** (1) a training free doubt flip rate evaluated as a detector of consistent
  errors, the case where sampling detectors and certainty based abstention fail (Tan et al.;
  CHOKE; Delusions); (2) a paired comparison with seven baselines, including MetaQA, on identical
  matched items with cost; (3) the decomposition result: subtracting the doubt flip rate from the
  counterclaim flip rate (CPS), intended as a sycophancy control, removes the signal, so general
  yielding is itself informative for consistent errors.
- Risk named in the paper: Delusions suggests the most entrenched errors may resist doubt; the
  pilot (excluded from the paper) showed consistent errors flipping more than consistent correct
  answers, which the full run tests.

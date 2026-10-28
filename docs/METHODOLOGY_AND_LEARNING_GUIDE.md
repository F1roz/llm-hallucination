---
title: "Methodology and Learning Guide"
subtitle: "Agreement Is Not Truth: Detecting Consistent Hallucinations with Contrastive Pressure"
date: "22 October 2026"
---

# Update, 27 October 2026: the main score changed

Read this first. The pilot changed the method:

- **Main score now:** the doubt flip rate (DFR). After answering, the model hears three
  doubt prompts ("Are you sure? ..."), each in a separate conversation. DFR is the share
  of replies that abandon the answer. 4 generations per question.
- **Why:** in the pilot, consistently wrong answers gave way more than consistently right
  ones under both counterclaims and doubt. CPS subtracts one from the other, so the
  difference cancelled (CPS AUROC 0.53 and 0.40), while DFR alone reached 0.69 and 0.65.
- **CPS stays in the paper** as hypothesis H3: "subtracting general yielding removes the
  signal". Where this guide says CPS is the method, read it as the variant tested in H3.
- **New hypotheses** are in `EXPERIMENTAL_PROTOCOL.md` section 2. **New closest prior work**
  is in `NOVELTY_CHECK_OCT2026.md`.

Extra defence questions for the new design:

1. **Is it fair to change the main score after a pilot?** Yes, if the pilot items are
   excluded, the change is logged and the criteria are fixed before the confirmatory run.
   All three hold here, and the original score is still reported.
2. **Isn't "Are you sure?" already known to reveal errors?** The correlation is known
   (Pawitan and Holmes; QueRE uses it as a trained feature). What is new is a training
   free score evaluated on consistent errors, where sampling fails, against seven
   baselines on identical items, plus the finding about CPS.
3. **Why might entrenched errors still escape?** Xu et al. (Delusions) found that
   high belief errors resist reflection. DFR can only flag errors the model is willing to
   give up; the cross family check covers part of the rest.
4. **Does asking "Are you sure?" make answers worse?** Only the flip is read. The
   reply to doubt is never shown to the user, so the deployed answer cannot get worse.

# How to use this guide

Read Part A first, then Parts C and E. Part B explains each concept at three levels:
**Beginner** (the idea), **Intermediate** (how it works) and **Technical** (the
formula or the code). Part G prepares you for questions from reviewers, supervisors or
an audience. Part H lists what to study, essential first.

# Part A. Research foundations

## A1. The problem

Large language models (LLMs) sometimes state false facts fluently. This is called
hallucination. A popular way to detect it uses only the model's text: ask the same
question several times with some randomness, and if the answers disagree, the model
probably does not know. This works because a model that knows an answer puts most of
its probability on it.

The weak point is an error the model makes **the same way every time**. Asked what
percentage of the brain humans use, a model may answer "about 10%" in every sample.
Every sampling detector sees perfect agreement and passes the answer. Tan et al.
(EMNLP 2025) named these *self-consistent errors* and showed three things: they do not
shrink as models grow, Semantic Entropy scores near chance on them, and the one fix
that works reads the internal states of two models and needs labelled training data.

## A2. Why it matters

- Consistent errors are the most convincing errors: the model sounds sure every time.
- Many deployed models are only available through an API, so methods that need
  internal states cannot be used.
- Larger models do not remove the problem.

## A3. The research gap

No method that uses only text and no training has been shown to catch consistent
errors. Methods that perturb the answer (MetaQA, counterfactual probing) were never
tested on this error type, and none of them separates real evidence from a model's
general habit of agreeing with the user.

## A4. Our idea and contribution

Instead of asking the same question again, push back on the answer:

- **Counterclaim:** "I am fairly sure the answer is *b*, not *a*." where *b* is the
  model's own most plausible alternative.
- **Doubt:** "Are you sure? Please reconsider." with no alternative named.

The **Contrastive Pressure Score** is the share of answers abandoned under
counterclaims minus the share abandoned under doubt. Doubt measures how easily the
model gives in to anything; the counterclaim adds the pull of a specific competitor.
The hypothesis is that a memorised error is held less firmly than a known fact when a
plausible competitor is named.

Contributions (stated as aims, not results):

1. a text only, training free detector for consistent errors;
2. a matched evaluation set of consistently wrong (CW) and consistently right (CR) answers;
3. a fair comparison with seven baselines on identical items with paired tests;
4. ablations that isolate the parts of the method.

## A5. Background theory in brief

- **Uncertainty and correctness.** Detectors assume that uncertainty signals error.
  Consistent errors break this assumption, so any detector that reads only uncertainty
  will fail on them.
- **Sycophancy.** Models tuned with human feedback tend to agree with the user. This is
  usually studied as a failure. Here it is a measurement tool, with a control.
- **Paired evaluation.** When two methods score the same items, their errors are
  correlated. Paired tests use that correlation and are more sensitive than comparing
  two separate confidence intervals.

# Part B. Technical foundations

## B1. Language model decoding

- **Beginner.** The model writes one token (word piece) at a time, each chosen from a
  probability list.
- **Intermediate.** Greedy decoding always takes the most likely token. Sampling picks
  at random in proportion to probability; temperature T sharpens (T below 1) or flattens
  (T above 1) the list.
- **Technical.** p(token) is the softmax of logits divided by T. We use T = 0 (greedy)
  for the answer and T = 0.5 with top-p 1 for the 10 samples. Code: `stages/generate.py`.

## B2. Chat templates and multi turn prompts

- **Beginner.** Chat models expect a conversation format with roles.
- **Intermediate.** The pressure prompt is a conversation: system message, question,
  the model's own answer, then the challenge.
- **Technical.** `tokenizer.apply_chat_template(messages, add_generation_prompt=True)`
  builds the prompt string for each model's own format. Code: `prompts.pressure_messages`.

## B3. Natural language inference (NLI)

- **Beginner.** An NLI model reads two sentences and says whether the first implies
  the second, contradicts it, or neither.
- **Intermediate.** Two answers mean the same thing when each implies the other.
  Adding the question to both sides gives the short answers context.
- **Technical.** DeBERTa-large-MNLI outputs probabilities over (contradiction,
  neutral, entailment). E(x, y) = 1 when argmax is entailment for "q x"→"q y" and for
  "q y"→"q x" (Kuhn et al., 2023). Code: `nli.py`.

## B4. Semantic Entropy

- **Beginner.** Count how many different meanings appear among the samples.
- **Intermediate.** Group samples that mean the same thing; the more evenly spread the
  groups, the higher the entropy.
- **Technical.** SE = −Σ p_c log p_c over clusters c, with p_c the share of samples in
  cluster c. Clustering assigns each sample to the first cluster whose representative
  is equivalent. Code: `scoring/baselines.py`.

## B5. SelfCheckGPT-NLI

- **Technical.** For each sample s_i, P(contradiction)/(P(contradiction)+P(entailment))
  with premise "q s_i" and hypothesis "q a"; the score is the mean.

## B6. P(True) and verbalized confidence

- **P(True):** the model judges its own answer true or false; we read the probability
  of the "True" token from the first generated position, normalised against "False".
- **Verbalized:** the model states a number from 0 to 100; score = 1 − number/100.

## B7. MetaQA

- **Intermediate.** The model rewrites its answer as statements with the same meaning
  (synonyms) and as negations (antonyms), then judges each statement. A correct answer
  should get "Yes" for synonyms and "No" for antonyms.
- **Technical.** Synonym: Yes 0, No 1, Not Sure 0.5; antonym: Yes 1, No 0, Not Sure 0.5;
  score = mean (Yang et al., 2025, Eq. 5).

## B8. The Contrastive Pressure Score

- **Technical.** flip(x) = 1 − E(x, a). F_C = mean flip over k counterclaim replies,
  F_D = mean flip over k doubt replies, CPS = F_C − F_D. Range −1 to 1. Code:
  `scoring/cps.py`, `stages/probe.py`, `stages/probe_scores.py`.

## B9. AUROC

- **Beginner.** Pick one wrong and one right answer at random. AUROC is the chance the
  detector gives the wrong one the higher score. 0.5 is a coin flip, 1.0 is perfect.
- **Intermediate.** It does not need a threshold and is not affected by class balance
  in the same way as accuracy.
- **Technical.** AUROC equals the Mann Whitney U statistic divided by n₁n₀. Code:
  `metrics.auroc`.

## B10. Bootstrap confidence intervals

- **Beginner.** Resample the data many times and see how much the result moves.
- **Intermediate.** Stratified bootstrap resamples CW and CR separately, so every
  resample keeps the same class sizes.
- **Technical.** 2,000 resamples, 2.5th and 97.5th percentiles. Code:
  `metrics.bootstrap_auroc`.

## B11. DeLong test

- **Intermediate.** Tests whether two AUROCs on the same items differ, using the
  covariance between them.
- **Technical.** Structural components from midranks give the variance of each AUROC
  and their covariance; z = (A₁ − A₂)/√(V₁ + V₂ − 2C). Implementation follows Sun and
  Xu (2014). Tested against bootstrap standard errors in `tests/test_metrics.py`.

## B12. Multiple comparisons and Holm correction

- **Beginner.** Testing seven baselines gives seven chances to find a difference by luck.
- **Technical.** Sort p values ascending; multiply the i-th smallest by (m − i + 1),
  enforce monotonicity, cap at 1. Code: `metrics.holm`.

## B13. Cohen's kappa

- **Technical.** κ = (p_o − p_e)/(1 − p_e), where p_o is observed agreement and p_e the
  agreement expected by chance from each rater's label rates. Above 0.7 is usually
  taken as good agreement for binary labels.

## B14. AURAC

- **Technical.** Sort answers from least to most suspicious; for each cut keep the
  least suspicious answers and compute accuracy; AURAC is the mean over all cuts.

## B15. Quantization

- **Intermediate.** 4-bit weights fit a 7B model into a 16 GB GPU. They change the
  numbers slightly and can change some answers, so the precision is recorded.

## B16. Software stack

Python, Hugging Face Transformers and Datasets, vLLM (optional), scikit-learn, SciPy,
pandas, matplotlib, LaTeX (IEEEtran), pandoc.

# Part C. Methodology explanation

Each step lists what, why, how and what could go wrong.

1. **Prepare questions.** *What:* load the datasets at pinned revisions, sample with a
   seed, save ids. *Why:* reproducibility. *Risk:* a dataset revision disappears;
   the pinned hash makes this visible.
2. **Generate.** *What:* greedy answer, 10 samples, P(True), verbal confidence. *Why:*
   the samples are needed for the split and for sampling baselines; generating once
   means every method sees the same samples. *Risk:* long answers break the short
   answer assumption; the prompt asks for a short phrase and outputs are trimmed to the
   first line.
3. **Judge.** *What:* refusal filter, alias match, then an LLM judge or NLI. *Why:*
   correctness defines CW and CR. *Risk:* judge errors; checked with 100 human labels.
4. **Sample scores.** *What:* NLI between samples and the answer; Semantic Entropy,
   SelfCheckGPT, cross family score. *Risk:* NLI errors on very short answers.
5. **Split.** *What:* CW, CR, INC with τ = 0.9 and a matched set. *Why:* on this split
   sampling detectors have nothing to work with, which is exactly the case we study.
   Matching within dataset prevents a dataset difference from posing as a detector
   difference. *Risk:* too few CW items; the pilot checks the count.
6. **Probe.** *What:* the model proposes b; three counterclaims and three doubts are
   asked; in the full run a random alternative too. *Why:* this is the method.
   *Risk:* b repeats a (handled by retries and measured by an ablation); the model
   refuses or rambles (replies are trimmed and compared by NLI).
7. **MetaQA.** *What:* the strongest prior perturbation baseline, run on the same items.
8. **Probe scores.** *What:* NLI decides which replies flipped.
9. **Evaluate.** *What:* every statistic and the four hypothesis verdicts into
   `summary.json`. *Why:* one file feeds the paper, so text and tables cannot disagree.
10. **Report.** *What:* figures, tables, sentences, PDF, Word file, validation, archive.

# Part D. Implementation explanation

| Module | Role | Connects to |
| --- | --- | --- |
| `config.py` | loads and checks YAML settings; fingerprint | every stage |
| `data.py` | dataset loading, seeded sampling, validation | Part C step 1 |
| `prompts.py` | every prompt in one place | steps 2, 6, 7 |
| `backends/` | vLLM, transformers, OpenAI, simulated model | steps 2, 3, 6, 7 |
| `nli.py` | NLI scorer with caching and the equivalence rule | steps 3, 4, 8 |
| `text.py` | answer cleaning, normalisation, refusal detection | steps 2, 3 |
| `stages/*.py` | one file per pipeline stage | Part C steps 1 to 9 |
| `scoring/` | baseline scores, MetaQA score, CPS, combination | B4 to B8 |
| `methods.py` | method list, access type, generations per question | tables |
| `metrics.py` | AUROC, bootstrap, DeLong, Holm, AURAC | B9 to B14 |
| `analysis.py` | turns score tables into reported numbers and verdicts | step 9 |
| `report/` | values, sentences, tables, figures, rendering, PDF, Word, archive | step 10 |
| `pipeline.py`, `cli.py` | stage order, process isolation, command line | all |

The simulated backend (`backends/mock_backend.py`) and the synthetic fixture exist so
the whole pipeline can be tested without a GPU. They are never used for results.

# Part E. Experiments

- **Baselines and why.** Token log probability and P(True) represent methods that read
  the model's confidence; verbal confidence is the text only version; Semantic Entropy
  and SelfCheckGPT are the standard sampling detectors; cross family disagreement is
  the text only form of Tan et al.'s verifier idea; MetaQA is the closest perturbation
  method.
- **Controls.** Same items, same samples, same prompts, same NLI model, same judge,
  matched classes, fixed seeds.
- **Metrics.** AUROC (primary), AURAC, generations per question, wall clock time.
- **Statistics.** Bootstrap intervals, DeLong with Holm, paired bootstrap differences.
- **Ablations.** Number of challenges, each half of CPS, random alternative, strict
  threshold, alternatives that repeat the answer, learned versus rule combination.
- **Reproducibility.** Pinned revisions, saved ids, environment records, raw outputs.

# Part F. Reading the results

- **Counts table.** How many consistent errors each model makes. A low CW count means
  wide intervals.
- **Main table.** For each method: AUROC on CW vs CR with its interval, AUROC on all
  answers, and the Holm adjusted p value against CPS. A baseline can be strong on
  "All" and near 0.5 on "CW vs CR"; that contrast is the paper's first finding.
- **Interval reading.** If an interval contains 0.5 the method is not shown to beat
  chance. Two methods' intervals can overlap and still differ significantly in a
  paired test; trust the DeLong p value and the difference interval for comparisons.
- **Flip figure.** Bars for CW and CR under counterclaims and doubt. CPS works when the
  gap between counterclaim and doubt bars is larger for CW than for CR.
- **Cost figure.** AUROC against generations per question; methods to the upper left
  give more for less.
- **Hypothesis verdicts.** Printed as met or not met by the rules fixed in advance. A
  verdict of "not met" is a result, not a failure of the study.

# Part G. Research defense questions

1. **Why not just use Semantic Entropy?** On consistent answers all samples agree, so
   its value is near zero for both right and wrong answers. Tan et al. measured
   AUROC near 0.5; H1 tests this again on our data.
2. **Is CPS just sycophancy?** Raw flipping is partly sycophancy. That is why CPS
   subtracts the doubt flip rate, and H3 tests whether the subtraction helps.
3. **Why should a model give up a wrong answer more easily than a right one?** It is a
   hypothesis, not an assumption. The rationale: a memorised error is supported by
   weaker or conflicting evidence in training than a well known fact, so a named
   competitor can pull it away. The experiment tests this.
4. **What if the model is stubborn about a misconception?** Then CPS treats it like a
   correct answer. No method that reads only the model's behaviour can separate the
   two; we report how often this happens and suggest retrieval for those cases.
5. **Why use the model's own alternative, not the reference answer?** Using the
   reference would leak the label into the score and would be impossible in practice.
6. **Does the alternative have to be plausible?** The random alternative ablation
   tests exactly this.
7. **Why AUROC and not accuracy?** AUROC needs no threshold and summarises ranking
   quality; any threshold would have to be tuned on held out data.
8. **Why matched classes?** So that CW and CR have the same size and dataset mix; a
   dataset effect cannot pose as a detector effect.
9. **Why τ = 0.9 and not 1.0?** 1.0 is stricter but gives fewer items; we report 1.0 as
   a sensitivity check.
10. **How reliable is the judge?** Alias matching first, then GPT-4o-mini, checked
    against 100 human labels with Cohen's kappa.
11. **Is NLI reliable for short answers?** Not perfectly. The question is prepended to
    both sides, which is the standard remedy; errors affect all methods alike.
12. **Why DeLong and Holm?** DeLong is the standard paired test for AUROCs on the same
    items; Holm controls family wise error over seven comparisons without assuming
    independence.
13. **Is 8 generations per question cheap?** Cheaper than Semantic Entropy with 10
    samples (11) and MetaQA (13); more than log probability (1). The cost figure shows
    the trade.
14. **Could quantization change the results?** It can change answers. The precision is
    recorded and reported; comparisons are within one run, so all methods see the
    same model.
15. **Why two models?** Two families show whether results hold beyond one model and
    let each act as the other's verifier.
16. **Why not larger models?** Compute. Tan et al. found consistent errors persist with
    scale, so the question remains relevant; testing larger models is future work.
17. **Why these three challenge phrasings?** They span a personal claim, an external
    source and social proof. Huang (2026) shows wording matters; we report this as a
    limitation and release the prompts.
18. **What does a negative CPS mean?** The model gives up its answer more under doubt
    than under a named counterclaim; this suggests the named alternative is less
    plausible to the model than its own answer.
19. **Why not train a classifier?** A training free rule works for any model without
    labels. A logistic regression is reported as a reference ceiling.
20. **How do you avoid tuning on the test data?** No threshold or weight is fitted;
    the only fitted model uses out of fold predictions.
21. **Is the comparison with MetaQA fair?** Same items, same model, same verification
    setting; MetaQA uses 5 + 5 mutations, near the setting its authors found optimal.
22. **Why is cross family disagreement a baseline and also part of your method?** Alone
    it is the text only version of a prior idea; combining it with CPS is our
    contribution and H4 tests whether the combination adds anything.
23. **Could the model detect that it is being tested?** Possibly, but the same
    conversation format applies to right and wrong answers alike; any such effect
    adds noise rather than bias toward one class.
24. **What about long answers?** Out of scope; CPS needs a single answer to compare
    against. Extension to claims is future work.
25. **What is the practical use?** Run sampling first; apply CPS only when the samples
    agree. That spends the extra generations only where sampling has no signal.
26. **What if the pilot shows few CW items?** Increase TruthfulQA share or question
    count; the pilot exists to size the full run.
27. **How many CW items do you need?** About 100 for a ±0.07 interval and 150 for about
    ±0.06 on an AUROC near 0.7 (normal approximation).
28. **Why report "All" AUROC at all?** It shows baselines work in general and fail on
    the consistent split, which is the motivating contrast.
29. **Is the code reproducible by others?** Pinned revisions, saved ids, seeds, recorded
    environment, released raw outputs and one command to rebuild every number.
30. **What would make you conclude the method does not work?** Any of: CPS AUROC below
    0.65, not above the strongest text only baseline, or Holm adjusted p at or above 0.05.
31. **Does the doubt condition change the model's later answers?** Each challenge is a
    separate conversation, so replies do not influence each other.
32. **Why not use an LLM to judge flips instead of NLI?** NLI is cheaper, local and
    already used for the split; using one equivalence rule everywhere keeps the
    analysis consistent.

# Part H. Learning roadmap

| Topic | Essential or optional | Resources | Practice |
| --- | --- | --- | --- |
| LLM basics, decoding, sampling | essential | Jurafsky and Martin, *Speech and Language Processing* (3rd ed. draft, free online), chapters on transformers and LLMs; Hugging Face course (huggingface.co/learn) | run `generate.py` on 5 questions and compare T = 0 and T = 0.5 |
| Hallucination detection | essential | Manakul et al. 2023 (SelfCheckGPT); Farquhar et al. 2024 (Nature); Tan et al. 2025; Huang et al. 2025 survey (ACM TOIS) | reimplement Semantic Entropy for 3 questions by hand |
| NLI | essential | He et al. 2021 (DeBERTa); Kuhn et al. 2023 | test `microsoft/deberta-large-mnli` on 10 answer pairs |
| Sycophancy | essential | Sharma et al. 2024; Laban et al. 2023; Huang 2026 | ask a model "Are you sure?" after right and wrong answers |
| ROC and AUROC | essential | James et al., *An Introduction to Statistical Learning*, classification chapter; scikit-learn model evaluation guide | compute AUROC by hand for 6 items |
| Bootstrap | essential | Efron and Tibshirani, *An Introduction to the Bootstrap* (1993) | bootstrap the mean of 20 numbers |
| DeLong, Holm | essential | DeLong et al. 1988; Sun and Xu 2014; Holm 1979 | run `tests/test_metrics.py` and read the code |
| Inter-rater agreement | essential | Cohen 1960 | compute kappa for the label sheet |
| Transformers and vLLM | essential for running | Hugging Face Transformers docs; vLLM docs (docs.vllm.ai) | load Qwen2.5-0.5B and generate |
| Quantization | optional | bitsandbytes and AWQ documentation | compare answers in 4-bit and 16-bit |
| Hidden state probing | optional | Orgad et al. 2025; Tan et al. 2025 | train a logistic probe on hidden states |
| LaTeX and IEEE format | essential for writing | Overleaf learn pages; IEEE template guidance | change a table and recompile |
| Academic writing | essential | `docs/WRITING_CHECKLIST.md`; Strunk and White | rewrite one paragraph using the checklist |

---
title: "Literature Review: Detecting Consistent Hallucinations in Large Language Models"
date: "22 October 2026"
---

# 1. Scope and method

This review covers hallucination detection for factual question answering, with a
focus on the case where a model repeats the same wrong answer. It builds on the 20
papers the team reviewed (the "all" tab of the team sheet) and adds the closest work
found in a targeted search on 22 October 2026 (arXiv, ACL Anthology, Semantic Scholar,
publisher pages). Search terms: "self-consistent errors", "consistent hallucination",
"consistency is not correctness", "counterfactual probing hallucination", "sycophancy
flip rate", "black box hallucination detection". Bibliographic details were checked
against arXiv records and publisher pages; see `CITATION_VERIFICATION_REPORT.md`.

# 2. What has been studied

**Consistency based detection.** SelfCheckGPT (Manakul et al., EMNLP 2023) compares an
answer with sampled answers. Semantic Entropy (Kuhn et al., ICLR 2023; Farquhar et
al., Nature 2024) clusters samples by bidirectional entailment and measures entropy
over meanings. Later work changes how samples are compared: Kernel Language Entropy
(Nikitin et al., NeurIPS 2024), graph measures (Lin et al., TMLR 2024), a Bayesian
estimate that needs half the samples (team sheet, arXiv 2504.03579), paraphrased
inputs (Semantic Reformulation Entropy, arXiv 2509.17445), clustering of hidden
embeddings (Cleanse, arXiv 2507.14649), AMR graphs (arXiv 2501.02020), and token level
variance across passes (arXiv 2507.04137).

**Cross model and verifier checks.** SAC3 (Zhang et al., EMNLP 2023) adds question
perturbation and a second model. Xue et al. (TMLR 2026) show that self consistency
methods already approach the best performance their input allows and that a verifier
model raises the ceiling; their two stage method calls the verifier only when
uncertain. Goel et al. (arXiv 2508.14314) use agreement among frontier models to find
and edit errors without external knowledge.

**Fact level and structured checks.** ConFactCheck (IJCNLP Findings 2025) probes key
facts with generated questions; FactSelfCheck (EACL Findings 2026) checks knowledge
graph triples across samples; RU (IJCNN 2025) scores uncertainty per fact and adds
trap questions about invented people.

**Perturbation and metamorphic testing.** MetaQA (Yang et al., FSE 2025) asks the model
for synonym and antonym versions of its answer and checks them. Counterfactual probing
(Feng, arXiv 2508.01862) measures how the model's confidence moves between a statement
and altered versions. Attention guided self reflection (AGSER, EMNLP 2025) splits the
query by attention.

**Reasoning and trained detectors.** HalluClean (AAAI 2026) prompts a plan, reasoning
and judgment cycle and revises the answer. HAD (ACL 2026 Industry) fine tunes 7B to 14B
detectors on about 90,000 synthetic examples across an 11 type taxonomy.

**Internal signals.** Probes on hidden states predict correctness (Orgad et al., ICLR
2025; Zhou et al., AAAI 2026). Zhou et al. also show that misleading context produces
confident wrong answers.

**Consistent errors.** Tan et al. (EMNLP 2025) define self-consistent errors: every
sample agrees with a wrong greedy answer. On TriviaQA and SciQ, these errors stay level
or increase with model size while inconsistent errors fall, four detector families
struggle on them, and Semantic Entropy scores at or below chance. Their fix fuses
hidden state probes of the target and a verifier model. Ganesh et al. (EACL 2026)
split errors under prompt variation across 16 models and conclude that common
detectors "detect consistency, not correctness"; TruthfulQA errors are mostly
consistent.

**Sycophancy and answer stability.** Models trained with human feedback agree with
users against the truth (Sharma et al., ICLR 2024). The FlipFlop experiment (Laban et
al., arXiv 2311.08596) shows accuracy drops when a model is asked "Are you sure?".
Huang (arXiv 2610.06522) shows a flip rate mixes corrections with capitulations and
that rankings depend on the challenge wording.

# 3. Established approaches and strong baselines

| Family | Strong baseline | Access | Cost per question |
| --- | --- | --- | --- |
| Sampling consistency | Semantic Entropy; SelfCheckGPT-NLI | text | 1 + n samples (n about 10) |
| Model confidence | token log probability; P(True); verbalized confidence | logits or text | 1 to 2 |
| Perturbation | MetaQA | text | about 13 |
| Cross model | verifier agreement (Xue et al.; SAC3) | text | 2 or more |
| Trained | hidden state probes (Orgad et al.; Tan et al.) | weights and labels | 1 plus training |

# 4. Datasets and protocols in use

TriviaQA, SciQ, Natural Questions and SQuAD dominate short answer evaluation;
TruthfulQA targets misconceptions; HaluEval, FELM and FactCHD target detection
directly. AUROC is the standard metric, often with AURAC; correctness labels come
mostly from LLM judges (GPT-4 class) with small human checks.

# 5. Supported findings

1. Sampling consistency detects many hallucinations in general QA (Farquhar et al.).
2. It fails on consistent errors (Tan et al.; Ganesh et al.).
3. Consistent errors do not disappear with scale (Tan et al.).
4. A second model from another family helps (Tan et al., white box; Xue et al., black box in general QA).
5. Models change answers under challenge, and this does not always track correctness (Laban et al.; Huang).

# 6. Genuine gaps

1. **No black box, training free detector has been shown to work on consistent errors.** The only working method needs hidden states of two models and labels.
2. **Pressure based signals have not been evaluated as detectors of consistent errors.** Counterfactual probing and MetaQA perturb the answer but were not tested on this error type, and none controls for the general tendency to give in.
3. **Many detectors are evaluated only on overall AUROC**, which mixes easy inconsistent errors with hard consistent ones.

Gaps that are real but outside this project: multilingual evaluation, long form
generation, and human labelled fact level benchmarks (limitation groups G5, G6 and G8
in the team sheet).

# 7. How this study differs

| Aspect | Tan et al. 2025 | Ganesh et al. 2026 | MetaQA 2025 | Counterfactual probing 2025 | This study |
| --- | --- | --- | --- | --- | --- |
| Targets consistent errors | yes | yes (analysis) | no | no | yes |
| Needs weights or logits | yes | logits | no | confidence scores | no |
| Needs training | yes | no | no | no | no |
| Controls for sycophancy | no | no | no | no | yes (doubt condition) |
| Free form answers | yes | no (MCQ) | yes | yes | yes |
| Paired tests on identical items | partly | no | no | no | yes |

# 8. Comparison table of reviewed papers

| Paper | Venue | Method | Data | Main limitation relevant here |
| --- | --- | --- | --- | --- |
| Zero knowledge cross model consistency (Goel et al.) | arXiv 2025 | cross model sampling with prompt variants, block editing | FELM, GPQA-Diamond, NQ | consensus can reinforce shared biases; high API cost |
| Token level variance signals | arXiv 2025 | variance of token log probabilities over passes | SQuAD v2, TriviaQA, XSum | fails when wrong answers have near zero variance |
| Decoding Memories (DMP) | arXiv 2025 | prefix KV reuse to speed self consistency | TriviaQA, NQ-Open, SQuAD, HaluEval | memory overhead; short answers |
| Bayesian Semantic Entropy | arXiv 2025 | Dirichlet prior over meaning classes | TriviaQA, NQ, SQuAD, SVAMP | assumes divergent errors; NLI clustering bottleneck |
| Semantic graph uncertainty | arXiv 2025 | AMR graph propagation of token uncertainty | WikiBio, NoteSum | parser errors propagate; white box |
| Small Updates, Big Doubts | arXiv 2026 | audit of fine tuning effects on detectors | TriviaQA, NQ-Open, SQuAD | consistency rises without better calibration |
| Semantic Reformulation Entropy | arXiv 2025 | paraphrased inputs plus hybrid clustering | SQuAD v2, TriviaQA | 24 generations; paraphrase drift |
| Cleanse | arXiv 2025 | clustering of hidden embeddings | SQuAD, CoQA | white box; consistent clusters misread |
| SelfCheck-Eval | arXiv 2025 | modular zero resource detectors | WikiBio, AIME | fails on correct math reasoning |
| Paraphrase consistency as evidence | IJWR | consistency across paraphrased questions | English and Persian QA | consistently wrong answers pass |
| AGSER | EMNLP 2025 | attention guided query splitting | Books, Movies, GCI | needs attention maps |
| ConFactCheck | IJCNLP Findings 2025 | key fact probing questions | NQ-Open, HotpotQA, WebQA, WikiBio | needs logits for one check; English tools |
| FactSelfCheck | EACL Findings 2026 | knowledge graph triples across samples | WikiBio, FavaMultiSamples | repeated false facts look reliable |
| MetaQA | FSE 2025 | synonym and antonym mutations, self verification | TruthfulQA, HotpotQA, FreshQA | mutation errors; one baseline compared |
| RU | IJCNN 2025 | fact level uncertainty with trap questions | MulFactTrap (77 questions) | white box; one trap scenario |
| Can LLMs detect their confabulations? | AAAI 2026 | evidential uncertainty plus hidden state probes | HotpotQA, NQ, TruthfulQA, TriviaQA, Math | white box; misleading context gives confident errors |
| Verify when Uncertain | TMLR 2026 | self plus cross model consistency, two stage | SQuAD, TriviaQA, NQ | needs labelled validation set; not tested on consistent errors |
| HalluClean | AAAI 2026 | plan, reason, judge, revise prompting | HaluEval, UMWP, ChatProtect, HaluBench | depends on backbone knowledge |
| HAD | ACL 2026 Industry | fine tuned taxonomy based detectors | HADTest, HaluEval, FactCHD, FaithBench | single span; synthetic training data |
| Token uncertainty + NLI + SE | IEEE conference 2025 | weighted fusion of three signals | SQuAD 2.0 subset | full text paywalled; one dataset |
| Too Consistent to Detect (Tan et al.) | EMNLP 2025 | study of self-consistent errors; cross model hidden state probe | TriviaQA, SciQ | white box and supervised |
| Rethinking Hallucinations (Ganesh et al.) | EACL 2026 | prompt multiplicity analysis | 8 benchmarks, 16 models | MCQ only; no new detector |
| Counterfactual Probing (Feng) | arXiv 2025 | confidence sensitivity to counterfactual statements | TruthfulQA (100), 200 statements | not tested on consistent errors |
| Anatomy of LLM Sycophancy (Huang) | arXiv 2026 | control condition for flip rates | moral and arithmetic banks | not about hallucination |

The full per paper extraction (research question, gap, mechanism, granularity,
results, limitations, future work) is in the team sheet, "all" tab, and the grouped
limitations are in the "Limitation Groups" and "Limitation Matrix" tabs.

# 9. Update after the pilot (27 October 2026)

The pilot showed that consistent errors give way more often than consistent correct
answers under plain doubt, and that subtracting the doubt flip rate (CPS) cancels this
difference. The main score became the doubt flip rate. A second search (23 more papers,
listed in `NOVELTY_CHECK_OCT2026.md`) found that the link between changing an answer
under challenge and being wrong is already known: Pawitan and Holmes (Harvard Data
Science Review 2025) show that kept answers are more accurate, and QueRE (Sam, Finzi
and Kolter, NeurIPS 2025) uses "Are you sure?" among about 50 follow-up questions as
features of a supervised correctness predictor. Studies of sycophancy (SycEval; Kim and
Khashabi 2025; Guo et al. 2026) report that models switch less when originally right.
Studies of high certainty errors (CHOKE, Simhi et al. 2025; Delusions, Xu et al. 2025)
confirm that many errors come with high confidence, and Delusions adds that the most
entrenched errors resist reflection.

**Remaining gap:** no study evaluates a training free doubt score as a detector of
consistent errors, where sampling fails, against sampling and perturbation baselines on
identical items, or tests whether controlling for general yielding helps. That is the
contribution of this project, stated at that scope in the paper.

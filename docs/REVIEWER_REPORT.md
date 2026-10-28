# Independent Review of the Package (protocol version 2, before the full run)

Written as a reviewer would read the placeholder paper and the code, 27 October 2026.
It judges the design, not results; the full run has not happened yet.

## Summary of the submission

The paper evaluates the doubt flip rate (DFR), the share of replies to "Are you sure?"
style prompts that abandon an answer, as a text only and training free detector of
consistent errors. It also tests a contrastive variant (CPS) and a combination with
cross family disagreement. Both are compared with seven baselines on a matched set of
consistently wrong and consistently right answers from two 7B to 8B models on three
datasets.

## Strengths

1. The target is narrow and well motivated: Tan et al. (2025), Ganesh et al. (2026)
   and the high certainty hallucination studies show the failure, and no black box,
   training free remedy has been evaluated on it.
2. The method is very cheap (4 generations) and easy to deploy after sampling.
3. Evaluation is paired, with DeLong tests, Holm correction and bootstrap intervals.
4. The change from CPS to DFR is handled openly: pilot items are excluded, criteria
   were fixed before the full run, and CPS is reported as a tested hypothesis (H3).
5. Every number in the paper comes from one file; revisions and seeds are pinned.

## Weaknesses a reviewer will raise

| # | Point | Severity | What to do |
| --- | --- | --- | --- |
| 1 | Novelty: "answers that change under challenge are less reliable" is known (Pawitan and Holmes 2025; QueRE uses "Are you sure?" as a supervised feature). | major | The paper cites both and narrows the claim: training free, consistent errors, paired comparison with sampling detectors, and the CPS result. Keep the claim that narrow in the review boxes. |
| 2 | The main score was chosen after a pilot. | moderate | Already addressed in Limitations and the protocol change log; do not add any other change after seeing full results. |
| 3 | Only two small open models. | major | State clearly; an API model on the matched set would be the best addition if compute allows. |
| 4 | Delusions (Xu et al. 2025): the most entrenched errors resist reflection, so DFR may miss them. | moderate | Discussed under "What DFR cannot see"; the examples table shows the CW item with the lowest DFR. |
| 5 | Llama flips under counterclaims almost always (pilot: about 90 percent for both classes). | minor | Explains why F_C is weak for Llama; mention when interpreting H3. |
| 6 | Correctness labels from a judge; NLI on short answers. | moderate | Cohen's kappa on 100 human labels; check 50 flip decisions by hand. |
| 7 | Three doubt phrasings; wording effects (Huang 2026). | minor | Per prompt replies are saved; the k ablation shows sensitivity. |
| 8 | Main track deadline (21 August 2026) has passed; S25 rules unconfirmed. | blocking for submission | Confirm the S25 deadline, page limit and blind policy. |
| 9 | Some bibliography entries are arXiv only or marked Check. | moderate | Resolve via `CITATION_VERIFICATION_REPORT.md` before submission. |

## Code review notes

- Every stage is resumable and refuses to run on unfinished input; GPU stages run one
  process per model and recover from out of memory errors by splitting batches.
- `prepare` refuses to reuse a results folder drawn with different question settings.
- The pipeline and the report were run end to end on real small models and real data
  on CPU (27 October 2026), including the case of a model with no consistent errors.
- No attribution strings, keys or personal paths in the repository (tested).

## Verdict on readiness

The design is sound for a workshop paper with a modest, honest claim. It is **not
ready to submit** until the full run, the human labels, author fields, the three
review paragraphs and the AI use statement are done and the venue rules confirmed.

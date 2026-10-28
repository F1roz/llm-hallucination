---
title: "Complete Research Standard Operating Procedure"
subtitle: "Agreement Is Not Truth: from topic to submission"
date: "22 October 2026"
---

# How to read this SOP

Each phase gives the objective, prerequisites, actions, commands, outputs, quality
checks, common errors, troubleshooting and the completion criterion. Status as of
22 October 2026: phases 1 to 11 and 16 to 17 are done in the repository (11 on the
simulated model); phases 12 to 15 and 18 to 23 need GPU runs or author judgment.

# Phase 1. Topic validation

- **Objective:** confirm the topic is relevant, feasible and new enough.
- **Prerequisites:** team literature sheet (20 papers).
- **Actions:** group limitations; score impact, feasibility, novelty; choose G1 (consistent errors).
- **Outputs:** "Limitation Groups" and "Experiment Options" tabs; proposal document.
- **Quality checks:** at least 10 papers share the gap; the experiment fits free or low cost GPUs.
- **Common errors:** choosing the most cited gap without checking it is already solved.
- **Troubleshooting:** run a targeted search for the exact gap before committing.
- **Completion:** gap and option recorded; novelty search done (found Tan et al. 2025 and narrowed to the black box, training free angle).

# Phase 2. Literature review

- **Objective:** know what exists, what works and what is missing.
- **Actions:** read the 20 papers; search arXiv, ACL Anthology, Semantic Scholar; extract method, data, results, limits.
- **Outputs:** `docs/LITERATURE_REVIEW.docx`; team sheet.
- **Quality checks:** every claim in the paper's related work traces to a read source.
- **Common errors:** citing from abstracts only; trusting secondary summaries.
- **Completion:** comparison table and gap statement written.

# Phase 3. Research gap identification

- **Objective:** a gap that is real, not speculative.
- **Actions:** compare the closest works on access, training, target error type, sycophancy control.
- **Outputs:** gap section of the review; Table 7 of the review.
- **Completion:** gap stated in one sentence and defended against the three closest papers.

# Phase 4. Research question formulation

- **Objective:** testable questions with fixed success criteria.
- **Outputs:** H1 to H4 in `docs/EXPERIMENTAL_PROTOCOL.md`.
- **Quality checks:** each hypothesis has a numeric criterion and a failure condition.
- **Completion:** criteria fixed before the pilot.

# Phase 5. Dataset selection

- **Objective:** data that produce consistent errors and match prior work.
- **Actions:** choose TruthfulQA, TriviaQA, SciQ; pin revisions; check licenses.
- **Outputs:** `docs/DATASET_DOCUMENTATION.md`; `data.py`.
- **Common errors:** unpinned dataset versions; license misuse (SciQ is non commercial).
- **Completion:** revisions pinned, licenses recorded.

# Phase 6. Methodology design

- **Objective:** a design where the comparison is fair and the conclusion follows from the data.
- **Actions:** paired design; matched classes; same samples for all methods; doubt control; ablations.
- **Outputs:** protocol sections 4 to 7; paper Method section.
- **Completion:** every design choice has a reason written next to it.

# Phase 7. Experimental protocol finalisation

- **Objective:** freeze settings before results exist.
- **Actions:** finalise configs; record the change log.
- **Commands:** `git add configs docs/EXPERIMENTAL_PROTOCOL.md && git commit -m "Freeze protocol"`.
- **Completion:** protocol committed.

# Phase 8. Codebase development

- **Objective:** a reproducible pipeline from questions to paper.
- **Outputs:** `src/pressure_probe/`, configs, notebooks.
- **Quality checks:** one responsibility per file; all prompts in one module; no hardcoded paths or keys.
- **Completion:** every stage runs on the simulated backend.

# Phase 9. Unit testing

- **Commands:** `python -m pytest -q tests/test_metrics.py tests/test_scoring.py tests/test_text_split_config.py`.
- **Quality checks:** metric code compared with scikit-learn and with bootstrap; Holm against known values.
- **Completion:** all unit tests pass (see `docs/TEST_REPORT.md`).

# Phase 10. Integration testing

- **Commands:** `python -m pytest -q tests/test_pipeline_mock.py tests/test_report.py`.
- **Quality checks:** resume after interruption; idempotent reruns; every result token resolves; PDF within 10 pages; Word file has yellow highlights.
- **Completion:** all pass.

# Phase 11. Smoke testing (Colab, Mode A)

- **Objective:** prove every stage works on a real GPU.
- **Actions:** run `notebooks/smoke_test.ipynb`.
- **Outputs:** `results/smoke/` and a throwaway PDF.
- **Common errors:** CPU runtime; missing `HF_TOKEN`; vLLM not supported (falls back automatically).
- **Completion:** all cells finish; validation report shows a page count.

# Phase 12. Pilot experiment (Mode B)

- **Objective:** check the CW count, the pressure signal and runtime.
- **Actions:** run `notebooks/pilot_experiment.ipynb`.
- **Quality checks:** CW at least 15 per model; a pressure score above 0.6 for one model; stage times in `timings`.
- **Outcome (27 October 2026):** CW 62 and 33; CPS failed (0.53, 0.40) while the doubt flip rate reached 0.69 and 0.65, so protocol version 2 made DFR the main score before the full run.
- **Troubleshooting:** low CW: raise TruthfulQA share; no signal: stop and review prompts before spending on the full run.
- **Completion:** go or no go decision written in the protocol change log.

# Phase 13. Full experiments (Mode C)

- **Actions:** size `configs/full.yaml` from the pilot; run `notebooks/full_experiment.ipynb` stage by stage.
- **Commands:** `python -m pressure_probe.cli run --config configs/full.yaml`.
- **Common errors:** session limits (rerun the cell; work resumes); out of memory (lower `batch_size`).
- **Completion:** `results/full/summary.json` exists with `mode: full`.

# Phase 14. Ablation studies

- **Objective:** isolate the contribution of the number of doubt prompts, the counterclaim, the subtraction (CPS) and the combination with X.
- **Actions:** none extra; ablations are computed in `evaluate` from stored replies (random alternative is collected in `probe`).
- **Completion:** ablation table filled in the paper.

# Phase 15. Statistical analysis

- **Actions:** `python -m pressure_probe.cli stage evaluate --config configs/full.yaml`.
- **Quality checks:** intervals present for every AUROC; Holm adjusted p values; hypothesis verdicts.
- **Common errors:** reading overlapping intervals as "no difference"; use the paired test.
- **Completion:** `summary.json` complete.

# Phase 16. Figure and table generation

- **Commands:** part of `report`.
- **Quality checks:** fonts at least 6 pt in print; colours pass the palette check; every mark labelled.
- **Completion:** figures in `paper/generated/`.

# Phase 17. Automated manuscript updating

- **Commands:** `python -m pressure_probe.cli report --config configs/full.yaml`.
- **Outputs:** `paper/main.tex`, `paper/paper.pdf`, `paper/paper.docx`, `paper/validation_report.json`, `dist/full_package.zip`.
- **Quality checks:** `unresolved_tokens` empty; `todo_marks` equals the author review boxes and author fields left.
- **Completion:** only author written parts remain yellow.

# Phase 18. Scientific interpretation

- **Objective:** explain the results without overclaiming.
- **Actions:** replace the three review boxes in `paper/main.template.tex`; use examples from `scores.csv` and the examples table.
- **Quality checks:** every causal statement is marked as interpretation; no new numbers typed by hand.
- **Completion:** all authors agree with the discussion.

# Phase 19. Citation verification

- **Actions:** resolve the six **Check** rows in `docs/CITATION_VERIFICATION_REPORT.md`; rerun the related work search.
- **Completion:** no unverified entry remains.

# Phase 20. Manuscript editing

- **Commands:** `python tools/check_writing.py paper/main.template.tex`.
- **Quality checks:** zero listed terms and dashes; hyphens only in names and identifiers; sentences under 25 words where possible.
- **Completion:** checker passes and two authors have read the paper aloud once.

# Phase 21. IEEE formatting validation

- **Quality checks:** at most 10 pages including references; IEEEtran conference class; no Type 3 fonts; figures readable; PDF eXpress if the portal asks.
- **Completion:** `ready_to_submit: true` in the validation report.

# Phase 22. Independent peer review

- **Actions:** a colleague outside the project reads the PDF with `docs/REVIEWER_REPORT.md` as a checklist; fix what they find.
- **Completion:** all serious points resolved or acknowledged in Limitations.

# Phase 23. Final submission preparation

- **Actions:** follow `docs/SUBMISSION_CHECKLIST.md`; confirm the S25 track deadline; all authors approve.
- **Completion:** submitted by hand through CyberChair.

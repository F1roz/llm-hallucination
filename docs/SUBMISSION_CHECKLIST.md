# Submission Checklist (IEEE BigData 2026)

Venue facts checked on 2026-10-22 at https://bigdataieee.org/BigData2026/calls/papers/:

- Full papers: up to 10 pages in IEEE two column format, **references count toward the 10 pages**, no appendix.
- Template: IEEE Computer Society proceedings (IEEEtran, conference option). `paper/IEEEtran.cls` is version 1.8b.
- Review: **single blind**, so author names and affiliations stay in the paper.
- Main track deadline was 21 August 2026; notification 24 October 2026; conference 14 to 17 December 2026, Phoenix.
- The submission link you were given (`subarea=S25`, `ws_submit.php`) is a workshop or special session portal. Its page could not be opened from here (the site blocks automated access). **Open it yourself and confirm the S25 deadline, page limit and topic fit before planning the run.**
- IEEE policy on AI generated content (https://open.ieee.org/author-guidelines-for-artificial-intelligence-ai-generated-text/): use of AI generated text, figures, images or code must be disclosed in the acknowledgments, naming the system and the sections concerned. Editing help is treated more leniently, but disclosure is recommended.

## Before running the full experiment

- [ ] S25 deadline and rules confirmed on the CyberChair page
- [ ] Smoke test passed on Colab
- [x] Pilot run on 27 October 2026; protocol version 2 fixed before the full run (change log in EXPERIMENTAL_PROTOCOL.md)
- [ ] Full config sized from the pilot (about 150 CW per model)

## After the full experiment

- [ ] `results/full/summary.json` exists and `mode` is `full`
- [ ] `human_labels.csv` filled by two authors; kappa printed in the paper
- [ ] `paper/authors.yaml` filled
- [ ] Three author review boxes replaced with your own text in `paper/main.template.tex`
- [ ] Acknowledgment includes funding and the AI use statement
- [ ] Report rerun; `paper/validation_report.json` shows `ready_to_submit: true`
- [ ] Page count at most 10 including references
- [ ] No `[todo]` marks, no yellow boxes in the PDF
- [ ] Abstract, results, discussion and conclusion match `summary.json` (spot check five numbers by hand)
- [ ] Hypothesis verdicts in the text match `summary.json` `hypotheses`
- [ ] `python tools/check_writing.py paper/main.template.tex` reports zero listed terms and dashes
- [ ] Six citations marked **Check** in `docs/CITATION_VERIFICATION_REPORT.md` confirmed
- [ ] Fresh related work search done (papers after 2026-10-22)
- [ ] Every figure readable when printed at 100%
- [ ] PDF fonts embedded, no Type 3 fonts (`type3_fonts: []` in the validation report)
- [ ] If required by the portal, PDF passes IEEE PDF eXpress
- [ ] All authors approved the final PDF
- [ ] Code and data release link prepared (GitHub repository, results archive)

## Submission

Submission is manual through CyberChair. Nothing in this project submits on its own.

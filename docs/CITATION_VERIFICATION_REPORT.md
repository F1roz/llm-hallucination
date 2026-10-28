# Citation Verification Report

Checked on 2026-10-22. Sources: the arXiv API (titles, full author lists, comments,
journal references, DOIs), the Semantic Scholar API (venue, year, pages, DOI), the
papers' own PDFs where opened, and publisher pages. Every entry in
`paper/references.bib` is listed. Status:

- **Verified**: title, authors and venue confirmed from a record listed in the row.
- **Verified, venue from author note**: venue confirmed only by the authors' own arXiv
  comment or PDF header.
- **Check**: one field could not be confirmed online; the row says which.

| Key | Status | Evidence | Claim it supports in the paper |
| --- | --- | --- | --- |
| tan2025consistent | Verified | arXiv 2505.17656 (11 authors), arXiv comment "EMNLP 2025 Main", Semantic Scholar venue EMNLP 2025; full text read | definition of self-consistent errors, scaling trend, SE near chance, cross model probe needs hidden states and labels |
| ganesh2026rethinking | Verified | arXiv 2602.00723, comment "To appear at EACL 2026", Semantic Scholar venue EACL 2026; full text read | detectors track consistency, consistent errors dominate TruthfulQA |
| feng2025counterfactual | Verified (preprint) | arXiv 2508.01862; full text read | counterfactual probing method, not evaluated on consistent errors |
| huang2026sycophancy | Verified (preprint) | arXiv 2610.06522; full text read | flip rate hides corrections versus capitulations; wording dependence |
| xue2026verify | Verified, venue from author note | arXiv 2502.15845 PDF header "Published in Transactions on Machine Learning Research (06/2026)"; full text read | self consistency near its ceiling; verifier raises it |
| yang2025metaqa | Verified | PDF: Proc. ACM Softw. Eng. 2(FSE), Article FSE020, doi 10.1145/3715735; full text read | MetaQA method and the "10% of the brain" example with SelfCheckGPT score 0.2 |
| zhou2026confabulations | Verified | arXiv 2508.08139 comment "Published at AAAI'26"; Semantic Scholar pages 38164 to 38172; full text read | misleading context gives confident wrong answers |
| joo2025uncertain | Verified (preprint) | arXiv 2509.21999 abstract page | consistency under uncertain expression |
| kuhn2023semantic | Verified | arXiv comment "ICLR 2023 (Spotlight)"; Semantic Scholar venue ICLR | bidirectional entailment equivalence, Semantic Entropy |
| farquhar2024detecting | Verified | Semantic Scholar: Nature 630, 625 to 630, doi 10.1038/s41586-024-07421-0 | Semantic Entropy, AURAC |
| manakul2023selfcheckgpt | Verified | arXiv comment "EMNLP 2023 (main conference)"; pages 9004 to 9017 and doi 10.18653/v1/2023.emnlp-main.557 from the reference list of a verified paper | SelfCheckGPT |
| kadavath2022language | Verified (preprint) | arXiv 2207.05221 author list | P(True) |
| xiong2024express | Verified, venue from author note | arXiv comment "accepted by ICLR 2024" | verbalized confidence |
| lin2024generating | Verified, venue from author note | arXiv comment "Published in TMLR (05/2024)" | graph based consistency measures |
| nikitin2024kernel | Verified | Semantic Scholar venue NeurIPS 2024 | Kernel Language Entropy |
| zhang2023sac3 | **Check** | arXiv comment "EMNLP 2023"; Semantic Scholar pages 15445 to 15458. Main conference versus Findings not confirmed; the bib says Findings. Confirm on the ACL Anthology before submission | SAC3 |
| goel2025zero | Verified (preprint) | arXiv 2508.14314 | cross model agreement without knowledge |
| tong2025sre | Verified (preprint) | arXiv 2509.17445; comment says submitted to ICASSP 2026 | Semantic Reformulation Entropy |
| orgad2025llms | Verified | arXiv 2410.02707 (7 authors); Semantic Scholar venue ICLR. Year printed as 2025 (conference year); Semantic Scholar lists 2024 (preprint year). Confirm on OpenReview | hidden states encode truthfulness |
| sharma2023sycophancy | Verified | arXiv 2310.13548; Semantic Scholar venue ICLR; conference year 2024 | sycophancy after human feedback |
| laban2023flipflop | Verified (preprint) | arXiv 2311.08596 | FlipFlop experiment |
| lin2022truthfulqa | Verified | Semantic Scholar: ACL 2022, pages 3214 to 3252, doi 10.18653/v1/2022.acl-long.229 | dataset |
| joshi2017triviaqa | Verified | Semantic Scholar: ACL 2017, doi 10.18653/v1/P17-1147 | dataset |
| welbl2017crowdsourcing | Verified | arXiv comment W-NUT 2017; doi 10.18653/v1/W17-4413 | dataset |
| qwen2024qwen25 | Verified (preprint) | arXiv 2412.15115 | model |
| grattafiori2024llama | Verified (preprint) | arXiv 2407.21783 | model |
| he2021deberta | **Check** | arXiv 2006.03654 authors confirmed. The ICLR 2021 venue is from prior knowledge; the Semantic Scholar lookup was rate limited. Confirm on OpenReview | NLI model |
| kwon2023efficient | Verified | Semantic Scholar: SOSP 2023, doi 10.1145/3600006.3613165; 9 authors from arXiv | vLLM |
| delong1988comparing | **Check** | standard reference; volume, pages and doi from prior knowledge, not looked up in this session | DeLong test |
| sun2014fast | **Check** | volume, pages and doi from prior knowledge, not looked up in this session | fast DeLong algorithm |
| holm1979simple | **Check** | standard reference; volume and pages from prior knowledge | Holm correction |
| cohen1960coefficient | **Check** | standard reference; volume, pages and doi from prior knowledge | Cohen's kappa |
| sam2025quere | Verified | NeurIPS 2025 poster page and ML Anthology entry (27 Oct 2026) | closest prior work: "Are you sure?" follow-ups as supervised features |
| pawitan2025confidence | Check | arXiv 2412.15296 says "to appear in Harvard Data Science Review"; volume and issue not confirmed | kept answers are more accurate than changed ones |
| xu2025delusions | Verified (arXiv) | arXiv 2503.06709 page (27 Oct 2026); no venue listed | entrenched errors resist reflection |
| simhi2025trust | Verified (arXiv) | arXiv 2502.12964 (27 Oct 2026); check for a later venue | high certainty hallucinations are 10 to 40 percent of hallucinations |
| guo2026conformity | Verified (arXiv) | arXiv 2605.27288 page (27 Oct 2026) | conformity rises with epistemic uncertainty |
| kim2025challenging | Verified (arXiv) | arXiv 2509.16533 page (27 Oct 2026) | models switch less when originally correct |
| fanous2025syceval | Verified (arXiv) | arXiv 2502.08177 page (27 Oct 2026) | more corrections than reversals under rebuttal |

## Actions before submission

1. Confirm the **Check** rows (seven, including pawitan2025confidence added on 27 October) on the publisher or proceedings page and correct the bib if needed.
2. Rerun a related work search (arXiv, ACL Anthology, Google Scholar) for "self-consistent errors", "consistent hallucination", "pressure", "sycophancy" and "hallucination detection" to catch papers published after 2026-10-22.
3. Compile and check that every `\cite` resolves (`paper/validation_report.json`, `undefined_citations`).

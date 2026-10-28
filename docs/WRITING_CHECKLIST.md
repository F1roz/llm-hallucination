# Writing Quality Checklist

This checklist governs every sentence of the manuscript and the project documents.
It combines two public guides on overused machine-style wording with the house rules
for this paper. The full term list is machine-readable in `tools/banned_terms.yaml`,
and `tools/check_writing.py` scans the LaTeX prose against it.

Sources:

- Grammarly, "Common AI Words": https://www.grammarly.com/blog/ai/common-ai-words/
- HumanizeAI, "Words and Phrases That Trigger AI Detection":
  https://thehumanizeai.pro/articles/words-phrases-that-trigger-ai-detection
  (its frequency multipliers are uncited; we use only the word lists)

AI detection scores are not a measure of writing quality or of authorship, and we do
not use them as one.

## 1. Words to replace

| Avoid | Write instead |
| --- | --- |
| delve, delve into | explore, examine |
| tapestry, landscape, realm | field, area, set |
| multifaceted, nuanced, intricate | complex, varied, subtle |
| comprehensive | full, complete |
| pivotal, crucial, paramount | important, key, necessary |
| leverage, utilize, harness | use |
| facilitate | help, enable |
| streamline | simplify |
| robust, robustness | reliable, stable (or define the technical sense once) |
| underscore, showcase, illuminate | show, explain |
| bolster | support |
| revolutionize, transformative, groundbreaking, game-changing, cutting-edge, innovative, seamless | cut, or state the measured change |
| foster, empower | encourage, help |

## 2. Transitions and fillers to cut

- Furthermore, Moreover, Additionally → "also", "and", or start a new sentence.
- Consequently, Nevertheless, That being said, With that in mind, In light of this → "so", "but", "still".
- In conclusion, To summarize → conclude without announcing it.
- It is worth noting, It is important to note, It goes without saying → state the fact.
- Generally speaking, Broadly speaking, To some extent, Arguably → give the specific claim or drop the hedge.
- When it comes to, In the realm of → "for", "in".
- Shed light on, A testament to, Navigate the complexities, Unlock the potential, Harness the power of → plain verbs.

## 3. Sentence and structure rules

1. Lead each paragraph with its point. No warm-up sentences.
2. Vary sentence length. A short sentence after two long ones is fine.
3. One claim per sentence where possible. Cut stacked qualifiers.
4. Every claim is specific: a number, a dataset, a citation, or a defined term.
5. No adjective without evidence ("strong", "large", "significant" need a number or a test).
6. "Significant" is reserved for statistical significance with a reported test.
7. Do not announce structure ("In this section we will..."). Exception: one short roadmap sentence at the end of the Introduction.
8. Prefer active voice when the actor matters ("we sample", "the model answers").
9. Avoid hyphens and dashes in prose. Rephrase compound modifiers ("a detector that needs no training" rather than "a training-free detector"). Keep hyphens only in official names, model identifiers, citations, equations and code.
10. No em dashes or en dashes in prose.
11. No rhetorical questions in the abstract or conclusion.
12. Do not invent personal experience, false uncertainty, or deliberate errors.
13. Results that have not been measured stay as yellow `[todo]` placeholders.

## 4. Before submission

- [ ] `python tools/check_writing.py paper/main.tex` reports zero listed terms.
- [ ] No em or en dashes in prose.
- [ ] Hyphens in prose appear only in names, identifiers and citations (checker lists them for review).
- [ ] Every number in the text traces to `results/<run>/summary.json`.
- [ ] Abstract, results and conclusion describe only measured findings.
- [ ] The acknowledgments state AI assistance as IEEE requires.

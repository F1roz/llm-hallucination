# Dataset Documentation

All three datasets are loaded from Hugging Face at a pinned commit. Facts below were
checked on 2026-10-22 against the dataset cards and the Hugging Face API; anything not
confirmed there is marked as such.

## TruthfulQA

| Field | Value |
| --- | --- |
| Source | `truthfulqa/truthful_qa`, config `generation`, split `validation` |
| Revision | `741b8276f2d1982aa3d5b832d3ee81ed3b896490` |
| Publication | Lin, Hilton, Evans. TruthfulQA: Measuring How Models Mimic Human Falsehoods. ACL 2022. doi:10.18653/v1/2022.acl-long.229 |
| License | Apache 2.0 |
| Size | 817 questions, 38 categories |
| Fields used | `question`, `best_answer`, `correct_answers` (references), `incorrect_answers` (given to the judge as negative references) |
| Missing values | none observed on the card |
| Labels | none in the sense of a class; correctness is judged per answer |
| Known limitations | built to elicit misconceptions, so the error profile differs from everyday use; many references are full sentences, which is why an LLM judge is used |
| Potential biases | English only; written by the dataset authors; cultural coverage not assessed here |
| Use here | pilot: 100 questions; full run: the 717 questions not used by the pilot; main source of consistent errors |

## TriviaQA

| Field | Value |
| --- | --- |
| Source | `mandarjoshi/trivia_qa`, config `rc.nocontext`, split `validation` |
| Revision | `0f7faf33a3908546c6fd5b73a660e0f8ff173c2f` |
| Publication | Joshi, Choi, Weld, Zettlemoyer. TriviaQA. ACL 2017. doi:10.18653/v1/P17-1147 |
| License | listed as "unknown" on the Hugging Face card; the card states the University of Washington does not own the copyright of questions and documents. **Check terms before redistributing questions.** |
| Size | validation split size is reported inconsistently (18,669 on the card, about 17.9k in the viewer); not needed because we sample with a seed |
| Fields used | `question`, `answer.value`, `answer.aliases` |
| Labels | aliases make string matching reliable for most answers |
| Known limitations | trivia style; some aliases are noisy |
| Use here | pilot: 50 questions; full run: 200 further questions sampled with the run seed after removing the pilot's 50; ids saved |

## SciQ

| Field | Value |
| --- | --- |
| Source | `allenai/sciq`, split `test` |
| Revision | `2c94ad3e1aafab77146f384e23536f97a4849815` |
| Publication | Welbl, Liu, Gardner. Crowdsourcing Multiple Choice Science Questions. W-NUT 2017. doi:10.18653/v1/W17-4413 |
| License | CC BY-NC 3.0 (non commercial) |
| Size | train 11,679, validation 1,000, test 1,000 |
| Fields used | `question`, `correct_answer` (reference); distractors given to the judge as negative references |
| Known limitations | written as multiple choice; asked here as an open question, so some questions accept several correct answers |
| Use here | pilot: 50 questions; full run: 200 further test questions sampled after removing the pilot's 50 |

## Data management

- `results/<run>/questions.jsonl` stores the exact questions, references and ids used.
- `results/<run>/config_used.json` and `environment.txt` record settings and package versions.
- Validation in `data.validate_questions`: unique ids, no empty question, at least one reference.
- No item is used for tuning; there is no training data.
- Synthetic items in `tests/fixtures/questions.jsonl` are for tests only and are labelled as such.

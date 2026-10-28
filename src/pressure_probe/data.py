"""Question loading from Hugging Face datasets, seeded subsampling and validation."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .config import Config
from .io_utils import get_logger, read_jsonl, write_jsonl

log = get_logger(__name__)

# Pinned dataset revisions (Hugging Face commit hashes checked on 2026-10-22).
DATASET_SOURCES = {
    "truthfulqa": {"path": "truthfulqa/truthful_qa", "name": "generation", "split": "validation",
                   "revision": "741b8276f2d1982aa3d5b832d3ee81ed3b896490"},
    "triviaqa": {"path": "mandarjoshi/trivia_qa", "name": "rc.nocontext", "split": "validation",
                 "revision": "0f7faf33a3908546c6fd5b73a660e0f8ff173c2f"},
    "sciq": {"path": "allenai/sciq", "name": None, "split": "test",
             "revision": "2c94ad3e1aafab77146f384e23536f97a4849815"},
}


@dataclass
class Question:
    qid: str
    dataset: str
    question: str
    gold_answers: list[str]
    incorrect_answers: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def questions_path(config: Config) -> Path:
    return config.run_dir / "questions.jsonl"


def load_questions(config: Config) -> list[Question]:
    return [Question(**row) for row in read_jsonl(questions_path(config))]


def prepare_questions(config: Config) -> list[Question]:
    """Load, subsample with the run seed, validate and save the question set."""
    target = questions_path(config)
    if target.exists():
        log.info("Questions already prepared at %s", target)
        return load_questions(config)
    if config.fixture_path:
        known = set(Question.__dataclass_fields__)
        pool = [Question(**{k: v for k, v in row.items() if k in known})
                for row in read_jsonl(Path(config.fixture_path))]
        questions = _subsample_fixture(pool, config)
    else:
        questions = []
        for name, count in config.datasets.items():
            items = _without_earlier_sample(_load_hf(name), config, name)
            questions.extend(_sample(items, count, config.seed, name))
    validate_questions(questions)
    write_jsonl(target, (q.to_dict() for q in questions))
    log.info("Saved %d questions to %s", len(questions), target)
    return questions


def validate_questions(questions: list[Question]) -> None:
    ids = [q.qid for q in questions]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate question ids")
    texts = [q.question.strip().lower() for q in questions]
    if len(texts) != len(set(texts)):
        log.warning("Some question texts repeat across items; they are kept as separate items")
    for q in questions:
        if not q.question.strip():
            raise ValueError(f"empty question {q.qid}")
        if not any(g.strip() for g in q.gold_answers):
            raise ValueError(f"no gold answer for {q.qid}")


def _sample(items: list[Question], count: int | None, seed: int, name: str) -> list[Question]:
    if count is None or count >= len(items):
        return items
    rng = random.Random(f"{seed}-{name}")
    chosen = sorted(rng.sample(range(len(items)), count))
    return [items[i] for i in chosen]


def _without_earlier_sample(items: list[Question], config: Config, name: str) -> list[Question]:
    """Drop the questions an earlier run (the pilot) drew from this dataset with the same seed."""
    if name not in config.exclude_sample:
        return items
    earlier = {q.qid for q in _sample(items, config.exclude_sample[name], config.seed, name)}
    kept = [q for q in items if q.qid not in earlier]
    log.info("%s: %d questions excluded as used by the earlier run, %d remain", name, len(earlier), len(kept))
    return kept


def _subsample_fixture(pool: list[Question], config: Config) -> list[Question]:
    selected = []
    for name, count in config.datasets.items():
        items = _without_earlier_sample([q for q in pool if q.dataset == name], config, name)
        selected.extend(_sample(items, count, config.seed, name))
    return selected


def _load_hf(name: str) -> list[Question]:
    from datasets import load_dataset  # imported lazily: not needed for analysis stages

    source = DATASET_SOURCES[name]
    rows = load_dataset(source["path"], source["name"], split=source["split"], revision=source["revision"])
    convert = {"truthfulqa": _from_truthfulqa, "triviaqa": _from_triviaqa, "sciq": _from_sciq}[name]
    questions = [convert(index, row) for index, row in enumerate(rows)]
    log.info("Loaded %d items from %s", len(questions), source["path"])
    return questions


def _from_truthfulqa(index: int, row: dict) -> Question:
    golds = [row["best_answer"], *row["correct_answers"]]
    return Question(
        qid=f"truthfulqa-{index:04d}",
        dataset="truthfulqa",
        question=row["question"].strip(),
        gold_answers=_unique(golds),
        incorrect_answers=_unique(row["incorrect_answers"]),
    )


def _from_triviaqa(index: int, row: dict) -> Question:
    answer = row["answer"]
    golds = [answer["value"], *answer.get("aliases", [])]
    return Question(
        qid=f"triviaqa-{row.get('question_id', index)}",
        dataset="triviaqa",
        question=row["question"].strip(),
        gold_answers=_unique(golds),
    )


def _from_sciq(index: int, row: dict) -> Question:
    distractors = [row.get(f"distractor{i}", "") for i in (1, 2, 3)]
    return Question(
        qid=f"sciq-{index:04d}",
        dataset="sciq",
        question=row["question"].strip(),
        gold_answers=[row["correct_answer"].strip()],
        incorrect_answers=_unique(distractors),
    )


def _unique(values: list[str]) -> list[str]:
    seen, out = set(), []
    for value in values:
        value = (value or "").strip()
        if value and value.lower() not in seen:
            seen.add(value.lower())
            out.append(value)
    return out

import os
from pathlib import Path

import pytest

from pressure_probe.config import load_config
from pressure_probe.stages.split import CR, CW, EXCL, INC, assign_group, matched_sample
from pressure_probe.text import alias_match, clean_answer, is_refusal, normalize

ROOT = Path(__file__).resolve().parents[1]


def test_clean_answer_strips_prefix_and_period():
    assert clean_answer("Answer: Paris.\nMore text") == "Paris"
    assert clean_answer('  "Canberra"  ') == "Canberra"


def test_normalize_removes_articles_and_punctuation():
    assert normalize("The Eiffel Tower!") == "eiffel tower"


def test_alias_match_whole_words_only():
    assert alias_match("It is Paris, France", ["Paris"])
    assert not alias_match("Parisian", ["Paris"])
    assert not alias_match("100", ["10"])


def test_refusal_detection():
    assert is_refusal("I don't know")
    assert is_refusal("")
    assert not is_refusal("Mount Everest")


@pytest.mark.parametrize("correct,frac,expected", [
    (False, 1.0, CW), (True, 0.9, CR), (True, 0.8, INC), (False, 0.5, INC), (None, 1.0, EXCL)])
def test_assign_group(correct, frac, expected):
    assert assign_group(correct, frac, 0.9) == expected


def test_matched_sample_balanced_within_dataset_and_deterministic():
    rows = [{"qid": f"a{i}", "dataset": "a", "group": CW if i < 3 else CR} for i in range(10)]
    rows += [{"qid": f"b{i}", "dataset": "b", "group": CW if i < 1 else CR} for i in range(5)]
    first = matched_sample(rows, 1.0, seed=1)
    assert first == matched_sample(rows, 1.0, seed=1)
    groups = {r["qid"]: r for r in rows}
    for dataset in ("a", "b"):
        chosen = [groups[q] for q in first if groups[q]["dataset"] == dataset]
        assert sum(r["group"] == CW for r in chosen) == sum(r["group"] == CR for r in chosen)


@pytest.mark.parametrize("name", ["smoke", "pilot", "full", "test"])
def test_configs_load(name):
    config = load_config(ROOT / "configs" / f"{name}.yaml")
    assert config.mode == name
    assert len(config.models) == 2


def test_invalid_threshold_rejected(tmp_path):
    text = (ROOT / "configs/test.yaml").read_text().replace("consistency_threshold: 0.9", "consistency_threshold: 0.4")
    path = tmp_path / "bad.yaml"
    path.write_text(text)
    with pytest.raises(ValueError):
        load_config(path)


def test_fingerprint_ignores_results_root():
    a = load_config(ROOT / "configs/test.yaml", {"results_root": "x"})
    b = load_config(ROOT / "configs/test.yaml", {"results_root": "y"})
    assert a.fingerprint() == b.fingerprint()


def test_matched_sample_caps_consistent_errors():
    from pressure_probe.stages.split import matched_sample

    rows = [{"qid": f"w{i}", "dataset": "a", "group": "CW"} for i in range(30)]
    rows += [{"qid": f"r{i}", "dataset": "a", "group": "CR"} for i in range(40)]
    chosen = matched_sample(rows, 1.0, seed=3, max_cw=10)
    assert sum(q.startswith("w") for q in chosen) == 10 and sum(q.startswith("r") for q in chosen) == 10
    assert chosen == matched_sample(rows, 1.0, seed=3, max_cw=10)


def test_earlier_sample_is_excluded(tmp_path):
    from pressure_probe.config import load_config
    from pressure_probe.data import prepare_questions

    pilot = load_config(ROOT / "configs/test.yaml", {"results_root": str(tmp_path / "p"), "run_name": "p",
                                                     "datasets": {"truthfulqa": 4, "triviaqa": 4, "sciq": 4}})
    full = load_config(ROOT / "configs/test.yaml", {"results_root": str(tmp_path / "f"), "run_name": "f",
                                                    "exclude_sample": {"truthfulqa": 4, "triviaqa": 4, "sciq": 4}})
    previous = os.getcwd()
    os.chdir(ROOT)
    try:
        pilot_ids = {q.qid for q in prepare_questions(pilot)}
        full_ids = {q.qid for q in prepare_questions(full)}
    finally:
        os.chdir(previous)
    assert len(pilot_ids) == 12 and len(full_ids) == 18
    assert not pilot_ids & full_ids


def test_prepare_refuses_changed_question_settings(tmp_path):
    from pressure_probe.stages import prepare

    first = load_config(ROOT / "configs/test.yaml", {"results_root": str(tmp_path)})
    changed = load_config(ROOT / "configs/test.yaml", {"results_root": str(tmp_path),
                                                       "datasets": {"truthfulqa": 4, "triviaqa": None, "sciq": None}})
    previous = os.getcwd()
    os.chdir(ROOT)
    try:
        prepare.run(first)
        prepare.run(first)  # same settings: fine
        with pytest.raises(RuntimeError, match="different datasets"):
            prepare.run(changed)
    finally:
        os.chdir(previous)

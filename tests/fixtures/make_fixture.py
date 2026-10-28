"""Writes tests/fixtures/questions.jsonl: SYNTHETIC test items for the simulated backend.

These items exercise every code path (consistently right, consistently wrong and easy
to move, consistently wrong and stubborn, inconsistent, refusal). They are not data and
no result computed from them may be reported.
"""

from __future__ import annotations

import json
from pathlib import Path

PROFILES = {
    # name: (answer is correct, firm, wobble, counter_flip, doubt_flip)
    "right_firm": (True, True, 0.0, 0.0, 0.0),
    "right_firm_sycophant": (True, True, 0.0, 0.34, 0.34),
    "wrong_pliable": (False, False, 0.0, 1.0, 0.0),
    "wrong_pliable_sycophant": (False, False, 0.0, 1.0, 0.67),
    "wrong_stubborn": (False, True, 0.0, 0.0, 0.0),
    "unsure": (True, False, 0.5, 0.5, 0.5),
}

# For each dataset: profile for model A, profile for model B (cross-family disagreement).
LAYOUT = [
    ("right_firm", "right_firm"), ("right_firm", "right_firm"), ("right_firm_sycophant", "right_firm"),
    ("right_firm", "wrong_stubborn"), ("wrong_pliable", "right_firm"), ("wrong_pliable", "right_firm"),
    ("wrong_pliable_sycophant", "right_firm"), ("wrong_stubborn", "wrong_stubborn"),
    ("unsure", "right_firm"), ("refusal", "right_firm"),
]


def belief(profile: str, gold: str, wrong: str, alternative_for_right: str) -> dict:
    if profile == "refusal":
        return {"answer": "I don't know", "firm": False, "alternative": gold}
    correct, firm, wobble, counter_flip, doubt_flip = PROFILES[profile]
    answer = gold if correct else wrong
    return {
        "answer": answer,
        "firm": firm,
        "wobble": wobble,
        "counter_flip": counter_flip,
        "doubt_flip": doubt_flip,
        "alternative": alternative_for_right if correct else gold,
        "believed_true": True,
    }


def main() -> None:
    rows = []
    for dataset in ("truthfulqa", "triviaqa", "sciq"):
        for index, (profile_a, profile_b) in enumerate(LAYOUT):
            gold, wrong, other = f"alpha{index}{dataset}", f"beta{index}{dataset}", f"gamma{index}{dataset}"
            question = f"Synthetic test question {index} for {dataset}: which token is correct?"
            rows.append({
                "qid": f"{dataset}-{index:04d}",
                "dataset": dataset,
                "question": question,
                "gold_answers": [gold],
                "incorrect_answers": [wrong],
                "mock": {
                    "model_a": belief(profile_a, gold, wrong, other),
                    "model_b": belief(profile_b, gold, wrong, other),
                },
            })
    target = Path(__file__).with_name("questions.jsonl")
    target.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"wrote {len(rows)} synthetic items to {target}")


if __name__ == "__main__":
    main()

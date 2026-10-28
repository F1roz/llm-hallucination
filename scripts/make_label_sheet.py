"""Create the human labelling sheet: 50 CW and 50 CR answers from the first model.

Two authors fill `rater1` and `rater2` independently with 1 (correct) or 0 (wrong),
then save the file as results/<run>/human_labels.csv. The report computes Cohen's
kappa from it and prints it in the paper.

    python scripts/make_label_sheet.py --config configs/full.yaml
"""

from __future__ import annotations

import argparse
import csv
import random

from pressure_probe.config import load_config
from pressure_probe.data import load_questions
from pressure_probe.io_utils import read_jsonl
from pressure_probe.stages import paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--per-group", type=int, default=50)
    args = parser.parse_args()
    config = load_config(args.config)
    key = config.models[0].key
    questions = {q.qid: q for q in load_questions(config)}
    answers = {r["qid"]: r["greedy"] for r in read_jsonl(paths.generations(config, key))}
    judged = {r["qid"]: r for r in read_jsonl(paths.judgments(config, key))}
    split = read_jsonl(paths.split(config, key))
    rng = random.Random(f"{config.seed}-labels")
    rows = []
    for group in ("CW", "CR"):
        ids = sorted(r["qid"] for r in split if r["group"] == group)
        rows += rng.sample(ids, min(args.per_group, len(ids)))
    rng.shuffle(rows)  # raters should not see the groups in order
    target = config.run_dir / "human_labels.csv"
    if target.exists():
        raise SystemExit(f"{target} exists; delete it first if you want a new sheet")
    with target.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["qid", "question", "reference_answers", "model_answer", "rater1", "rater2", "notes"])
        for qid in rows:
            q = questions[qid]
            writer.writerow([qid, q.question, " | ".join(q.gold_answers[:8]), answers[qid], "", "", ""])
    print(f"Wrote {len(rows)} rows to {target}. Judge labels are kept out of the sheet on purpose.")
    print(f"Judge method counts for reference: { {m: sum(1 for r in judged.values() if r['method'] == m) for m in {r['method'] for r in judged.values()}} }")


if __name__ == "__main__":
    main()

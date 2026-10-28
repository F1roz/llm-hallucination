"""Stage 4: consistently wrong (CW), consistently right (CR), inconsistent (INC), excluded (EXCL).

The matched set holds every CW item plus a random sample of CR items of the same size,
drawn within each dataset so the dataset mix of CR follows CW.
"""

from __future__ import annotations

import random
from collections import defaultdict

from ..config import Config
from ..data import load_questions
from ..io_utils import get_logger, read_jsonl, write_json, write_jsonl
from . import paths
from .checks import require_complete

log = get_logger(__name__)

CW, CR, INC, EXCL = "CW", "CR", "INC", "EXCL"


def assign_group(correct: bool | None, frac_match: float, threshold: float) -> str:
    if correct is None:
        return EXCL
    if frac_match >= threshold:
        return CR if correct else CW
    return INC


def matched_sample(rows: list[dict], ratio: float, seed: int, max_cw: int | None = None) -> list[str]:
    """All CW items (or a seeded sample of max_cw of them) plus as many CR items per dataset."""
    by_dataset: dict[str, dict[str, list[str]]] = defaultdict(lambda: {CW: [], CR: []})
    kept_cw = _capped_cw(rows, max_cw, seed)
    for row in rows:
        if row["group"] == CR or (row["group"] == CW and row["qid"] in kept_cw):
            by_dataset[row["dataset"]][row["group"]].append(row["qid"])
    chosen = []
    for dataset in sorted(by_dataset):
        groups = by_dataset[dataset]
        cw_ids, cr_ids = sorted(groups[CW]), sorted(groups[CR])
        wanted = min(len(cr_ids), round(ratio * len(cw_ids)))
        rng = random.Random(f"{seed}-matched-{dataset}")
        chosen.extend(cw_ids)
        chosen.extend(sorted(rng.sample(cr_ids, wanted)))
    return chosen


def _capped_cw(rows: list[dict], max_cw: int | None, seed: int) -> set[str]:
    cw_ids = sorted(row["qid"] for row in rows if row["group"] == CW)
    if max_cw is None or len(cw_ids) <= max_cw:
        return set(cw_ids)
    return set(random.Random(f"{seed}-cw-cap").sample(cw_ids, max_cw))


def run(config: Config, model_key: str) -> None:
    datasets = {q.qid: q.dataset for q in load_questions(config)}
    require_complete(paths.judgments(config, model_key), set(datasets), "judge", model_key)
    require_complete(paths.sample_scores(config, model_key), set(datasets), "sample_scores", model_key)
    judged = {row["qid"]: row for row in read_jsonl(paths.judgments(config, model_key))}
    scored = {row["qid"]: row for row in read_jsonl(paths.sample_scores(config, model_key))}
    rows = []
    for qid, score in scored.items():
        verdict = judged[qid]
        rows.append({
            "qid": qid,
            "dataset": datasets[qid],
            "correct": verdict["correct"],
            "frac_match": score["frac_match"],
            "group": assign_group(verdict["correct"], score["frac_match"], config.consistency_threshold),
            "strict_group": assign_group(verdict["correct"], score["frac_match"], 1.0),
        })
    write_jsonl(paths.split(config, model_key), rows)
    chosen = matched_sample(rows, config.matched_ratio, config.seed, config.max_cw)
    write_json(paths.matched(config, model_key), chosen)
    counts = {g: sum(1 for r in rows if r["group"] == g) for g in (CW, CR, INC, EXCL)}
    log.info("[%s] split counts %s; matched set %d", model_key, counts, len(chosen))
    if counts[CW] < 15:
        log.warning("[%s] only %d CW items; AUROC intervals will be wide", model_key, counts[CW])

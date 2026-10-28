"""Stage 7: decide with NLI which pressure replies abandoned the original answer."""

from __future__ import annotations

from ..config import Config
from ..data import load_questions
from ..io_utils import get_logger, read_json, read_jsonl, write_jsonl
from ..nli import load_nli
from . import paths
from .checks import already_complete, require_complete
from .timing import timed

log = get_logger(__name__)
KINDS = ("counter", "doubt", "random_counter")


def run(config: Config, model_key: str) -> None:
    target = paths.probe_scores(config, model_key)
    questions = {q.qid: q.question for q in load_questions(config)}
    require_complete(paths.split(config, model_key), set(questions), "split", model_key)
    matched = set(read_json(paths.matched(config, model_key)))
    require_complete(paths.probes(config, model_key), matched, "probe", model_key)
    if already_complete(target, matched, "probe_scores", model_key):
        return
    probes = read_jsonl(paths.probes(config, model_key))
    judge = load_nli(config)
    items = []
    for row in probes:
        question = questions[row["qid"]]
        items.append((question, row["answer"], row["alternative"]))
        for kind in KINDS:
            items.extend((question, reply, row["answer"]) for reply in row.get(f"{kind}_replies", []))
    with timed(paths.timings(config, model_key), "probe_nli"):
        judge.compare_many(items)
        rows = [_score(judge, questions[row["qid"]], row) for row in probes]
    write_jsonl(target, rows)
    log.info("[%s] scored %d probe rows", model_key, len(rows))


def _score(judge, question: str, row: dict) -> dict:
    out = {"qid": row["qid"],
           "alternative_equivalent": judge.compare_many([(question, row["answer"], row["alternative"])])[0].equivalent}
    for kind in KINDS:
        replies = row.get(f"{kind}_replies", [])
        verdicts = judge.compare_many([(question, reply, row["answer"]) for reply in replies])
        out[f"{kind}_flips"] = [not v.equivalent for v in verdicts]
    return out

"""Stage 5: pressure probes on the matched set.

For each answer a: ask for a plausible different answer b, then re-ask the question
under three counter-claims naming b and three content-free challenges. In full runs a
random alternative (another question's answer from the same dataset) is also tested.
"""

from __future__ import annotations

import random

from ..backends import GenRequest, load_backend
from ..config import Config
from ..data import load_questions
from ..io_utils import append_jsonl, done_ids, get_logger, read_json, read_jsonl
from ..prompts import alternative_messages, counter_follow_ups, doubt_follow_ups, pressure_messages
from ..text import clean_answer, normalize
from . import paths
from .checks import require_complete
from .timing import timed

log = get_logger(__name__)
CHUNK = 64
ALT_RETRIES = 3


def run(config: Config, model_key: str) -> None:
    target = paths.probes(config, model_key)
    all_ids = {q.qid for q in load_questions(config)}
    require_complete(paths.split(config, model_key), all_ids, "split", model_key)
    matched = read_json(paths.matched(config, model_key))
    finished = done_ids(target)
    todo = [qid for qid in matched if qid not in finished]
    if not todo:
        log.info("[%s] probes already complete", model_key)
        return
    questions = {q.qid: q for q in load_questions(config)}
    greedy = {row["qid"]: row["greedy"] for row in read_jsonl(paths.generations(config, model_key))}
    random_alts = _random_alternatives(config, questions, greedy, matched) if config.run_random_alternative else {}
    backend = load_backend(config, config.model(model_key))
    counters = {"generations": 0}
    with timed(paths.timings(config, model_key), "probe", counters):
        for start in range(0, len(todo), CHUNK):
            chunk = todo[start:start + CHUNK]
            rows = _probe_chunk(config, backend, [(questions[qid], greedy[qid]) for qid in chunk],
                                random_alts, counters)
            append_jsonl(target, rows)
            log.info("[%s] probed %d / %d", model_key, start + len(chunk), len(todo))
    backend.close()


def _probe_chunk(config: Config, backend, items: list, random_alts: dict, counters: dict) -> list[dict]:
    alternatives = _alternatives(config, backend, items, counters)
    requests, layout = [], []
    for (q, answer), alt in zip(items, alternatives):
        follow_ups = [("counter", f) for f in counter_follow_ups(answer, alt)[:config.n_pressure]]
        follow_ups += [("doubt", f) for f in doubt_follow_ups()[:config.n_pressure]]
        if q.qid in random_alts:
            follow_ups += [("random_counter", f)
                           for f in counter_follow_ups(answer, random_alts[q.qid])[:config.n_pressure]]
        for kind, text in follow_ups:
            requests.append(GenRequest(pressure_messages(q.question, answer, text), n=1, temperature=0.0,
                                       max_new_tokens=config.max_new_tokens, seed=config.seed))
            layout.append((q.qid, kind))
    results = backend.generate(requests)
    counters["generations"] += len(requests)

    replies: dict[str, dict[str, list[str]]] = {q.qid: {} for q, _ in items}
    for (qid, kind), result in zip(layout, results):
        replies[qid].setdefault(kind, []).append(clean_answer(result.texts[0]))
    rows = []
    for (q, answer), alt in zip(items, alternatives):
        rows.append({
            "qid": q.qid,
            "answer": answer,
            "alternative": alt,
            "counter_replies": replies[q.qid].get("counter", []),
            "doubt_replies": replies[q.qid].get("doubt", []),
            "random_alternative": random_alts.get(q.qid),
            "random_counter_replies": replies[q.qid].get("random_counter", []),
        })
    return rows


def _alternatives(config: Config, backend, items: list, counters: dict) -> list[str]:
    """Greedy alternative first; sampled retries when it repeats the answer."""
    requests = [GenRequest(alternative_messages(q.question, a), n=1, temperature=0.0, max_new_tokens=24,
                           seed=config.seed) for q, a in items]
    alternatives = [clean_answer(r.texts[0]) for r in backend.generate(requests)]
    counters["generations"] += len(requests)
    for attempt in range(ALT_RETRIES):
        bad = [i for i, (alt, (_, a)) in enumerate(zip(alternatives, items)) if _same(alt, a)]
        if not bad:
            break
        retry = [GenRequest(alternative_messages(items[i][0].question, items[i][1]), n=1, temperature=0.8,
                            max_new_tokens=24, seed=config.seed + 100 + attempt) for i in bad]
        for i, result in zip(bad, backend.generate(retry)):
            alternatives[i] = clean_answer(result.texts[0])
        counters["generations"] += len(retry)
    return alternatives


def _same(a: str, b: str) -> bool:
    na, nb = normalize(a), normalize(b)
    return not na or na == nb or na in nb or nb in na


def _random_alternatives(config: Config, questions: dict, greedy: dict, matched: list[str]) -> dict[str, str]:
    """Another question's greedy answer from the same dataset: an implausible counter-claim."""
    rng = random.Random(f"{config.seed}-random-alt")
    pools: dict[str, list[str]] = {}
    for qid, answer in greedy.items():
        pools.setdefault(questions[qid].dataset, []).append(qid)
    chosen = {}
    for qid in matched:
        pool = [other for other in pools[questions[qid].dataset]
                if other != qid and not _same(greedy[other], greedy[qid])]
        if pool:
            chosen[qid] = greedy[rng.choice(sorted(pool))]
    return chosen

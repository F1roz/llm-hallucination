"""Stage 6: MetaQA baseline (Yang et al., 2025) on the matched set.

The model writes synonym and antonym mutations of its own answer, then judges each
mutation as Yes, No or Not Sure in a fresh conversation.
"""

from __future__ import annotations

import re

from ..backends import GenRequest, load_backend
from ..config import Config
from ..data import load_questions
from ..io_utils import append_jsonl, done_ids, get_logger, read_json, read_jsonl
from ..prompts import metaqa_antonym_messages, metaqa_synonym_messages, metaqa_verify_messages
from . import paths
from .checks import require_complete
from .timing import timed

log = get_logger(__name__)
CHUNK = 64
_NUMBERED = re.compile(r"^\s*\d+[.)]\s*(.+)$")


def parse_numbered(text: str, limit: int) -> list[str]:
    items = []
    for line in text.splitlines():
        match = _NUMBERED.match(line)
        if match and match.group(1).strip():
            items.append(match.group(1).strip())
    return items[:limit]


def parse_verdict(text: str) -> str:
    lowered = text.strip().lower()
    if lowered.startswith("not sure") or lowered.startswith("unsure"):
        return "not_sure"
    if lowered.startswith("yes"):
        return "yes"
    if lowered.startswith("no"):
        return "no"
    return "not_sure"


def run(config: Config, model_key: str) -> None:
    if not config.run_metaqa:
        return
    target = paths.metaqa(config, model_key)
    all_ids = {q.qid for q in load_questions(config)}
    require_complete(paths.split(config, model_key), all_ids, "split", model_key)
    matched = read_json(paths.matched(config, model_key))
    finished = done_ids(target)
    todo = [qid for qid in matched if qid not in finished]
    if not todo:
        log.info("[%s] MetaQA already complete", model_key)
        return
    questions = {q.qid: q for q in load_questions(config)}
    greedy = {row["qid"]: row["greedy"] for row in read_jsonl(paths.generations(config, model_key))}
    backend = load_backend(config, config.model(model_key))
    counters = {"generations": 0}
    with timed(paths.timings(config, model_key), "metaqa", counters):
        for start in range(0, len(todo), CHUNK):
            chunk = todo[start:start + CHUNK]
            append_jsonl(target, _chunk(config, backend, [(questions[q], greedy[q]) for q in chunk], counters))
            log.info("[%s] MetaQA %d / %d", model_key, start + len(chunk), len(todo))
    backend.close()


def _chunk(config: Config, backend, items: list, counters: dict) -> list[dict]:
    n = config.metaqa_mutations
    gen_requests = []
    for q, answer in items:
        gen_requests.append(GenRequest(metaqa_synonym_messages(q.question, answer, n), max_new_tokens=256,
                                       seed=config.seed))
        gen_requests.append(GenRequest(metaqa_antonym_messages(q.question, answer, n), max_new_tokens=256,
                                       seed=config.seed))
    lists = backend.generate(gen_requests)
    counters["generations"] += len(gen_requests)

    mutations = []
    for index in range(len(items)):
        synonyms = parse_numbered(lists[2 * index].texts[0], n)
        antonyms = parse_numbered(lists[2 * index + 1].texts[0], n)
        mutations.append((synonyms, antonyms))
    verify_requests, layout = [], []
    for index, (synonyms, antonyms) in enumerate(mutations):
        for kind, statements in (("synonym", synonyms), ("antonym", antonyms)):
            for statement in statements:
                verify_requests.append(GenRequest(metaqa_verify_messages(statement), max_new_tokens=4,
                                                  seed=config.seed))
                layout.append((index, kind))
    verdicts = backend.generate(verify_requests) if verify_requests else []
    counters["generations"] += len(verify_requests)

    per_item = [{"synonym": [], "antonym": []} for _ in items]
    for (index, kind), result in zip(layout, verdicts):
        per_item[index][kind].append(parse_verdict(result.texts[0]))
    rows = []
    for (q, _), (synonyms, antonyms), verdict in zip(items, mutations, per_item):
        rows.append({"qid": q.qid, "synonyms": synonyms, "antonyms": antonyms,
                     "synonym_verdicts": verdict["synonym"], "antonym_verdicts": verdict["antonym"]})
    return rows

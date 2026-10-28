"""Stage 2: label each greedy answer correct, wrong or refusal.

Order: refusal check, alias match, then GPT-4o-mini when available, else NLI
equivalence against every gold answer. The method used is stored per row.
"""

from __future__ import annotations

from ..backends import GenRequest
from ..config import Config
from ..data import load_questions
from ..io_utils import get_logger, read_jsonl, write_json, write_jsonl
from ..prompts import judge_messages
from ..text import alias_match, is_refusal
from . import paths
from .checks import already_complete, require_complete
from .timing import timed

log = get_logger(__name__)


def run(config: Config, model_key: str) -> None:
    target = paths.judgments(config, model_key)
    questions = {q.qid: q for q in load_questions(config)}
    require_complete(paths.generations(config, model_key), set(questions), "generate", model_key)
    if already_complete(target, set(questions), "judge", model_key):
        return
    generations = read_jsonl(paths.generations(config, model_key))
    use_llm = _resolve_backend(config)
    rows, pending = [], []
    for gen in generations:
        q = questions[gen["qid"]]
        answer = gen["greedy"]
        if is_refusal(answer):
            rows.append({"qid": q.qid, "correct": None, "refusal": True, "method": "refusal"})
        elif alias_match(answer, q.gold_answers):
            rows.append({"qid": q.qid, "correct": True, "refusal": False, "method": "alias"})
        else:
            pending.append((q, answer))
    with timed(paths.timings(config, model_key), "judge"):
        rows.extend(_llm_judge(config, pending) if use_llm else _nli_judge(config, pending))
    write_jsonl(target, rows)
    counts = {m: sum(1 for r in rows if r["method"] == m) for m in {r["method"] for r in rows}}
    write_json(config.model_dir(model_key) / "judge_methods.json", counts)
    log.info("[%s] judged %d answers: %s", model_key, len(rows), counts)


def _resolve_backend(config: Config) -> bool:
    from ..backends.openai_backend import openai_available

    if config.judge.backend == "openai":
        return True
    if config.judge.backend == "match_nli":
        return False
    available = openai_available()
    log.info("Judge backend: %s", "openai" if available else "alias + NLI (no OPENAI_API_KEY)")
    return available


def _llm_judge(config: Config, pending: list) -> list[dict]:
    from ..backends.openai_backend import OpenAIBackend

    backend = OpenAIBackend(config.judge.openai_model)
    requests = [GenRequest(judge_messages(q.question, a, q.gold_answers, q.incorrect_answers), n=1,
                           temperature=0.0, max_new_tokens=3, seed=config.seed) for q, a in pending]
    results = backend.generate(requests)
    rows = []
    for (q, _), result in zip(pending, results):
        verdict = result.texts[0].strip().lower()
        rows.append({"qid": q.qid, "correct": verdict.startswith("yes"), "refusal": False,
                     "method": f"llm:{config.judge.openai_model}", "judge_raw": result.texts[0]})
    return rows


def _nli_judge(config: Config, pending: list) -> list[dict]:
    from ..nli import load_nli

    judge = load_nli(config)
    rows = []
    for q, answer in pending:
        comparisons = judge.compare_many([(q.question, answer, gold) for gold in q.gold_answers])
        rows.append({"qid": q.qid, "correct": any(c.equivalent for c in comparisons), "refusal": False,
                     "method": "nli"})
    return rows

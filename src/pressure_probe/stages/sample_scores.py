"""Stage 3: NLI over sampled answers.

For each question it records which samples match the greedy answer (used for the
consistency split), Semantic Entropy, SelfCheckGPT-NLI and cross-family disagreement.
"""

from __future__ import annotations

from ..config import Config
from ..data import load_questions
from ..io_utils import get_logger, read_jsonl, write_jsonl
from ..nli import EquivalenceJudge, load_nli
from ..scoring.baselines import cluster_entropy, greedy_clusters, selfcheck_score
from ..text import normalize
from . import paths
from .checks import already_complete, require_complete
from .timing import timed

log = get_logger(__name__)
CHUNK = 256


def run(config: Config, model_key: str) -> None:
    target = paths.sample_scores(config, model_key)
    questions = {q.qid: q for q in load_questions(config)}
    require_complete(paths.generations(config, model_key), set(questions), "generate", model_key)
    for other in config.other_models(model_key)[:1]:  # the cross family verifier must be finished too
        require_complete(paths.generations(config, other.key), set(questions), "generate", other.key)
    if already_complete(target, set(questions), "sample_scores", model_key):
        return
    generations = read_jsonl(paths.generations(config, model_key))
    others = _other_greedy(config, model_key)
    judge = load_nli(config)
    rows = []
    with timed(paths.timings(config, model_key), "sample_nli"):
        for start in range(0, len(generations), CHUNK):
            chunk = generations[start:start + CHUNK]
            _prefetch(judge, questions, chunk, others)
            rows.extend(_score(judge, questions[g["qid"]].question, g, others) for g in chunk)
            log.info("[%s] NLI scored %d / %d", model_key, start + len(chunk), len(generations))
    write_jsonl(target, rows)


def _other_greedy(config: Config, model_key: str) -> dict[str, str]:
    """Greedy answers of the first other model, used as the cross-family verifier."""
    for spec in config.other_models(model_key):
        path = paths.generations(config, spec.key)
        if path.exists():
            log.info("[%s] cross-family verifier: %s", model_key, spec.key)
            return {row["qid"]: row["greedy"] for row in read_jsonl(path)}
    log.warning("[%s] no other model generations found; cross-family score skipped", model_key)
    return {}


def _prefetch(judge: EquivalenceJudge, questions: dict, chunk: list[dict], others: dict[str, str]) -> None:
    """One batched NLI call for every pair the chunk needs."""
    items = []
    for gen in chunk:
        question = questions[gen["qid"]].question
        uniques = _uniques(gen["samples"])
        items.extend((question, s, gen["greedy"]) for s in uniques)
        items.extend((question, a, b) for i, a in enumerate(uniques) for b in uniques[i + 1:])
        if gen["qid"] in others:
            items.append((question, gen["greedy"], others[gen["qid"]]))
    judge.compare_many(items)


def _score(judge: EquivalenceJudge, question: str, gen: dict, others: dict[str, str]) -> dict:
    samples, greedy = gen["samples"], gen["greedy"]
    vs_greedy = judge.compare_many([(question, s, greedy) for s in samples])
    matches = [c.equivalent for c in vs_greedy]
    clusters = greedy_clusters(samples, lambda a, b: judge.compare_many([(question, a, b)])[0].equivalent)
    row = {
        "qid": gen["qid"],
        "matches_greedy": matches,
        "frac_match": sum(matches) / len(matches),
        "semantic_entropy": cluster_entropy(clusters),
        "n_clusters": len(set(clusters)),
        "selfcheck_nli": selfcheck_score([(c.contradict_forward, c.entail_forward) for c in vs_greedy]),
        "cross_family": None,
        "cross_family_equivalent": None,
        "verifier_answer": others.get(gen["qid"]),
    }
    if gen["qid"] in others:
        cross = judge.compare_many([(question, greedy, others[gen["qid"]])])[0]
        row["cross_family"] = 1.0 - 0.5 * (cross.entail_forward + cross.entail_backward)
        row["cross_family_equivalent"] = cross.equivalent
    return row


def _uniques(samples: list[str]) -> list[str]:
    seen, out = set(), []
    for s in samples:
        key = normalize(s)
        if key not in seen:
            seen.add(key)
            out.append(s)
    return out

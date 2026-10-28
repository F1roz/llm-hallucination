"""Stage 1: greedy answer, sampled answers, P(True) and verbalised confidence for every question."""

from __future__ import annotations

from ..backends import GenRequest, load_backend
from ..config import Config
from ..data import Question, load_questions
from ..io_utils import append_jsonl, done_ids, get_logger, write_json
from ..prompts import answer_messages, ptrue_messages, verbal_messages
from ..text import clean_answer
from . import paths
from .timing import timed

log = get_logger(__name__)
CHUNK = 64


def run(config: Config, model_key: str) -> None:
    target = paths.generations(config, model_key)
    finished = done_ids(target)
    todo = [q for q in load_questions(config) if q.qid not in finished]
    if not todo:
        log.info("[%s] generation already complete", model_key)
        return
    backend = load_backend(config, config.model(model_key))
    write_json(paths.backend_info(config, model_key), {"backend": backend.name, "precision": backend.precision})
    counters = {"generations": 0}
    with timed(paths.timings(config, model_key), "generate", counters):
        for start in range(0, len(todo), CHUNK):
            chunk = todo[start:start + CHUNK]
            rows = _generate_chunk(config, backend, chunk, counters)
            append_jsonl(target, rows)
            log.info("[%s] generated %d / %d", model_key, start + len(chunk), len(todo))
    backend.close()


def _generate_chunk(config: Config, backend, chunk: list[Question], counters: dict[str, int]) -> list[dict]:
    greedy_requests = [GenRequest(answer_messages(q.question), n=1, temperature=0.0,
                                  max_new_tokens=config.max_new_tokens, want_logprobs=True, seed=config.seed)
                       for q in chunk]
    sample_requests = [GenRequest(answer_messages(q.question), n=config.n_samples,
                                  temperature=config.sample_temperature, max_new_tokens=config.max_new_tokens,
                                  seed=config.seed + i) for i, q in enumerate(chunk)]
    greedy = backend.generate(greedy_requests)
    samples = backend.generate(sample_requests)
    answers = [clean_answer(g.texts[0]) for g in greedy]

    ptrue_requests = [GenRequest(ptrue_messages(q.question, a), n=1, temperature=0.0, max_new_tokens=2,
                                 first_token_top_k=20, seed=config.seed) for q, a in zip(chunk, answers)]
    verbal_requests = [GenRequest(verbal_messages(q.question, a), n=1, temperature=0.0, max_new_tokens=6,
                                  seed=config.seed) for q, a in zip(chunk, answers)]
    ptrue = backend.generate(ptrue_requests)
    verbal = backend.generate(verbal_requests)
    counters["generations"] += len(chunk) * (3 + config.n_samples)

    rows = []
    for q, g, s, p, v, answer in zip(chunk, greedy, samples, ptrue, verbal, answers):
        rows.append({
            "qid": q.qid,
            "greedy_raw": g.texts[0],
            "greedy": answer,
            "greedy_logprobs": (g.token_logprobs or [[]])[0],
            "samples_raw": s.texts,
            "samples": [clean_answer(t) for t in s.texts],
            "ptrue_first_token": p.first_token_logprobs,
            "ptrue_text": p.texts[0],
            "verbal_raw": v.texts[0],
        })
    return rows

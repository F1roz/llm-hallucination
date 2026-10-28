"""Stage 8: compute every reported number and write summary.json, the single source of truth."""

from __future__ import annotations

import datetime as dt

from .. import analysis
from ..config import Config
from ..data import load_questions
from ..io_utils import get_logger, read_json, read_jsonl, write_json
from . import paths
from .assemble import build_table
from .checks import require_complete

log = get_logger(__name__)


def run(config: Config) -> dict:
    models = {}
    for spec in config.models:
        _require_inputs(config, spec.key)
        models[spec.key] = _model_summary(config, spec.key, spec.display_name)
    summary = {
        "run_name": config.run_name,
        "mode": config.mode,
        "config_fingerprint": config.fingerprint(),
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "settings": {
            "max_cw": config.max_cw,
            "exclude_sample": config.exclude_sample,
            "n_samples": config.n_samples,
            "sample_temperature": config.sample_temperature,
            "consistency_threshold": config.consistency_threshold,
            "n_pressure": config.n_pressure,
            "metaqa_mutations": config.metaqa_mutations,
            "bootstrap_samples": config.bootstrap_samples,
            "datasets": config.datasets,
            "nli_model": config.nli_model,
            "seed": config.seed,
        },
        "models": models,
    }
    write_json(paths.summary(config), summary)
    log.info("Wrote %s", paths.summary(config))
    return summary


def _require_inputs(config: Config, key: str) -> None:
    """Every model must have finished every stage, so the paper never mixes complete and partial results."""
    all_ids = {q.qid for q in load_questions(config)}
    require_complete(paths.split(config, key), all_ids, "split", key)
    matched = set(read_json(paths.matched(config, key)))
    require_complete(paths.probe_scores(config, key), matched, "probe_scores", key)
    if config.run_metaqa:
        require_complete(paths.metaqa(config, key), matched, "metaqa", key)


def _model_summary(config: Config, key: str, display_name: str) -> dict:
    table = build_table(config, key)
    frame = analysis.matched_frame(table)
    methods = analysis.method_results(config, table, frame)
    text_best = analysis.best_baseline(methods, "text")
    any_best = analysis.best_baseline(methods, None)
    comparison = analysis.comparisons(config, frame, methods)
    ablation = analysis.ablations(config, table, frame, text_best)
    return {
        "display_name": display_name,
        "backend": _read(paths.backend_info(config, key)),
        "judge_methods": _read(config.model_dir(key) / "judge_methods.json"),
        "timings": _read(paths.timings(config, key)),
        "counts": analysis.counts(table),
        "matched": {"n": int(len(frame)), "CW": int(frame["label"].sum()), "CR": int((1 - frame["label"]).sum())},
        "methods": methods,
        "best_text_baseline": text_best,
        "best_baseline": any_best,
        "comparisons_vs_ours": comparison,
        "flip_rates": analysis.flip_rates(config, frame),
        "per_dataset": analysis.per_dataset(frame, methods, text_best),
        "ablations": ablation,
        "hypotheses": analysis.hypotheses(methods, comparison, ablation, text_best),
        "examples": _examples(config, key, frame),
    }


def _examples(config: Config, key: str, frame) -> list[dict]:
    """Rule based picks: top two CW by doubt flip rate, bottom CW, top CR."""
    questions = {q.qid: q for q in load_questions(config)}
    probes = {r["qid"]: r for r in read_jsonl(paths.probes(config, key))}
    scored = frame[frame["dfr"].notna()]
    cw = scored[scored["group"] == "CW"].sort_values("dfr", ascending=False, kind="mergesort")
    cr = scored[scored["group"] == "CR"].sort_values("dfr", ascending=False, kind="mergesort")
    picks = list(cw.index[:2]) + list(cw.index[-1:]) + list(cr.index[:1])
    out, seen = [], set()
    for qid in picks:
        if qid in seen or qid not in probes:
            continue
        seen.add(qid)
        probe, row = probes[qid], frame.loc[qid]
        out.append({"qid": qid, "group": row["group"], "question": questions[qid].question,
                    "reference": questions[qid].gold_answers[0], "answer": probe["answer"],
                    "alternative": probe["alternative"],
                    "reply": (probe["doubt_replies"] or [""])[0],
                    "f_counter": float(row["f_counter"]), "f_doubt": float(row["f_doubt"])})
    return out


def _read(path):
    return read_json(path) if path.exists() else {}

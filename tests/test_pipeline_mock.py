"""Integration test on the simulated backend. Checks plumbing, not scientific claims."""

import os

from pressure_probe.io_utils import read_json, read_jsonl
from pressure_probe.pipeline import run_all
from pressure_probe.stages import paths

from .conftest import ROOT


def test_every_stage_wrote_output(mock_run):
    for spec in mock_run.models:
        for path in (paths.generations, paths.judgments, paths.sample_scores, paths.split, paths.probes,
                     paths.metaqa, paths.probe_scores, paths.scores_table):
            assert path(mock_run, spec.key).exists(), path.__name__
    assert paths.summary(mock_run).exists()


def test_split_matches_fixture_design(mock_run):
    rows = read_jsonl(paths.split(mock_run, "model_a"))
    counts = {g: sum(r["group"] == g for r in rows) for g in ("CW", "CR", "INC", "EXCL")}
    assert counts == {"CW": 12, "CR": 12, "INC": 3, "EXCL": 3}


def test_matched_set_is_balanced(mock_run):
    matched = read_json(paths.matched(mock_run, "model_a"))
    groups = {r["qid"]: r["group"] for r in read_jsonl(paths.split(mock_run, "model_a"))}
    assert sum(groups[q] == "CW" for q in matched) == sum(groups[q] == "CR" for q in matched)


def test_pliable_errors_flip_more(mock_run):
    summary = read_json(paths.summary(mock_run))
    flips = summary["models"]["model_a"]["flip_rates"]
    assert flips["CW"]["f_counter"]["value"] > flips["CR"]["f_counter"]["value"]
    assert flips["CW"]["f_doubt"]["value"] > flips["CR"]["f_doubt"]["value"]


def test_main_score_is_doubt_flip_rate(mock_run):
    model = read_json(paths.summary(mock_run))["models"]["model_a"]
    assert model["methods"]["dfr"]["is_ours"] and model["methods"]["dfr"]["calls"] == 4
    assert model["methods"]["cps"]["role"] == "variant"
    assert "dfr" not in model["comparisons_vs_ours"] and "cps" not in model["comparisons_vs_ours"]
    assert model["hypotheses"]["H2"]["dfr"] == model["methods"]["dfr"]["auroc_matched"]["value"]


def test_sampling_baselines_are_uninformative_on_consistent_split(mock_run):
    methods = read_json(paths.summary(mock_run))["models"]["model_a"]["methods"]
    assert methods["semantic_entropy"]["auroc_matched"]["value"] == 0.5


def test_summary_contains_all_sections(mock_run):
    model = read_json(paths.summary(mock_run))["models"]["model_a"]
    for key in ("counts", "methods", "comparisons_vs_ours", "flip_rates", "per_dataset", "ablations",
                "hypotheses", "examples", "timings", "backend"):
        assert key in model
    assert set(model["hypotheses"]) == {"H1", "H2", "H3", "H4"}


def test_rerun_is_idempotent(mock_run):
    before = paths.generations(mock_run, "model_a").read_text()
    previous = os.getcwd()
    os.chdir(ROOT)
    try:
        run_all(mock_run, str(ROOT / "configs/test.yaml"), isolate=False)
    finally:
        os.chdir(previous)
    assert paths.generations(mock_run, "model_a").read_text() == before


def test_resume_after_partial_generation(mock_run, tmp_path):
    from pressure_probe.config import load_config
    from pressure_probe.stages import generate, prepare

    previous = os.getcwd()
    os.chdir(ROOT)
    try:
        config = load_config(ROOT / "configs/test.yaml", {"results_root": str(tmp_path)})
        prepare.run(config)
        generate.run(config, "model_a")
        target = paths.generations(config, "model_a")
        lines = target.read_text().splitlines()
        target.write_text("\n".join(lines[:10]) + "\n")
        generate.run(config, "model_a")
        assert len(read_jsonl(target)) == len(lines)
        assert len({r["qid"] for r in read_jsonl(target)}) == len(lines)
    finally:
        os.chdir(previous)

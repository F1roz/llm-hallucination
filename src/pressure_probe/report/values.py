"""Single values for \\RES{key}: settings, author details, counts, scores and agreement."""

from __future__ import annotations

import csv
from pathlib import Path

import yaml

from ..config import Config
from ..data import load_questions, questions_path
from ..methods import calls_per_question
from .fmt import TODO, integer, latex_escape, num

PAPER_DIR = Path("paper")
VERIFIED_FULL_SIZES = {"truthfulqa": 817, "sciq": 1000}  # generation split; SciQ test split


def setting_values(config: Config) -> dict[str, str]:
    calls = calls_per_question(config)
    values = {
        "set.samples": str(config.n_samples),
        "set.temperature": f"{config.sample_temperature:g}",
        "set.tau": f"{config.consistency_threshold:g}",
        "set.k": str(config.n_pressure),
        "set.mutations": str(config.metaqa_mutations),
        "set.bootstrap": f"{config.bootstrap_samples:,}",
        "set.seed": str(config.seed),
        "set.cps_calls": str(calls["cps"]),
        "set.dfr_calls": str(calls["dfr"]),
        "set.max_cw": str(config.max_cw) if config.max_cw else "all",
    }
    for index, spec in enumerate(config.models[:2], start=1):
        values[f"m{index}.name"] = latex_escape(spec.display_name)
    values.update(_dataset_sizes(config))
    return values


def _dataset_sizes(config: Config) -> dict[str, str]:
    if questions_path(config).exists():
        questions = load_questions(config)
        sizes = {name: sum(1 for q in questions if q.dataset == name) for name in config.datasets}
    else:
        sizes = {name: count if count is not None else _full_size(config, name) for name, count in config.datasets.items()}
    values = {f"n.{name}": integer(size) for name, size in sizes.items()}
    known = [s for s in sizes.values() if s is not None]
    values["n.total"] = integer(sum(known)) if len(known) == len(sizes) else TODO
    return values


def _full_size(config: Config, name: str) -> int | None:
    """Whole dataset size, minus the questions excluded as used by the pilot."""
    full = VERIFIED_FULL_SIZES.get(name)
    if full is None:
        return None
    return full - (config.exclude_sample.get(name) or 0)


def author_values(paper_dir: Path = PAPER_DIR) -> dict[str, str]:
    path = paper_dir / "authors.yaml"
    authors = yaml.safe_load(path.read_text()).get("authors", []) if path.exists() else []
    values = {}
    for index in range(1, 4):
        entry = authors[index - 1] if index <= len(authors) else {}
        for field in ("name", "affiliation", "email"):
            text = (entry or {}).get(field) or ""
            values[f"author.{index}.{field}"] = latex_escape(text) if text.strip() else TODO
    return values


def result_values(summary: dict | None, config: Config) -> dict[str, str]:
    if not summary:
        return {}
    values = {}
    for index, spec in enumerate(config.models[:2], start=1):
        model = summary["models"].get(spec.key)
        if not model:
            continue
        prefix = f"m{index}"
        counts = model["counts"]
        for group in ("CW", "CR", "INC", "EXCL"):
            values[f"{prefix}.{group.lower()}"] = integer(counts[group])
        methods = model["methods"]
        for key, method in methods.items():
            values[f"{prefix}.{key}.auroc"] = num(method["auroc_matched"]["value"])
        values[f"{prefix}.precision"] = latex_escape(str(model.get("backend", {}).get("precision", "")) or "") or TODO
    return values


def kappa_value(config: Config) -> dict[str, str]:
    """Cohen's kappa between the two human raters in results/<run>/human_labels.csv."""
    path = config.run_dir / "human_labels.csv"
    if not path.exists():
        return {"label.kappa": TODO}
    rows = [r for r in csv.DictReader(path.open()) if r.get("rater1", "").strip() and r.get("rater2", "").strip()]
    if len(rows) < 10:
        return {"label.kappa": TODO}
    first = [_label(r["rater1"]) for r in rows]
    second = [_label(r["rater2"]) for r in rows]
    return {"label.kappa": num(cohen_kappa(first, second), 2), "label.n": str(len(rows))}


def cohen_kappa(first: list[int], second: list[int]) -> float:
    n = len(first)
    observed = sum(a == b for a, b in zip(first, second)) / n
    p1, p2 = sum(first) / n, sum(second) / n
    expected = p1 * p2 + (1 - p1) * (1 - p2)
    return 1.0 if expected == 1 else (observed - expected) / (1 - expected)


def _label(text: str) -> int:
    return 1 if text.strip().lower() in {"1", "correct", "yes", "true", "c"} else 0

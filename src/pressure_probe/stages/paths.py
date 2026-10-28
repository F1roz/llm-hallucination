"""File locations for every stage output, in one place."""

from __future__ import annotations

from pathlib import Path

from ..config import Config


def generations(config: Config, model_key: str) -> Path:
    return config.model_dir(model_key) / "generations.jsonl"


def judgments(config: Config, model_key: str) -> Path:
    return config.model_dir(model_key) / "judgments.jsonl"


def sample_scores(config: Config, model_key: str) -> Path:
    return config.model_dir(model_key) / "sample_scores.jsonl"


def split(config: Config, model_key: str) -> Path:
    return config.model_dir(model_key) / "split.jsonl"


def matched(config: Config, model_key: str) -> Path:
    return config.model_dir(model_key) / "matched_ids.json"


def probes(config: Config, model_key: str) -> Path:
    return config.model_dir(model_key) / "probes.jsonl"


def metaqa(config: Config, model_key: str) -> Path:
    return config.model_dir(model_key) / "metaqa.jsonl"


def probe_scores(config: Config, model_key: str) -> Path:
    return config.model_dir(model_key) / "probe_scores.jsonl"


def timings(config: Config, model_key: str) -> Path:
    return config.model_dir(model_key) / "timings.json"


def backend_info(config: Config, model_key: str) -> Path:
    return config.model_dir(model_key) / "backend.json"


def scores_table(config: Config, model_key: str) -> Path:
    return config.model_dir(model_key) / "scores.csv"


def summary(config: Config) -> Path:
    return config.run_dir / "summary.json"

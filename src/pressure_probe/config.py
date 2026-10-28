"""Experiment configuration: loading, defaults and a stable hash."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelSpec:
    """One target model. The low memory variant is used on GPUs under 20 GB."""

    key: str
    display_name: str
    hf_id: str
    revision: str | None = None
    low_memory_hf_id: str | None = None
    low_memory_revision: str | None = None
    low_memory_quantization: str | None = None  # "awq" for vLLM, "bnb4" for transformers


@dataclass
class JudgeConfig:
    backend: str = "auto"  # auto | openai | match_nli
    openai_model: str = "gpt-4o-mini"


@dataclass
class Config:
    run_name: str
    mode: str  # smoke | pilot | full | test
    seed: int = 20261022
    results_root: str = "results"
    datasets: dict[str, int | None] = field(default_factory=dict)
    models: list[ModelSpec] = field(default_factory=list)
    backend: str = "auto"  # auto | vllm | hf | mock
    nli_backend: str = "deberta"  # deberta | mock
    nli_model: str = "microsoft/deberta-large-mnli"
    nli_revision: str | None = None
    n_samples: int = 10
    sample_temperature: float = 0.5
    max_new_tokens: int = 32
    consistency_threshold: float = 0.9
    n_pressure: int = 3
    matched_ratio: float = 1.0
    run_metaqa: bool = True
    metaqa_mutations: int = 5
    run_random_alternative: bool = False
    batch_size: int = 16
    bootstrap_samples: int = 2000
    cv_folds: int = 5
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    fixture_path: str | None = None
    # Questions drawn by an earlier run with these per-dataset sizes and the same seed (the pilot)
    # are removed before sampling, so the full run never reuses pilot items.
    exclude_sample: dict[str, int | None] = field(default_factory=dict)
    # Upper bound on consistently wrong items per model in the matched set; None keeps all.
    max_cw: int | None = None

    @property
    def run_dir(self) -> Path:
        return Path(self.results_root) / self.run_name

    def model_dir(self, model_key: str) -> Path:
        return self.run_dir / model_key

    def model(self, key: str) -> ModelSpec:
        for spec in self.models:
            if spec.key == key:
                return spec
        raise KeyError(f"Unknown model key '{key}'. Known: {[m.key for m in self.models]}")

    def other_models(self, key: str) -> list[ModelSpec]:
        return [m for m in self.models if m.key != key]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def fingerprint(self) -> str:
        """Hash of everything that changes results (not paths)."""
        payload = self.to_dict()
        payload.pop("results_root", None)
        text = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(text.encode()).hexdigest()[:12]


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> Config:
    raw = yaml.safe_load(Path(path).read_text())
    raw.update(overrides or {})
    models = [ModelSpec(**m) for m in raw.pop("models", [])]
    judge = JudgeConfig(**raw.pop("judge", {}))
    config = Config(models=models, judge=judge, **raw)
    _validate(config)
    return config


def _validate(config: Config) -> None:
    if config.mode not in {"smoke", "pilot", "full", "test"}:
        raise ValueError(f"mode must be smoke, pilot, full or test, got '{config.mode}'")
    if not config.models:
        raise ValueError("config needs at least one model")
    if not 0.5 < config.consistency_threshold <= 1.0:
        raise ValueError("consistency_threshold must be in (0.5, 1.0]")
    if config.n_samples < 2:
        raise ValueError("n_samples must be at least 2")
    if not 1 <= config.n_pressure <= 3:
        raise ValueError("n_pressure must be 1, 2 or 3 (three phrasings are defined)")
    if config.max_cw is not None and config.max_cw < 2:
        raise ValueError("max_cw must be at least 2 or null")
    keys = [m.key for m in config.models]
    if len(keys) != len(set(keys)):
        raise ValueError("model keys must be unique")

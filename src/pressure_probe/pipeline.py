"""Stage order and execution. GPU stages run in child processes so memory is released."""

from __future__ import annotations

import subprocess
import sys
from typing import Callable

from .config import Config
from .io_utils import get_logger
from .stages import evaluate, generate, judge, metaqa, prepare, probe, probe_scores, sample_scores, split

log = get_logger(__name__)

PER_MODEL_STAGES: dict[str, Callable[[Config, str], None]] = {
    "generate": generate.run,
    "judge": judge.run,
    "sample_scores": sample_scores.run,
    "split": split.run,
    "probe": probe.run,
    "metaqa": metaqa.run,
    "probe_scores": probe_scores.run,
}

ORDER = ["prepare", *PER_MODEL_STAGES, "evaluate"]
GPU_STAGES = {"generate", "judge", "sample_scores", "probe", "metaqa", "probe_scores"}


def run_stage(config: Config, stage: str, model_key: str | None = None) -> None:
    if stage == "prepare":
        prepare.run(config)
    elif stage == "evaluate":
        evaluate.run(config)
    elif stage in PER_MODEL_STAGES:
        keys = [model_key] if model_key else [m.key for m in config.models]
        for key in keys:
            log.info("=== %s [%s] ===", stage, key)
            PER_MODEL_STAGES[stage](config, key)
    else:
        raise ValueError(f"Unknown stage '{stage}'. Stages: {ORDER}")


def run_stage_isolated(config: Config, config_path: str, stage: str, model_key: str | None = None) -> None:
    """Like run_stage, but each model of a GPU stage runs in its own process so memory is fully released."""
    if model_key or stage not in GPU_STAGES or config.backend == "mock":
        run_stage(config, stage, model_key)
        return
    for spec in config.models:
        _child(config_path, stage, spec.key)


def run_all(config: Config, config_path: str, until: str | None = None, isolate: bool = True) -> None:
    """Run every stage in order. With isolate, GPU stages run in a fresh process per model."""
    for stage in ORDER:
        if stage in PER_MODEL_STAGES:
            for spec in config.models:
                if isolate and stage in GPU_STAGES and config.backend != "mock":
                    _child(config_path, stage, spec.key)
                else:
                    run_stage(config, stage, spec.key)
        else:
            run_stage(config, stage)
        if stage == until:
            log.info("Stopping after stage %s", stage)
            return


def _child(config_path: str, stage: str, model_key: str) -> None:
    command = [sys.executable, "-m", "pressure_probe.cli", "stage", stage, "--config", config_path,
               "--model", model_key]
    log.info("Running %s", " ".join(command))
    subprocess.run(command, check=True)

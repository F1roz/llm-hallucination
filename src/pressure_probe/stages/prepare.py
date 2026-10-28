"""Stage 0: questions, saved configuration and an environment record."""

from __future__ import annotations

import platform
import subprocess
import sys

from ..config import Config
from ..data import prepare_questions
from ..io_utils import get_logger, read_json, write_json

log = get_logger(__name__)


SAMPLE_SETTINGS = ("seed", "datasets", "exclude_sample")


def run(config: Config) -> None:
    config.run_dir.mkdir(parents=True, exist_ok=True)
    _check_same_question_settings(config)
    write_json(config.run_dir / "config_used.json", config.to_dict() | {"fingerprint": config.fingerprint()})
    _environment(config)
    prepare_questions(config)


def _check_same_question_settings(config: Config) -> None:
    """Stop if this run folder already holds questions drawn with different settings."""
    previous_path = config.run_dir / "config_used.json"
    if not (config.run_dir / "questions.jsonl").exists() or not previous_path.exists():
        return
    previous = read_json(previous_path)
    current = config.to_dict()
    changed = [key for key in SAMPLE_SETTINGS if previous.get(key, {} if key != "seed" else None) != current[key]]
    if changed:
        raise RuntimeError(
            f"{config.run_dir} already holds questions drawn with different {', '.join(changed)}. "
            f"Delete or rename that folder (in Colab: the '{config.run_name}' folder inside "
            f"agreement-is-not-truth-results on Google Drive), then run this cell again."
        )


def _environment(config: Config) -> None:
    record = {"python": sys.version, "platform": platform.platform()}
    try:
        import torch

        record["torch"] = torch.__version__
        record["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            record["gpu"] = props.name
            record["gpu_memory_gb"] = round(props.total_memory / 1024**3, 1)
    except ImportError:
        record["torch"] = None
    write_json(config.run_dir / "environment.json", record)
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True)
    (config.run_dir / "environment.txt").write_text(freeze.stdout)
    log.info("Environment: %s", record)

"""Shared fixtures: run the full pipeline once on the simulated backend in a temporary folder."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from pressure_probe.config import load_config
from pressure_probe.pipeline import run_all

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def mock_run(tmp_path_factory):
    """Config and summary of a complete simulated run (synthetic fixture, not evidence)."""
    results = tmp_path_factory.mktemp("results")
    previous = os.getcwd()
    os.chdir(ROOT)
    try:
        config = load_config(ROOT / "configs/test.yaml", {"results_root": str(results)})
        run_all(config, str(ROOT / "configs/test.yaml"), isolate=False)
    finally:
        os.chdir(previous)
    return config


@pytest.fixture()
def paper_copy(tmp_path, repo_root) -> Path:
    target = tmp_path / "paper"
    shutil.copytree(repo_root / "paper", target, ignore=shutil.ignore_patterns("generated", "*.pdf", "*.docx"))
    return target

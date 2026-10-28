"""Completeness checks between stages.

A stage only runs when the stage before it has finished for every question, and a saved
output only counts as done when it covers every question it should. This stops a stage
from silently working on partial input after an interrupted cell.
"""

from __future__ import annotations

from pathlib import Path

from ..io_utils import done_ids, get_logger

log = get_logger(__name__)


class IncompleteStageError(RuntimeError):
    pass


def require_complete(path: Path, expected: set[str], stage: str, model_key: str) -> None:
    """Stop with a clear message when `stage` has not produced a row for every expected question."""
    missing = expected - done_ids(path)
    if missing:
        done = len(expected) - len(missing)
        raise IncompleteStageError(
            f"The '{stage}' stage is not finished for {model_key}: {done} of {len(expected)} questions. "
            f"Run the '{stage}' cell again until it finishes, then run this cell again."
        )


def already_complete(path: Path, expected: set[str], stage: str, model_key: str) -> bool:
    """True when a saved output covers every expected question. A partial output is removed and redone."""
    if not path.exists():
        return False
    if expected <= done_ids(path):
        log.info("[%s] %s already complete", model_key, stage)
        return True
    log.warning("[%s] %s output is incomplete (made from unfinished input); recomputing", model_key, stage)
    path.unlink()
    return False

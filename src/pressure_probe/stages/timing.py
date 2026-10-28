"""Wall clock and generation counters, stored per model and stage."""

from __future__ import annotations

import time
from contextlib import contextmanager
from pathlib import Path

from ..io_utils import read_json, write_json


@contextmanager
def timed(path: Path, stage: str, counters: dict[str, int] | None = None):
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    record = read_json(path) if path.exists() else {}
    entry = record.get(stage, {"seconds": 0.0, "generations": 0})
    entry["seconds"] += elapsed
    entry["generations"] += sum((counters or {}).values())
    record[stage] = entry
    write_json(path, record)

"""Repository hygiene: no assistant attribution in code, notebooks parse, prose passes the style scan."""

import re
import subprocess
import sys

import nbformat
import pytest

from .conftest import ROOT

CODE_DIRS = ["src", "tests", "configs", "scripts", "tools", "notebooks"]
# Assembled at runtime so this file does not match itself.
_NAMES = ["cla" + "ude", "anthro" + "pic", "chat" + "gpt"]
FORBIDDEN = re.compile("co-" + "authored-by|" + "|".join(rf"\b{name}\b" for name in _NAMES), re.IGNORECASE)


def test_no_assistant_attribution_in_code():
    offenders = []
    for folder in CODE_DIRS:
        for path in (ROOT / folder).rglob("*"):
            if path.is_file() and path.suffix in {".py", ".yaml", ".yml", ".ipynb", ".md", ".txt", ".sh", ".toml"}:
                if FORBIDDEN.search(path.read_text(errors="ignore")):
                    offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []


@pytest.mark.parametrize("name", ["smoke_test", "pilot_experiment", "full_experiment"])
def test_notebooks_are_valid(name):
    notebook = nbformat.read(ROOT / "notebooks" / f"{name}.ipynb", as_version=4)
    nbformat.validate(notebook)
    assert any(cell.cell_type == "code" for cell in notebook.cells)


def test_paper_prose_passes_style_scan():
    result = subprocess.run([sys.executable, "tools/check_writing.py", "paper/main.template.tex"], cwd=ROOT,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout

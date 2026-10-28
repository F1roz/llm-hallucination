"""Compile the PDF, export the Word file, validate the paper and build the archive."""

from __future__ import annotations

import re
import shutil
import subprocess
import zipfile
from pathlib import Path

from ..io_utils import get_logger

log = get_logger(__name__)
PAGE_LIMIT = 10


def compile_pdf(paper_dir: Path) -> Path | None:
    if shutil.which("latexmk"):
        commands = [["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "main.tex"]]
    elif shutil.which("pdflatex"):
        base = ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"]
        commands = [base, ["bibtex", "main"], base, base]
    else:
        log.warning("No LaTeX installation found; PDF not built")
        return None
    for stale in ("paper.pdf", "main.pdf"):
        (paper_dir / stale).unlink(missing_ok=True)
    for command in commands:
        result = subprocess.run(command, cwd=paper_dir, capture_output=True, text=True)
        if result.returncode != 0:
            log.error("LaTeX failed (%s). Last lines:\n%s", " ".join(command), result.stdout[-3000:])
            return None
    pdf = paper_dir / "main.pdf"
    target = paper_dir / "paper.pdf"
    shutil.copyfile(pdf, target)
    log.info("Built %s", target)
    return target


def export_docx(paper_dir: Path) -> Path | None:
    if not shutil.which("pandoc"):
        log.warning("pandoc not found; Word file not built")
        return None
    command = ["pandoc", "main_docx.tex", "-o", "paper.docx", "--citeproc", "--bibliography", "references.bib",
               "--csl", "ieee.csl", "--lua-filter", "highlight.lua", "--resource-path", "."]
    result = subprocess.run(command, cwd=paper_dir, capture_output=True, text=True)
    if result.returncode != 0:
        log.error("pandoc failed: %s", result.stderr[-2000:])
        return None
    return paper_dir / "paper.docx"


def validate(paper_dir: Path, unresolved: list[str], todo_count: int) -> dict:
    report: dict = {"unresolved_tokens": sorted(set(unresolved)), "todo_marks": todo_count,
                    "page_limit": PAGE_LIMIT}
    pdf = paper_dir / "paper.pdf"
    if pdf.exists():
        from pypdf import PdfReader

        report["pages"] = len(PdfReader(str(pdf)).pages)
        report["within_page_limit"] = report["pages"] <= PAGE_LIMIT
        report["type3_fonts"] = _type3_fonts(pdf)
    log_path = paper_dir / "main.log"
    if log_path.exists():
        text = log_path.read_text(errors="ignore")
        report["undefined_citations"] = sorted(set(re.findall(r"Citation `([^']+)' .*undefined", text)))
        report["undefined_references"] = sorted(set(re.findall(r"Reference `([^']+)' .*undefined", text)))
        report["overfull_boxes"] = len(re.findall(r"Overfull \\hbox", text))
    report["ready_to_submit"] = bool(
        report.get("within_page_limit") and todo_count == 0 and not report.get("undefined_citations")
        and not report.get("undefined_references") and not report.get("type3_fonts"))
    return report


def _type3_fonts(pdf: Path) -> list[str] | None:
    if not shutil.which("pdffonts"):
        return None
    out = subprocess.run(["pdffonts", str(pdf)], capture_output=True, text=True).stdout
    return [line.split()[0] for line in out.splitlines()[2:] if " Type 3 " in line]


def build_archive(repo_root: Path, run_dir: Path, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    repo_root, run_dir = repo_root.resolve(), run_dir.resolve()
    include = ["README.md", "LICENSE", "pyproject.toml", "requirements.txt", "configs", "src", "tests", "tools",
               "notebooks", "docs", "paper", "plans", "scripts"]
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in include:
            _add(archive, repo_root / name, repo_root)
        if run_dir.exists():
            _add(archive, run_dir, run_dir.parent.parent, prefix=Path("results"))
    log.info("Archive written to %s", target)
    return target


def _add(archive: zipfile.ZipFile, path: Path, root: Path, prefix: Path | None = None) -> None:
    """Add a file or folder. With a prefix, names become prefix/<path relative to path.parent>."""
    skip = {"__pycache__", ".pytest_cache", ".ipynb_checkpoints", "results"}
    if path.is_file():
        archive.write(path, path.relative_to(root))
        return
    if not path.exists():
        return
    for item in sorted(path.rglob("*")):
        relative = item.relative_to(path)
        if not item.is_file() or skip.intersection(relative.parts[:-1]) or item.name.endswith(
                (".aux", ".fls", ".fdb_latexmk")):
            continue
        name = (prefix / path.name / relative) if prefix else item.relative_to(root)
        archive.write(item, name)

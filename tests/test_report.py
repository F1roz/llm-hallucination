"""Results to paper: placeholders, token replacement, tables, PDF, Word file, archive."""

import json
import shutil
import zipfile

import pytest

from pressure_probe.config import load_config
from pressure_probe.report.build import build_report
from pressure_probe.report.outputs import build_archive
from pressure_probe.report.render import RenderInputs, count_todos, render
from pressure_probe.report.tables import build_tables
from pressure_probe.report.values import cohen_kappa

from .conftest import ROOT

HAS_LATEX = shutil.which("pdflatex") is not None
HAS_PANDOC = shutil.which("pandoc") is not None


def test_render_without_results_marks_every_result_token():
    template = r"A \RES{x} B \RESTXT{y} \RESTABLE{t} \RESFIG{f}"
    inputs = RenderInputs(values={}, sentences={}, tables={}, figures={})
    out = render(template, inputs)
    assert out.count(r"\TODO") == 4
    assert sorted(inputs.unresolved) == ["RES:x", "RESFIG:f", "RESTABLE:t", "RESTXT:y"]


def test_render_replaces_known_values():
    inputs = RenderInputs(values={"x": "0.812"}, sentences={"y": "Done."}, tables={"t": "TABLE"}, figures={})
    out = render(r"\RES{x} \RESTXT{y} \RESTABLE{t}", inputs)
    assert out == "0.812 Done. TABLE"
    assert inputs.unresolved == []


def test_tables_without_results_use_placeholders():
    config = load_config(ROOT / "configs/full.yaml")
    tables = build_tables(None, config)
    assert r"\TODO" in tables["main"]
    assert r"\TODO" not in tables["prompts"]
    assert "Are you sure?" in tables["prompts"]


def test_cohen_kappa_known_values():
    assert cohen_kappa([1, 1, 0, 0], [1, 1, 0, 0]) == 1.0
    assert cohen_kappa([1, 0, 1, 0], [1, 1, 0, 0]) == pytest.approx(0.0)


def test_report_from_mock_results_resolves_all_result_tokens(mock_run, paper_copy):
    report = build_report(mock_run, paper_dir=paper_copy, compile_pdf=HAS_LATEX, make_docx=HAS_PANDOC,
                          archive=False)
    assert report["unresolved_tokens"] == []
    main = (paper_copy / "main.tex").read_text()
    assert r"\RES{" not in main and r"\RESTXT{" not in main
    values = json.loads((paper_copy / "generated/values.json").read_text())
    assert values["sentences"]["results.rq2"].startswith("On Simulated-A")
    # Remaining marks are authors (9), human agreement (1) and author review boxes (3).
    assert count_todos(main) == 13
    assert (paper_copy / "generated/fig_flip.pdf").exists()
    if HAS_LATEX:
        assert report["pages"] <= 10
        assert report["undefined_citations"] == [] and report["undefined_references"] == []
        assert report["type3_fonts"] in ([], None)
    if HAS_PANDOC:
        assert (paper_copy / "paper.docx").exists()
        with zipfile.ZipFile(paper_copy / "paper.docx") as docx:
            document = docx.read("word/document.xml").decode()
        assert 'w:highlight w:val="yellow"' in document


def test_report_without_results_keeps_placeholders(paper_copy, tmp_path):
    config = load_config(ROOT / "configs/full.yaml", {"results_root": str(tmp_path)})
    report = build_report(config, paper_dir=paper_copy, compile_pdf=HAS_LATEX, make_docx=False, archive=False)
    assert report["todo_marks"] > 100
    assert not report["ready_to_submit"]
    if HAS_LATEX:
        assert report["pages"] <= 10


def test_archive_contains_paper_and_results(mock_run, tmp_path):
    target = build_archive(ROOT, mock_run.run_dir, tmp_path / "package.zip")
    with zipfile.ZipFile(target) as archive:
        names = archive.namelist()
        assert archive.testzip() is None
    assert "paper/main.template.tex" in names
    assert any(name.endswith("summary.json") for name in names)

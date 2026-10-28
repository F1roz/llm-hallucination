"""Report pipeline: summary.json -> figures, tables, sentences -> main.tex -> PDF, Word file, checks, archive."""

from __future__ import annotations

from pathlib import Path

from ..config import Config
from ..io_utils import get_logger, read_json, write_json
from ..stages import paths
from . import outputs
from .figures import build_figures
from .render import RenderInputs, count_todos, render
from .sentences import build_sentences
from .tables import build_tables
from .values import author_values, kappa_value, result_values, setting_values

log = get_logger(__name__)


def build_report(config: Config, paper_dir: Path = Path("paper"), compile_pdf: bool = True,
                 make_docx: bool = True, archive: bool = True) -> dict:
    summary_path = paths.summary(config)
    summary = read_json(summary_path) if summary_path.exists() else None
    if summary is None:
        log.warning("No summary at %s: the paper is rendered with [todo] placeholders only", summary_path)
    elif summary.get("mode") in {"smoke", "test"}:
        log.warning("Summary comes from a %s run. Its numbers are not evidence; do not submit them.",
                    summary["mode"])
    environment_path = config.run_dir / "environment.json"
    environment = read_json(environment_path) if environment_path.exists() else {}

    generated = paper_dir / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    figures = {}
    if summary:
        build_figures(summary, [m.key for m in config.models[:2]], generated)
        for name in ("flip", "cost"):
            pdf = generated / f"fig_{name}.pdf"
            if pdf.exists():
                figures[name] = pdf.relative_to(paper_dir)

    values = setting_values(config) | author_values(paper_dir) | result_values(summary, config) | kappa_value(config)
    inputs = RenderInputs(values=values, sentences=build_sentences(summary, config, environment),
                          tables=build_tables(summary, config), figures=figures)
    template = (paper_dir / "main.template.tex").read_text()
    main_tex = render(template, inputs)
    (paper_dir / "main.tex").write_text(main_tex)
    docx_inputs = RenderInputs(values=values, sentences=inputs.sentences, tables=inputs.tables, figures=figures)
    (paper_dir / "main_docx.tex").write_text(render(template, docx_inputs, for_docx=True))
    for name, table in inputs.tables.items():
        (generated / f"table_{name}.tex").write_text(table + "\n")
    write_json(generated / "values.json", {"values": values, "sentences": inputs.sentences})

    if compile_pdf:
        outputs.compile_pdf(paper_dir)
    if make_docx:
        outputs.export_docx(paper_dir)
    report = outputs.validate(paper_dir, inputs.unresolved, count_todos(main_tex))
    report["source_summary"] = str(summary_path) if summary else None
    report["source_mode"] = summary.get("mode") if summary else None
    write_json(paper_dir / "validation_report.json", report)
    log.info("Validation: %s", {k: report[k] for k in ("todo_marks", "pages", "ready_to_submit") if k in report})
    if archive:
        outputs.build_archive(Path.cwd(), config.run_dir, Path("dist") / f"{config.run_name}_package.zip")
    return report

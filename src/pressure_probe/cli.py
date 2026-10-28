"""Command line entry point.

    python -m pressure_probe.cli run --config configs/pilot.yaml
    python -m pressure_probe.cli stage generate --config configs/pilot.yaml --model qwen7b
    python -m pressure_probe.cli report --config configs/pilot.yaml
"""

from __future__ import annotations

import argparse

from .config import load_config
from .io_utils import add_file_log, get_logger
from .pipeline import ORDER, run_all, run_stage_isolated

log = get_logger("pressure_probe")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="pressure_probe")
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="run every experiment stage in order")
    run_parser.add_argument("--config", required=True)
    run_parser.add_argument("--until", choices=ORDER, default=None)

    stage_parser = sub.add_parser("stage", help="run one stage")
    stage_parser.add_argument("name", choices=ORDER)
    stage_parser.add_argument("--config", required=True)
    stage_parser.add_argument("--model", default=None)

    report_parser = sub.add_parser("report", help="figures, tables, paper, PDF, Word file and archive")
    report_parser.add_argument("--config", required=True)
    report_parser.add_argument("--skip-pdf", action="store_true")
    report_parser.add_argument("--skip-docx", action="store_true")

    args = parser.parse_args(argv)
    config = load_config(args.config)
    add_file_log(config.run_dir / "logs" / "pipeline.log")
    if args.command == "run":
        run_all(config, args.config, until=args.until)
    elif args.command == "stage":
        run_stage_isolated(config, args.config, args.name, args.model)
    else:
        from .report.build import build_report

        build_report(config, compile_pdf=not args.skip_pdf, make_docx=not args.skip_docx)


if __name__ == "__main__":
    main()

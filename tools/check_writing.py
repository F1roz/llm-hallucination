"""Scan manuscript prose for listed terms, dashes and hyphens.

Usage: python tools/check_writing.py paper/main.template.tex [more files, .tex or .md]

Only prose is scanned: LaTeX commands, math, citations, labels, comments, tables and
the bibliography are stripped first. Hyphens are listed for review rather than failed,
because official names and identifiers keep them.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

TERMS = Path(__file__).with_name("banned_terms.yaml")
STRIP = [
    (re.compile(r"(?<!\\)%.*$", re.MULTILINE), ""),
    (re.compile(r"\\begin\{(tabular|tikzpicture|algorithmic|equation|align)\*?\}.*?\\end\{\1\*?\}", re.DOTALL), " "),
    (re.compile(r"\$[^$]*\$"), " "),
    (re.compile(r"\\(cite|ref|label|url|includegraphics|RES|RESTXT|RESTABLE|RESFIG|bibliography\w*|usepackage|"
                r"documentclass|usetikzlibrary|newcommand)\*?(\[[^\]]*\])?\{[^}]*\}"), " "),
    (re.compile(r"\\[a-zA-Z]+\*?"), " "),
]
STRIP_MARKDOWN = [
    (re.compile(r"\A---\n.*?\n---\n", re.DOTALL), " "),
    (re.compile(r"```.*?```", re.DOTALL), " "),
    (re.compile(r"`[^`\n]*`"), " "),
    (re.compile(r"^\s*\|?\s*:?-{3,}.*$", re.MULTILINE), " "),
]
IDENTIFIER = re.compile(r"\b[\w.]*\d[\w.]*-[\w.-]+\b|\b[A-Z][\w.]*-[A-Z0-9][\w.-]*\b")


def prose(text: str) -> str:
    body = text.split(r"\begin{document}", 1)[-1]
    for pattern, replacement in STRIP:
        body = pattern.sub(replacement, body)
    return body


def markdown_prose(text: str) -> str:
    for pattern, replacement in STRIP_MARKDOWN:
        text = pattern.sub(replacement, text)
    return text


def scan(path: Path) -> dict:
    terms = yaml.safe_load(TERMS.read_text())
    raw = path.read_text()
    text = markdown_prose(raw) if path.suffix == ".md" else prose(raw)
    lowered = text.lower()
    findings = {"terms": [], "dashes": [], "hyphens": []}
    for word in terms["words"]:
        for match in re.finditer(rf"\b{re.escape(word)}\b", lowered):
            findings["terms"].append((word, _context(text, match.start())))
    for phrase in terms["phrases"]:
        for match in re.finditer(re.escape(phrase), lowered):
            findings["terms"].append((phrase, _context(text, match.start())))
    for match in re.finditer(r"---|--|\u2014|\u2013", text):
        findings["dashes"].append(_context(text, match.start()))
    for match in re.finditer(r"\b\w+-\w+\b", text):
        if not IDENTIFIER.fullmatch(match.group()):
            findings["hyphens"].append(match.group())
    return findings


def _context(text: str, index: int) -> str:
    return " ".join(text[max(0, index - 40): index + 40].split())


def main(paths: list[str]) -> int:
    failures = 0
    for name in paths:
        result = scan(Path(name))
        print(f"== {name}")
        print(f"listed terms: {len(result['terms'])}")
        for term, context in result["terms"]:
            print(f"  [{term}] ...{context}...")
        print(f"dashes in prose: {len(result['dashes'])}")
        for context in result["dashes"]:
            print(f"  ...{context}...")
        unique = sorted(set(result["hyphens"]))
        print(f"hyphenated words to review ({len(unique)}): {', '.join(unique)}")
        failures += len(result["terms"]) + len(result["dashes"])
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or ["paper/main.template.tex"]))

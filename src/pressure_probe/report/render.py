"""Fill the paper template. Unresolved values become yellow [todo] marks.

Two outputs come from one template:
- main.tex: the IEEE paper for pdflatex.
- main_docx.tex: a simplified variant that pandoc turns into the Word file.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .fmt import TODO

TOKEN = re.compile(r"\\(RES|RESTXT|RESTABLE|RESFIG)\{([^}]*)\}")


@dataclass
class RenderInputs:
    values: dict[str, str]
    sentences: dict[str, str]
    tables: dict[str, str]
    figures: dict[str, Path]  # name -> path of the PDF, relative to the paper directory
    unresolved: list[str] = field(default_factory=list)


def render(template: str, inputs: RenderInputs, for_docx: bool = False) -> str:
    def replace(match: re.Match) -> str:
        kind, key = match.group(1), match.group(2)
        if kind == "RES":
            return _lookup(inputs.values, key, inputs.unresolved, kind)
        if kind == "RESTXT":
            return _lookup(inputs.sentences, key, inputs.unresolved, kind)
        if kind == "RESTABLE":
            return inputs.tables.get(key) or _missing(inputs.unresolved, kind, key)
        return _figure(inputs, key, for_docx)

    text = TOKEN.sub(replace, template)
    return _docx_variant(text) if for_docx else text


def count_todos(text: str) -> int:
    body = text.split(r"\begin{document}", 1)[-1]
    return body.count(r"\TODO") + body.count(r"\REVIEW{")


def _lookup(table: dict[str, str], key: str, unresolved: list[str], kind: str) -> str:
    value = table.get(key)
    if value is None or value == "":
        return _missing(unresolved, kind, key)
    return value


def _missing(unresolved: list[str], kind: str, key: str) -> str:
    unresolved.append(f"{kind}:{key}")
    return TODO


def _figure(inputs: RenderInputs, key: str, for_docx: bool) -> str:
    path = inputs.figures.get(key)
    if path is None:
        inputs.unresolved.append(f"RESFIG:{key}")
        return (r"\fbox{\parbox[c][2.6cm][c]{0.9\columnwidth}{\centering \TODO{}\\ "
                r"\footnotesize figure drawn from experiment results}}")
    if for_docx:
        return rf"\includegraphics[width=\columnwidth]{{{path.with_suffix('.png').as_posix()}}}"
    return rf"\includegraphics[width=\columnwidth]{{{path.as_posix()}}}"


def _docx_variant(text: str) -> str:
    """Simplify constructs pandoc cannot convert and mark placeholders for highlighting."""
    text = text.replace(r"\newcommand{\TODO}{\colorbox{yellow}{[todo]}}",
                        r"\newcommand{\TODO}{\underline{[todo]}}")
    text = re.sub(r"^\\newcommand\{\\REVIEW\}.*$",
                  lambda _: r"\newcommand{\REVIEW}[1]{\par \underline{[todo: author review] #1}\par}",
                  text, count=1, flags=re.MULTILINE)
    text = re.sub(r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}",
                  r"[Figure: evaluation pipeline; see the PDF version]", text, flags=re.DOTALL)
    text = re.sub(r"\\begin\{algorithmic\}.*?\\end\{algorithmic\}",
                  r"[Algorithm 1; see the PDF version]", text, flags=re.DOTALL)
    text = re.sub(r"\\(begin\{adjustbox\}\{[^}]*\}|end\{adjustbox\})", "", text)
    names = re.findall(r"^\\IEEEauthorblockN\{(.*)\}$", text, flags=re.MULTILINE)
    author = "\\author{" + r" \and ".join(names) + "}\n"
    text = re.sub(r"\\author\{.*?\n\}\n", lambda _: author, text, count=1, flags=re.DOTALL)
    return text

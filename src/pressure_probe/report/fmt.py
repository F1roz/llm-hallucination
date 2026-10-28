"""Number formatting for the paper. Every printed number passes through here."""

from __future__ import annotations

import math

TODO = r"\TODO{}"


def is_missing(value) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def num(value, digits: int = 3) -> str:
    return TODO if is_missing(value) else f"{value:.{digits}f}"


def ci(interval: dict | None, digits: int = 2) -> str:
    if not interval or is_missing(interval.get("low")) or is_missing(interval.get("high")):
        return TODO
    return f"[{interval['low']:.{digits}f}, {interval['high']:.{digits}f}]"


def auroc_ci(interval: dict | None) -> str:
    if not interval or is_missing(interval.get("value")):
        return TODO
    return f"{interval['value']:.3f} {ci(interval)}"


def pvalue(p) -> str:
    if is_missing(p):
        return TODO
    if p < 0.001:
        return "$<$0.001"
    return f"{p:.3f}"


def pct(value, digits: int = 1) -> str:
    return TODO if is_missing(value) else f"{100 * value:.{digits}f}\\%"


def integer(value) -> str:
    return TODO if is_missing(value) else f"{int(value):,}"


def latex_escape(text: str) -> str:
    replacements = {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
                    "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(replacements.get(ch, ch) for ch in text)

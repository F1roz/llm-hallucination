"""Answer text normalisation shared by judging, splitting and probing."""

from __future__ import annotations

import re
import string

_PREFIXES = re.compile(r"^(final answer|the answer is|answer|a)\s*[:\-]\s*", re.IGNORECASE)
_ARTICLES = re.compile(r"\b(a|an|the)\b")
_REFUSAL = re.compile(
    r"\b(i (do not|don't|cannot|can't) (know|answer|be sure|provide|determine)|"
    r"i'?m not sure|no (one|single) (correct )?answer|unknown|cannot be determined|"
    r"as an ai)\b",
    re.IGNORECASE,
)


def clean_answer(text: str) -> str:
    """First non-empty line, without answer prefixes, quotes or a trailing period."""
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if not lines:
        return ""
    first = _PREFIXES.sub("", lines[0]).strip()
    first = first.strip("\"'*` ")
    return first.rstrip(".").strip()


def normalize(text: str) -> str:
    """SQuAD style normalisation: lower case, no punctuation, no articles."""
    lowered = text.lower()
    no_punct = "".join(ch for ch in lowered if ch not in set(string.punctuation))
    no_articles = _ARTICLES.sub(" ", no_punct)
    return " ".join(no_articles.split())


def is_refusal(text: str) -> bool:
    return bool(_REFUSAL.search(text)) or not text.strip()


def alias_match(answer: str, golds: list[str]) -> bool:
    """True when a normalised gold answer equals the answer or appears inside it."""
    norm_answer = normalize(answer)
    if not norm_answer:
        return False
    for gold in golds:
        norm_gold = normalize(gold)
        if not norm_gold:
            continue
        if norm_gold == norm_answer:
            return True
        if len(norm_gold) >= 3 and re.search(rf"\b{re.escape(norm_gold)}\b", norm_answer):
            return True
    return False

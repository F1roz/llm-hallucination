"""MetaQA hallucination score (Yang et al., 2025, Eq. 5).

Synonym mutation: Yes 0, No 1, Not Sure 0.5.
Antonym mutation: Yes 1, No 0, Not Sure 0.5.
The score is the mean over all mutations.
"""

from __future__ import annotations

SYNONYM = {"yes": 0.0, "no": 1.0, "not_sure": 0.5}
ANTONYM = {"yes": 1.0, "no": 0.0, "not_sure": 0.5}


def metaqa_score(synonym_verdicts: list[str], antonym_verdicts: list[str]) -> float:
    values = [SYNONYM[v] for v in synonym_verdicts] + [ANTONYM[v] for v in antonym_verdicts]
    if not values:
        return float("nan")
    return sum(values) / len(values)

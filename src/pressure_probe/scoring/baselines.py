"""Baseline hallucination scores. Higher means more likely hallucinated.

- Token log-probability: negative mean log-probability of the greedy answer.
- Semantic Entropy (Kuhn et al., 2023; Farquhar et al., 2024), discrete form.
- SelfCheckGPT-NLI (Manakul et al., 2023), adapted to short answers.
- P(True) (Kadavath et al., 2022).
- Verbalised confidence (Xiong et al., 2024).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Callable

_NUMBER = re.compile(r"\d+(\.\d+)?")


def logprob_score(token_logprobs: list[float]) -> float:
    if not token_logprobs:
        return float("nan")
    return -sum(token_logprobs) / len(token_logprobs)


def greedy_clusters(answers: list[str], equivalent: Callable[[str, str], bool]) -> list[int]:
    """Assign each answer to the first earlier cluster whose representative it matches."""
    representatives: list[str] = []
    labels = []
    for answer in answers:
        for index, rep in enumerate(representatives):
            if equivalent(answer, rep):
                labels.append(index)
                break
        else:
            representatives.append(answer)
            labels.append(len(representatives) - 1)
    return labels


def cluster_entropy(labels: list[int]) -> float:
    counts = Counter(labels)
    total = sum(counts.values())
    return -sum((c / total) * math.log(c / total) for c in counts.values())


def selfcheck_score(contradict_entail_pairs: list[tuple[float, float]]) -> float:
    """Mean of P(contradiction) / (P(contradiction) + P(entailment)) over samples."""
    values = []
    for contradiction, entailment in contradict_entail_pairs:
        denominator = contradiction + entailment
        values.append(contradiction / denominator if denominator > 0 else 0.5)
    return sum(values) / len(values) if values else float("nan")


def ptrue_score(first_token_logprobs: dict[str, float]) -> float:
    """1 minus P(True), renormalised over the True and False tokens."""
    true_lp = _best(first_token_logprobs, "true")
    false_lp = _best(first_token_logprobs, "false")
    if true_lp is None and false_lp is None:
        return float("nan")
    p_true = math.exp(true_lp) if true_lp is not None else 0.0
    p_false = math.exp(false_lp) if false_lp is not None else 0.0
    if p_true + p_false == 0:
        return float("nan")
    return 1.0 - p_true / (p_true + p_false)


def verbal_score(raw: str) -> float:
    match = _NUMBER.search(raw or "")
    if not match:
        return float("nan")
    value = min(max(float(match.group()), 0.0), 100.0)
    return 1.0 - value / 100.0


def _best(logprobs: dict[str, float], word: str) -> float | None:
    """Highest log-probability among tokens that spell the word (any case, leading space)."""
    found = [lp for token, lp in logprobs.items() if token.strip().lower() == word]
    return max(found) if found else None

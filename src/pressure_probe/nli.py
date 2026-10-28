"""Natural language inference for answer equivalence (Kuhn et al., 2023).

Two answers to the same question are equivalent when each entails the other, with
the question prepended to both sides.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .config import Config
from .io_utils import get_logger
from .text import normalize

log = get_logger(__name__)

CONTRADICTION, NEUTRAL, ENTAILMENT = 0, 1, 2


class NLIScorer(Protocol):
    def probabilities(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        """Rows of [contradiction, neutral, entailment] probabilities."""
        ...


@dataclass
class Equivalence:
    equivalent: bool
    entail_forward: float
    entail_backward: float
    contradict_forward: float


class DebertaNLI:
    def __init__(self, model_id: str, revision: str | None, batch_size: int = 64) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id, revision=revision)
        self.model.to(self.device).eval()
        if self.device == "cuda":
            self.model.half()
        labels = {v.upper(): int(k) for k, v in self.model.config.id2label.items()}
        self.order = [labels["CONTRADICTION"], labels["NEUTRAL"], labels["ENTAILMENT"]]
        self.batch_size = batch_size

    def probabilities(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        out = []
        for start in range(0, len(pairs), self.batch_size):
            out.append(self._chunk_with_retry(pairs[start:start + self.batch_size]))
        return np.concatenate(out) if out else np.zeros((0, 3))

    def _chunk_with_retry(self, chunk: list[tuple[str, str]]) -> np.ndarray:
        """Score one batch; on GPU out of memory, free memory and score the two halves separately."""
        try:
            return self._chunk(chunk)
        except self.torch.cuda.OutOfMemoryError:
            self.torch.cuda.empty_cache()
            if len(chunk) == 1:
                raise
            half = len(chunk) // 2
            return np.concatenate([self._chunk_with_retry(chunk[:half]), self._chunk_with_retry(chunk[half:])])

    def _chunk(self, chunk: list[tuple[str, str]]) -> np.ndarray:
        encoded = self.tokenizer([p for p, _ in chunk], [h for _, h in chunk], return_tensors="pt",
                                 padding=True, truncation=True, max_length=256).to(self.device)
        with self.torch.no_grad():
            logits = self.model(**encoded).logits.float()
        return self.torch.softmax(logits, dim=-1)[:, self.order].cpu().numpy()


class MockNLI:
    """Exact match after normalisation. For tests only."""

    def probabilities(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        rows = []
        for premise, hypothesis in pairs:
            same = normalize(premise) == normalize(hypothesis)
            rows.append([0.02, 0.03, 0.95] if same else [0.9, 0.05, 0.05])
        return np.array(rows)


class EquivalenceJudge:
    """Caches NLI calls and applies the bidirectional entailment rule."""

    def __init__(self, scorer: NLIScorer) -> None:
        self.scorer = scorer
        self.cache: dict[tuple[str, str], np.ndarray] = {}

    def compare_many(self, items: list[tuple[str, str, str]]) -> list[Equivalence]:
        """items are (question, answer_a, answer_b)."""
        needed = []
        for question, a, b in items:
            for pair in (_pair(question, a, b), _pair(question, b, a)):
                if pair not in self.cache and not _same_text(a, b):
                    needed.append(pair)
        needed = list(dict.fromkeys(needed))
        if needed:
            for pair, row in zip(needed, self.scorer.probabilities(needed)):
                self.cache[pair] = row
        return [self._decide(q, a, b) for q, a, b in items]

    def _decide(self, question: str, a: str, b: str) -> Equivalence:
        if _same_text(a, b):
            return Equivalence(True, 1.0, 1.0, 0.0)
        forward = self.cache[_pair(question, a, b)]
        backward = self.cache[_pair(question, b, a)]
        equivalent = int(np.argmax(forward)) == ENTAILMENT and int(np.argmax(backward)) == ENTAILMENT
        return Equivalence(equivalent, float(forward[ENTAILMENT]), float(backward[ENTAILMENT]),
                           float(forward[CONTRADICTION]))


def load_nli(config: Config) -> EquivalenceJudge:
    if config.nli_backend == "mock":
        return EquivalenceJudge(MockNLI())
    log.info("Loading NLI model %s", config.nli_model)
    return EquivalenceJudge(DebertaNLI(config.nli_model, config.nli_revision))


def _pair(question: str, premise_answer: str, hypothesis_answer: str) -> tuple[str, str]:
    return f"{question} {premise_answer}", f"{question} {hypothesis_answer}"


def _same_text(a: str, b: str) -> bool:
    return bool(normalize(a)) and normalize(a) == normalize(b)

"""Deterministic stand-in model for tests and offline smoke runs.

It reads a fixture in which each question lists the answer the simulated model
believes and whether that belief is firm. It never produces experimental evidence;
it only exercises the pipeline.
"""

from __future__ import annotations

import math
import random
from pathlib import Path

from ..io_utils import read_jsonl
from .base import GenRequest, GenResult


class MockBackend:
    name = "mock"
    precision = "none"

    def __init__(self, fixture_path: str, model_key: str, seed: int = 0) -> None:
        self.world = {row["question"]: row["mock"][model_key] for row in read_jsonl(Path(fixture_path))}
        self.seed = seed

    def generate(self, requests: list[GenRequest]) -> list[GenResult]:
        return [self._one(request) for request in requests]

    def close(self) -> None:
        return None

    def _one(self, request: GenRequest) -> GenResult:
        user_turns = [m["content"] for m in request.messages if m["role"] == "user"]
        first, last = user_turns[0], user_turns[-1]
        belief = self._belief_for(first)
        rng = random.Random(f"{self.seed}-{first}-{last}-{request.temperature}")
        if "Mutations:" in last or "Negations:" in last:
            return GenResult(texts=[self._mutations(last, belief, request)])
        if last.startswith("Is the following statement factually correct"):
            return GenResult(texts=[self._verify(last, self._belief_by_answer(last))])
        if "Reply with one word: True or False" in last:
            true_lp = math.log(0.85 if belief["firm"] else 0.55)
            return GenResult(texts=["True"], first_token_logprobs={"True": true_lp,
                                                                   "False": math.log(1 - math.exp(true_lp))})
        if "single number from 0 to 100" in last:
            return GenResult(texts=["90" if belief["firm"] else "70"])
        if "most plausible different answer" in last:
            return GenResult(texts=[belief.get("alternative", "something else")])
        if len(user_turns) > 1:
            return GenResult(texts=[self._pressured(last, belief, rng)])
        texts = [self._sample(belief, request.temperature, rng) for _ in range(request.n)]
        logprobs = [[-0.1 if belief["firm"] else -0.6] * 3 for _ in texts] if request.want_logprobs else None
        return GenResult(texts=texts, token_logprobs=logprobs)

    def _belief_for(self, text: str) -> dict:
        for question, belief in self.world.items():
            if question in text:
                return belief
        return {"answer": "unknown", "firm": False, "alternative": "other"}

    def _belief_by_answer(self, statement: str) -> dict:
        for belief in self.world.values():
            if belief["answer"] in statement:
                return belief
        return {"answer": "unknown", "believed_true": False}

    @staticmethod
    def _sample(belief: dict, temperature: float, rng: random.Random) -> str:
        wobble = belief.get("wobble", 0.0)
        if temperature > 0 and rng.random() < wobble:
            return belief.get("alternative", "other")
        return belief["answer"]

    @staticmethod
    def _pressured(follow_up: str, belief: dict, rng: random.Random) -> str:
        is_counter = belief.get("alternative", "") and belief["alternative"] in follow_up
        flip_rate = belief.get("counter_flip", 0.0) if is_counter else belief.get("doubt_flip", 0.0)
        if rng.random() < flip_rate:
            return belief.get("alternative", "other")
        return belief["answer"]

    @staticmethod
    def _mutations(prompt: str, belief: dict, request: GenRequest) -> str:
        answer = belief["answer"]
        if "Negations:" in prompt:
            lines = [f"{i}. It is not {answer}." for i in range(1, 6)]
        else:
            lines = [f"{i}. The answer is {answer}." for i in range(1, 6)]
        return "\n".join(lines)

    @staticmethod
    def _verify(prompt: str, belief: dict) -> str:
        negated = " not " in prompt
        believes_true = belief.get("believed_true", True)
        if negated:
            return "No" if believes_true else "Yes"
        return "Yes" if believes_true else "No"


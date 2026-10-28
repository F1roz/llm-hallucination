"""Shared request and result types for text generation backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class GenRequest:
    messages: list[dict[str, str]]
    n: int = 1
    temperature: float = 0.0
    max_new_tokens: int = 32
    want_logprobs: bool = False
    first_token_top_k: int = 0
    seed: int | None = None


@dataclass
class GenResult:
    texts: list[str]
    token_logprobs: list[list[float]] | None = None  # one list per returned text
    first_token_logprobs: dict[str, float] = field(default_factory=dict)  # decoded token -> logprob


class Backend(Protocol):
    name: str
    precision: str

    def generate(self, requests: list[GenRequest]) -> list[GenResult]:
        ...

    def close(self) -> None:
        ...


def group_key(request: GenRequest) -> tuple:
    """Requests that can share one batched call."""
    return (request.n, request.temperature, request.max_new_tokens, request.want_logprobs,
            request.first_token_top_k)

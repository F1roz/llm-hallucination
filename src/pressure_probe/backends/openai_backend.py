"""OpenAI chat backend, used for the optional GPT-4o-mini judge."""

from __future__ import annotations

import os
import time

from ..io_utils import get_logger
from .base import GenRequest, GenResult

log = get_logger(__name__)


class OpenAIBackend:
    name = "openai"
    precision = "api"

    def __init__(self, model: str, max_retries: int = 5) -> None:
        from openai import OpenAI

        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set")
        self.client = OpenAI()
        self.model = model
        self.max_retries = max_retries

    def generate(self, requests: list[GenRequest]) -> list[GenResult]:
        return [self._one(r) for r in requests]

    def close(self) -> None:
        return None

    def _one(self, request: GenRequest) -> GenResult:
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=request.messages,
                    n=request.n,
                    temperature=request.temperature,
                    max_tokens=request.max_new_tokens,
                    seed=request.seed,
                )
                return GenResult(texts=[choice.message.content or "" for choice in response.choices])
            except Exception as error:  # network and rate limit errors are retried
                wait = 2 ** attempt
                log.warning("OpenAI call failed (%s); retrying in %ss", error, wait)
                time.sleep(wait)
        raise RuntimeError("OpenAI call failed after retries")


def openai_available() -> bool:
    if not os.environ.get("OPENAI_API_KEY"):
        return False
    try:
        import openai  # noqa: F401
    except ImportError:
        return False
    return True

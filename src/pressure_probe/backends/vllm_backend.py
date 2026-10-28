"""vLLM backend: fast batched generation with prefix caching. Optional."""

from __future__ import annotations

from ..io_utils import get_logger
from .base import GenRequest, GenResult

log = get_logger(__name__)


class VLLMBackend:
    name = "vllm"

    def __init__(self, hf_id: str, revision: str | None, quantization: str | None, seed: int,
                 bf16_ok: bool) -> None:
        from vllm import LLM

        self.llm = LLM(
            model=hf_id,
            revision=revision,
            quantization=quantization,
            dtype="auto" if bf16_ok else "float16",
            gpu_memory_utilization=0.85,
            max_model_len=2048,
            seed=seed,
            enable_prefix_caching=True,
        )
        self.tokenizer = self.llm.get_tokenizer()
        self.precision = f"{quantization or 'native'}-{'auto' if bf16_ok else 'fp16'}"
        log.info("Loaded %s with vLLM (%s)", hf_id, self.precision)

    def generate(self, requests: list[GenRequest]) -> list[GenResult]:
        prompts = [self.tokenizer.apply_chat_template(r.messages, tokenize=False, add_generation_prompt=True)
                   for r in requests]
        params = [self._params(r) for r in requests]
        outputs = self.llm.generate(prompts, params, use_tqdm=False)
        return [self._result(r, o) for r, o in zip(requests, outputs)]

    def close(self) -> None:
        del self.llm

    @staticmethod
    def _params(request: GenRequest):
        from vllm import SamplingParams

        logprobs = max(request.first_token_top_k, 1) if (request.want_logprobs or request.first_token_top_k) else None
        kwargs = dict(n=request.n, temperature=request.temperature, top_p=1.0, max_tokens=request.max_new_tokens,
                      repetition_penalty=1.0, seed=request.seed, logprobs=logprobs)
        try:
            return SamplingParams(top_k=-1, **kwargs)
        except (ValueError, TypeError):  # newer releases use 0 to disable top-k
            return SamplingParams(top_k=0, **kwargs)

    @staticmethod
    def _result(request: GenRequest, output) -> GenResult:
        texts, logprobs, top = [], [], {}
        for k, completion in enumerate(output.outputs):
            texts.append(completion.text)
            if completion.logprobs is None:
                continue
            per_token = [step[token_id].logprob for token_id, step in zip(completion.token_ids, completion.logprobs)]
            logprobs.append(per_token)
            if k == 0 and request.first_token_top_k and completion.logprobs:
                for entry in completion.logprobs[0].values():
                    token = (entry.decoded_token or "").strip()
                    top.setdefault(token, entry.logprob)
        return GenResult(texts=texts, token_logprobs=logprobs if request.want_logprobs else None,
                         first_token_logprobs=top)

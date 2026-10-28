"""Out of memory recovery in the transformers backend and the NLI scorer, without a GPU."""

from types import SimpleNamespace

import numpy as np

from pressure_probe.backends.base import GenRequest, GenResult
from pressure_probe.backends.hf_backend import HFBackend
from pressure_probe.nli import DebertaNLI


class FakeOutOfMemory(Exception):
    pass


def fake_torch(calls: list[str]) -> SimpleNamespace:
    cuda = SimpleNamespace(OutOfMemoryError=FakeOutOfMemory, empty_cache=lambda: calls.append("empty"))
    return SimpleNamespace(cuda=cuda)


def test_generation_splits_batch_after_out_of_memory():
    calls: list[str] = []
    backend = object.__new__(HFBackend)
    backend.torch = fake_torch(calls)
    sizes = []

    def batch(requests):
        sizes.append(len(requests))
        if len(requests) > 2:
            raise FakeOutOfMemory()
        return [GenResult(texts=[r.messages[0]["content"]]) for r in requests]

    backend._batch = batch
    requests = [GenRequest(messages=[{"role": "user", "content": str(i)}]) for i in range(7)]
    results = backend._batch_with_retry(requests)
    assert [r.texts[0] for r in results] == [str(i) for i in range(7)]
    assert max(size for size in sizes if size <= 2) <= 2 and "empty" in calls


def test_generation_reraises_when_single_prompt_does_not_fit():
    backend = object.__new__(HFBackend)
    backend.torch = fake_torch([])

    def batch(requests):
        raise FakeOutOfMemory()

    backend._batch = batch
    try:
        backend._batch_with_retry([GenRequest(messages=[{"role": "user", "content": "x"}])])
    except FakeOutOfMemory:
        return
    raise AssertionError("expected the error to propagate")


def test_nli_splits_batch_after_out_of_memory():
    scorer = object.__new__(DebertaNLI)
    scorer.torch = fake_torch([])
    scorer.batch_size = 8

    def chunk(pairs):
        if len(pairs) > 3:
            raise FakeOutOfMemory()
        return np.array([[0.0, 0.0, float(p[0])] for p in pairs])

    scorer._chunk = chunk
    pairs = [(str(i), "h") for i in range(10)]
    probs = scorer.probabilities(pairs)
    assert probs.shape == (10, 3)
    assert probs[:, 2].tolist() == [float(i) for i in range(10)]


def test_sampling_batches_are_capped_by_sequence_budget():
    backend = object.__new__(HFBackend)
    backend.batch_size = 16
    backend.max_sequences = 64
    seen = []

    def run(requests):
        seen.append(len(requests))
        return [GenResult(texts=["a"] * r.n) for r in requests]

    backend._batch_with_retry = run
    requests = [GenRequest(messages=[{"role": "user", "content": "q"}], n=10, temperature=0.5) for _ in range(20)]
    results = backend.generate(requests)
    assert len(results) == 20 and all(len(r.texts) == 10 for r in results)
    assert max(seen) * 10 <= 64

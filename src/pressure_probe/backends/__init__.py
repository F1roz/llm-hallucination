"""Backend selection: vLLM when it works, transformers otherwise, mock for tests."""

from __future__ import annotations

from ..config import Config, ModelSpec
from ..io_utils import get_logger
from .base import Backend, GenRequest, GenResult

log = get_logger(__name__)

LOW_MEMORY_GB = 20.0

__all__ = ["Backend", "GenRequest", "GenResult", "load_backend", "gpu_memory_gb"]


def gpu_memory_gb() -> float:
    try:
        import torch
    except ImportError:
        return 0.0
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.get_device_properties(0).total_memory / 1024**3


def load_backend(config: Config, spec: ModelSpec) -> Backend:
    if config.backend == "mock":
        from .mock_backend import MockBackend

        return MockBackend(config.fixture_path or "", spec.key, config.seed)
    memory = gpu_memory_gb()
    if memory == 0.0:
        raise RuntimeError("No CUDA GPU found. Use a GPU runtime or the mock backend for tests.")
    low_memory = memory < LOW_MEMORY_GB
    if config.backend in {"auto", "vllm"}:
        try:
            return _vllm(config, spec, low_memory)
        except Exception as error:
            if config.backend == "vllm":
                raise
            log.warning("vLLM unavailable (%s); falling back to transformers", error)
    from .hf_backend import HFBackend

    return HFBackend(spec.hf_id, spec.revision, low_memory, config.batch_size)


def _vllm(config: Config, spec: ModelSpec, low_memory: bool) -> Backend:
    import torch

    from .vllm_backend import VLLMBackend

    bf16_ok = torch.cuda.is_bf16_supported()
    if low_memory and spec.low_memory_hf_id:
        return VLLMBackend(spec.low_memory_hf_id, spec.low_memory_revision, spec.low_memory_quantization,
                           config.seed, bf16_ok)
    return VLLMBackend(spec.hf_id, spec.revision, None, config.seed, bf16_ok)

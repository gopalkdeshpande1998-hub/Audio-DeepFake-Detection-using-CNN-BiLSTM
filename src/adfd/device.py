"""Which device inference runs on, on Windows, Linux and macOS.

    best_device()          "cuda" (NVIDIA GPU) > "mps" (Apple silicon GPU, macOS) > "cpu"
    resolve(name)          a requested device if this machine has it, else the CPU
    gpu_device()           the GPU this machine has ("cuda" / "mps"), or None
    describe(name)         a readable label, e.g. "GPU (NVIDIA GeForce RTX 3050)" or "Apple GPU (MPS)"
    empty_cache(name)      release cached GPU memory after an out-of-memory error

The ADFD_DEVICE environment variable overrides the choice (cpu | cuda | mps). Mixed precision (float16
autocast) is used on CUDA only; MPS and the CPU run in float32. Operators MPS lacks fall back to the CPU
(PYTORCH_ENABLE_MPS_FALLBACK=1, set in adfd.paths.configure_caches before torch is imported).
"""

from __future__ import annotations

import os


def _mps_available() -> bool:
    import torch

    mps = getattr(torch.backends, "mps", None)
    return bool(mps is not None and mps.is_available())


def gpu_device() -> str | None:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if _mps_available():
        return "mps"
    return None


def best_device() -> str:
    env = os.environ.get("ADFD_DEVICE")
    if env:
        return resolve(env)
    return gpu_device() or "cpu"


def resolve(name: str | None) -> str:
    """The requested device when this machine has it; otherwise the CPU (never raises)."""
    name = (name or "cpu").lower()
    if name.startswith("cuda"):
        import torch

        return name if torch.cuda.is_available() else "cpu"
    if name == "mps":
        return "mps" if _mps_available() else "cpu"
    return "cpu"


def describe(name: str) -> str:
    name = resolve(name)
    if name.startswith("cuda"):
        import torch

        return f"GPU ({torch.cuda.get_device_name(0)})"
    if name == "mps":
        return "Apple GPU (MPS)"
    return "CPU"


def empty_cache(name: str | None = None) -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if _mps_available() and hasattr(torch, "mps"):
        torch.mps.empty_cache()

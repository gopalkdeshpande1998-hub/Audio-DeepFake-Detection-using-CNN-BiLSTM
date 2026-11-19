"""One seed for Python, NumPy and PyTorch (CPU and CUDA)."""

from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed every random generator the pipeline uses.

    deterministic=True also forces deterministic cuDNN kernels: runs become
    bit-reproducible on the same GPU, at some cost in speed.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:  # manifest tools work without torch
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.benchmark = True


def worker_init_fn(worker_id: int) -> None:
    """DataLoader worker seeding: each worker gets a distinct, reproducible stream."""
    import torch

    base = torch.initial_seed() % 2**32
    np.random.seed((base + worker_id) % 2**32)
    random.seed(base + worker_id)

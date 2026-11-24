"""Device choice (CUDA / Apple MPS / CPU) and portable path files (macOS support)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from adfd import device as dev


@pytest.fixture
def machine(monkeypatch):
    """Pretend this machine has the given GPUs."""
    def set_(cuda: bool, mps: bool):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)
        monkeypatch.delenv("ADFD_DEVICE", raising=False)
    return set_


@pytest.mark.parametrize("cuda,mps,best", [(True, False, "cuda"), (False, True, "mps"), (False, False, "cpu"),
                                          (True, True, "cuda")])
def test_best_device_prefers_cuda_then_mps_then_cpu(machine, cuda, mps, best):
    machine(cuda, mps)
    assert dev.best_device() == best
    assert dev.gpu_device() == (None if best == "cpu" else best)


def test_requested_device_falls_back_to_cpu_when_missing(machine):
    machine(False, True)                                  # an Apple silicon Mac
    assert dev.resolve("cuda") == "cpu" and dev.resolve("mps") == "mps" and dev.resolve(None) == "cpu"
    assert dev.describe("mps") == "Apple GPU (MPS)" and dev.describe("cuda") == "CPU"


def test_env_override(machine, monkeypatch):
    machine(False, True)
    monkeypatch.setenv("ADFD_DEVICE", "cpu")
    assert dev.best_device() == "cpu"


def test_analyzers_accept_mps_on_a_mac(machine):
    """inference.Analyzer and TemporalAnalyzer resolve devices through adfd.device (no silent CUDA-only check)."""
    import inspect

    from adfd import inference, temporal

    assert "resolve_device(device)" in inspect.getsource(inference.Analyzer.__init__)
    assert "resolve_device(device)" in inspect.getsource(temporal.TemporalAnalyzer.__init__)


def test_paths_expand_home_and_keep_relative_paths(tmp_path, monkeypatch):
    from adfd.paths import REPO_ROOT, load_paths

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    p = load_paths(REPO_ROOT / "configs" / "paths.macos.yaml")
    assert p.dataset("asvspoof2019_la") == Path(tmp_path) / "datasets" / "ASVspoof2019-dataset"
    assert p.runs_dir == REPO_ROOT / "runs"


def test_mps_fallback_is_enabled_before_torch_runs(monkeypatch, tmp_path):
    from adfd.paths import configure_caches, load_paths

    monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)
    configure_caches(load_paths())
    import os

    assert os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] == "1"

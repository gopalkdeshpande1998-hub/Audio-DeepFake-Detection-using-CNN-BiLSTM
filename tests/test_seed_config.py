"""Seeding, config merging and run-info tests."""

import random

import numpy as np
import pytest

from adfd.config import deep_merge, load_config
from adfd.runinfo import collect_run_info
from adfd.seed import seed_everything


def _draw():
    out = [random.random(), float(np.random.rand())]
    torch = pytest.importorskip("torch")
    out.append(float(torch.rand(1)))
    return out


def test_seed_everything_is_reproducible():
    seed_everything(123)
    a = _draw()
    seed_everything(123)
    b = _draw()
    seed_everything(124)
    c = _draw()
    assert a == b
    assert a != c


def test_deep_merge_overrides_nested_keys_only():
    base = {"train": {"lr": 1e-3, "batch_size": 32}, "seed": 1}
    merged = deep_merge(base, {"train": {"lr": 5e-4}})
    assert merged == {"train": {"lr": 5e-4, "batch_size": 32}, "seed": 1}
    assert base["train"]["lr"] == 1e-3  # base is not modified


def test_base_config_keeps_test_sets_out_of_training():
    cfg = load_config()
    assert cfg["data"]["train_split"] == "train"
    assert cfg["data"]["val_split"] == "dev"
    assert "in_the_wild:test" in cfg["data"]["test_sets"]
    assert cfg["train"]["amp"] is True  # needed for the 4 GB GPU


def test_collect_run_info_has_core_fields():
    info = collect_run_info({"seed": 1})
    for key in ("git_commit", "config", "device", "packages", "created_utc"):
        assert key in info

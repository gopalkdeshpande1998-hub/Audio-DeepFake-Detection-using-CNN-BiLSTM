"""Checks on the built manifests against the real datasets.

Skipped when a manifest has not been built or the dataset is not on this machine.
Build them with: python scripts/data/build_manifests.py
"""

import pandas as pd
import pytest
import soundfile as sf

from adfd.data.manifests import validate_manifest
from adfd.paths import DATASET_NAMES, load_paths

pytestmark = pytest.mark.datasets
PATHS = load_paths()


def _load(name):
    manifest = PATHS.manifest(name)
    if not manifest.exists() or not PATHS.dataset(name).exists():
        pytest.skip(f"{name}: manifest or dataset not available")
    return pd.read_csv(manifest, keep_default_na=False)


@pytest.mark.parametrize("name", DATASET_NAMES)
def test_manifest_is_valid(name):
    assert validate_manifest(_load(name), name) == []


@pytest.mark.parametrize("name", DATASET_NAMES)
def test_sample_of_files_decode(name):
    df = _load(name).sample(n=20, random_state=0)
    root = PATHS.dataset(name)
    for row in df.itertuples():
        start = int(row.start_s * row.sample_rate)
        stop = int(row.end_s * row.sample_rate)
        audio, sr = sf.read(str(root / row.path), start=start, stop=stop, always_2d=True)
        assert sr == row.sample_rate
        assert audio.shape[0] > 0


def test_no_utterance_in_two_asvspoof_splits():
    df = _load("asvspoof2019_la")
    assert df.groupby("utt_id")["split"].nunique().max() == 1


def test_cross_dataset_sets_are_test_only():
    for name in ("in_the_wild", "deep_voice"):
        assert set(_load(name)["split"]) == {"test"}

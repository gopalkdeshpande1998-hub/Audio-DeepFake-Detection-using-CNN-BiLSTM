"""Frame-label alignment of the temporal training crops (adfd.data.partialspoof)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from adfd.data.partialspoof import HOP, TemporalCropDataset, frames_for, samples_for, vad_to_regions


def test_crop_length_gives_exactly_k_backbone_frames():
    for k in (1, 12, 25, 60):
        assert frames_for(samples_for(k)) == k


def _clip(tmp_path, labels):
    """Audio whose every 0.16-s frame carries its own constant value = frame index, so crops can be checked."""
    x = np.concatenate([np.full(HOP, (i + 1) / 100, np.float32) for i in range(len(labels))])
    p = tmp_path / "u.wav"
    sf.write(p, x, 16000, subtype="FLOAT")
    rows = pd.DataFrame([{"utt_id": "u", "path": str(p), "source": "ps", "label": int(max(labels)), "recording_id": ""}])
    return rows, {"u": np.array(labels, np.int8)}


@pytest.mark.parametrize("train", [True, False])
def test_crop_audio_and_labels_stay_aligned(tmp_path, train):
    labels = [0] * 10 + [1] * 7 + [0] * 13                         # 30 frames
    rows, seg = _clip(tmp_path, labels)
    ds = TemporalCropDataset(rows, seg, 25, train)
    for _ in range(20):
        wav, y, clip, _ = ds[0]
        assert len(wav) == samples_for(25) and len(y) == 25
        idx = np.round(wav.numpy()[: 25 * HOP: HOP] * 100).astype(int) - 1   # frame index carried by the audio
        assert np.array_equal(y.numpy(), np.array(labels)[idx])
        assert clip.item() == float(y.max() > 0)


def test_short_clip_repeats_audio_with_labels(tmp_path):
    labels = [0, 1, 1, 0, 0]
    rows, seg = _clip(tmp_path, labels)
    wav, y, _, _ = TemporalCropDataset(rows, seg, 25, False)[0]
    idx = np.round(wav.numpy()[: 25 * HOP: HOP] * 100).astype(int) - 1
    assert np.array_equal(y.numpy(), np.array(labels)[idx]) and np.array_equal(idx[:10], [0, 1, 2, 3, 4] * 2)


def test_vad_regions_merge_fake_runs_across_silence():
    vad = [(0.0, 0.69, 2), (0.69, 1.07, 1), (1.07, 1.074, 2), (1.074, 1.359, 0), (1.359, 1.3595, 2),
           (1.3595, 1.80, 0), (1.80, 1.97, 2)]
    assert vad_to_regions(vad) == [(1.074, 1.80)]

"""Phase 4-6 prototype: calibration, OOD checks, whole-file analysis and the REST API (CPU, tiny model)."""

import io
import json

import numpy as np
import pytest
import soundfile as sf
import torch

from adfd.calibration import ece, fit_calibration, three_way_stats
from adfd.ood import EmbeddingOOD, check_signal, signal_features

SR = 16000
CFG = {"name": "tiny", "seed": 0,
       "model": {"name": "crnn", "frontend": {"name": "lfcc"}, "kwargs": {"cnn": "light", "rnn": "gru", "pooling": "asp"}},
       "audio": {"sample_rate": SR, "crop_seconds": 4.0, "pad_mode": "repeat", "trim_silence": True}}


def speech_like(seconds=3.0, seed=0):
    """Pitched, syllable-modulated harmonic signal with pauses (passes the speech checks)."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    f0 = 120 + 20 * np.sin(2 * np.pi * 0.5 * t)
    phase = 2 * np.pi * np.cumsum(f0) / SR
    x = sum(np.sin(k * phase) / k for k in range(1, 15)) + 0.05 * rng.standard_normal(len(t))
    env = np.clip(np.sin(2 * np.pi * 4 * t), 0, None) ** 2
    return (0.3 * x * env / np.abs(x).max()).astype(np.float32)


def test_calibration_separates_and_thresholds_are_ordered():
    rng = np.random.default_rng(0)
    z = np.r_[rng.normal(-2, 1, 500), rng.normal(2, 1, 500)]
    y = np.r_[np.zeros(500), np.ones(500)].astype(int)
    cal = fit_calibration(z, y, "synthetic")
    p = cal.prob_fake(z)
    assert cal.t_real <= cal.t_fake
    assert p[y == 1].mean() > 0.8 and p[y == 0].mean() < 0.2
    st = three_way_stats(p, y, cal)
    assert st["real_called_fake_pct"] <= 5.5 and st["fake_called_real_pct"] <= 5.5
    assert ece(p, y) < 5


def test_ece_is_zero_for_perfect_confident_predictions():
    assert ece(np.array([0.0, 1.0, 0.0, 1.0]), np.array([0, 1, 0, 1])) == pytest.approx(0, abs=1e-9)


def test_signal_checks_flag_tones_noise_silence_but_not_speech():
    t = np.arange(4 * SR) / SR
    assert check_signal(signal_features(speech_like())) == []
    assert check_signal(signal_features(0.3 * np.sin(2 * np.pi * 440 * t)))
    assert check_signal(signal_features(0.3 * np.random.default_rng(1).standard_normal(4 * SR)))
    assert check_signal(signal_features(np.zeros(4 * SR) + 1e-6))
    assert any("shorter" in r for r in check_signal(signal_features(speech_like(0.5))))


def test_mahalanobis_threshold_and_roundtrip():
    emb = np.random.default_rng(0).standard_normal((500, 8))
    eo = EmbeddingOOD.fit(emb)
    assert (eo.distance(emb) > eo.threshold).mean() == pytest.approx(0.01, abs=0.01)
    eo2 = EmbeddingOOD.from_dict(json.loads(json.dumps(eo.to_dict())))
    assert eo2.distance(emb[:1] + 10)[0] > eo2.threshold


@pytest.fixture(scope="module")
def tiny_run(tmp_path_factory):
    from adfd.models import build_model

    run = tmp_path_factory.mktemp("runs") / "tiny_s42"
    run.mkdir()
    torch.manual_seed(0)
    torch.save({"model": build_model(CFG).state_dict(), "config": CFG}, run / "best.pt")
    (run / "train_summary.json").write_text("{}", encoding="utf-8")      # marks training as finished
    return run


def test_analyzer_windows_timeline_and_saliency(tiny_run):
    from adfd.inference import Analyzer

    an = Analyzer(tiny_run)
    x = np.concatenate([np.zeros(SR, np.float32), speech_like(9.0), np.zeros(SR, np.float32)])
    res = an.analyze(x)
    assert res["verdict"] in {"Real", "Fake"} and 0 <= res["p_fake"] <= 1      # uncalibrated: two-way
    s, e = res["analysed_region_s"]
    assert s >= 0.9 and e <= 10.1                                             # silence trimmed
    assert len(res["timeline"]) >= 6 and res["timeline"][0]["end_s"] - res["timeline"][0]["start_s"] == pytest.approx(4)
    sal = an.saliency(x[SR:5 * SR])
    assert sal["map"].shape[0] == 60 and np.isfinite(sal["map"]).all()
    short = an.analyze(speech_like(1.5))
    assert len(short["timeline"]) == 1


def test_api_predict_and_health(tiny_run, monkeypatch):
    from fastapi.testclient import TestClient

    from adfd import registry
    import api.main as api

    monkeypatch.setattr(registry, "runs_dir", lambda: tiny_run.parent)
    monkeypatch.setattr(registry, "MODEL", tiny_run.name)          # the single production model slot
    monkeypatch.setattr(registry, "LABELS", {tiny_run.name: "tiny test model"})
    monkeypatch.setattr(registry, "default_run", lambda: tiny_run)
    api.analyzer.cache_clear()
    client = TestClient(api.app)
    assert client.get("/health").json()["default_model"] == "tiny_s42"
    assert "tiny_s42" in client.get("/models").json()["models"]
    buf = io.BytesIO()
    sf.write(buf, speech_like(5.0), SR, format="WAV")
    r = client.post("/predict?timeline=true", files={"file": ("a.wav", buf.getvalue(), "audio/wav")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["verdict"] in {"Real", "Fake"} and len(body["timeline"]) >= 1
    assert abs(body["p_fake"] + body["p_real"] - 1) < 1e-9
    assert body["confidence"] == max(body["p_fake"], body["p_real"])
    assert body["model_label"] == "tiny test model" and "members" not in body and "decision" not in body
    assert body["trust"]["n_members"] == 1 and body["explanation"]
    assert client.post("/predict", files={"file": ("x.wav", b"not audio", "audio/wav")}).status_code == 415
    assert client.post("/predict?model=nope", files={"file": ("a.wav", buf.getvalue())}).status_code == 404

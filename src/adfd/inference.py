"""Whole-file analysis for the app, the API and explain.py.

    an = Analyzer("runs/e4a_rawboost_s42")         # loads best.pt + reliability.json
    res = an.analyze(an.load_audio("clip.wav"))   # verdict, probability, timeline, OOD

Steps for one file:
  1. decode, downmix, resample to 16 kHz;
  2. OOD signal checks (adfd.ood) on the whole file;
  3. the region the model was trained on: the speech region for silence-trimmed models,
     the whole file otherwise;
  4. sliding windows of the training crop length (4 s), hop 1 s; short files give one
     window, repeat-padded exactly as in training;
  5. spoof logit per window; the file-level logit is the mean over windows;
  6. calibrated p_fake (adfd.calibration, fitted on dev, class-balanced) and a two-way
     verdict: Fake if p_fake >= 0.5, else Real (the app never answers Uncertain); an
     input that fails the OOD checks keeps its verdict and carries a warning;
     embedding OOD distance from the penultimate layer.
A saliency map (gradient x input on the model's input features) is available for the
most suspicious window.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from adfd.calibration import Calibration, sigmoid
from adfd.data.datasets import fit_length, normalize_level
from adfd.data.silence import speech_bounds
from adfd.models import build_model, count_parameters
from adfd.models.voiceguard import CLIP_S as VG_CLIP_S, VOICEGUARD_MODELS, load_voiceguard
from adfd.ood import DEFAULT_RANGES, EmbeddingOOD, check_signal, signal_features

SR = 16000
HOP_S = 1.0
DECISION_P = 0.5          # Fake if calibrated P(fake) >= 0.5, else Real
SPECTRAL_FRONTENDS = {"mfcc", "lfcc", "logmel", "cqcc", "sinc", "vg_logmel"}


def load_audio(src) -> np.ndarray:
    """Path, bytes or file-like -> float32 mono 16 kHz in [-1, 1]."""
    import soundfile as sf
    import soxr

    if isinstance(src, (bytes, bytearray)):
        src = io.BytesIO(src)
    try:
        audio, sr = sf.read(src, dtype="float32", always_2d=True)
    except Exception:  # formats libsndfile cannot read (M4A/AAC, WebM, video files): decode with PyAV
        if hasattr(src, "seek"):
            src.seek(0)
        try:
            audio, sr = _decode_av(src)
        except Exception:
            import librosa

            if hasattr(src, "seek"):
                src.seek(0)
            audio, sr = librosa.load(src, sr=None, mono=False)
            audio = np.atleast_2d(audio).T.astype(np.float32)
    audio = audio.mean(axis=1)
    if sr != SR:
        audio = soxr.resample(audio, sr, SR, quality="HQ")
    return np.clip(audio, -1.0, 1.0).astype(np.float32)


def _decode_av(src) -> tuple[np.ndarray, int]:
    """Audio track of any container FFmpeg knows (MP4, MOV, MKV, WebM, M4A, ...) via PyAV."""
    import av

    with av.open(src) as box:
        stream = next(s for s in box.streams if s.type == "audio")
        res = av.AudioResampler(format="flt", layout="mono", rate=SR)
        chunks = []
        for frame in box.decode(stream):
            for f in res.resample(frame):
                chunks.append(f.to_ndarray().reshape(-1))
        for f in res.resample(None):
            chunks.append(f.to_ndarray().reshape(-1))
    if not chunks:
        raise ValueError("no audio in file")
    return np.concatenate(chunks)[:, None].astype(np.float32), SR


def _last_linear(model: nn.Module) -> nn.Linear:
    last = None
    for m in model.modules():
        if isinstance(m, nn.Linear):
            last = m
    return last


@dataclass
class Window:
    start_s: float
    end_s: float
    logit: float
    p_fake: float


class Analyzer:
    def __init__(self, run_dir, device: str = "cpu"):
        self.run_dir = Path(run_dir)
        self.device = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
        if self.run_dir.name in VOICEGUARD_MODELS and not (self.run_dir / "best.pt").exists():
            # external VoiceGuard checkpoint: 5 s zero-padded log-mel input, no trimming
            self.cfg = {"audio": {"crop_seconds": VG_CLIP_S, "pad_mode": "zero", "trim_silence": False,
                                  "normalize": None},
                        "model": {"name": "voiceguard", "frontend": {"name": "vg_logmel"}}}
            self.model = load_voiceguard(self.run_dir.name)
        else:
            ckpt = torch.load(self.run_dir / "best.pt", map_location="cpu", weights_only=False)
            self.cfg = ckpt["config"]
            self.model = build_model(self.cfg)
            self.model.load_state_dict(ckpt["model"])
        self.model.to(self.device).eval()
        audio = self.cfg["audio"]
        self.crop = int(audio["crop_seconds"] * SR)
        self.pad_mode = audio.get("pad_mode", "repeat")
        self.trim = bool(audio.get("trim_silence"))
        self.normalize = audio.get("normalize")
        self.frontend_name = self.cfg["model"].get("frontend", {}).get("name", "mfcc")
        self.name = self.run_dir.name
        self.n_params = count_parameters(self.model, False)
        rel_f = self.run_dir / "reliability.json"
        rel = json.loads(rel_f.read_text(encoding="utf-8")) if rel_f.exists() else {}
        self.calibration = Calibration.from_dict(rel["calibration"]) if "calibration" in rel else None
        self.signal_ranges = {k: tuple(v) for k, v in rel.get("signal_ranges", {}).items()} or DEFAULT_RANGES
        self.emb_ood = EmbeddingOOD.from_dict(rel["embedding_ood"]) if "embedding_ood" in rel else None
        self._emb = None
        _last_linear(self.model).register_forward_hook(lambda m, inp, out: setattr(self, "_emb", inp[0].detach()))

    load_audio = staticmethod(load_audio)

    # -- windows ------------------------------------------------------------------------
    def region(self, x: np.ndarray) -> tuple[int, int]:
        return speech_bounds(x) if self.trim else (0, len(x))

    def windows(self, x: np.ndarray) -> list[tuple[int, int]]:
        s, e = self.region(x)
        if e - s <= self.crop:
            return [(s, e)]
        hop = int(HOP_S * SR)
        starts = list(range(s, e - self.crop + 1, hop))
        if starts[-1] + self.crop < e:
            starts.append(e - self.crop)
        return [(a, a + self.crop) for a in starts]

    def _prep(self, seg: np.ndarray) -> np.ndarray:
        seg = fit_length(seg, self.crop, self.pad_mode).astype(np.float32)
        return normalize_level(seg, self.normalize)

    @torch.no_grad()
    def _score(self, segs: list[np.ndarray], batch: int = 16) -> tuple[np.ndarray, np.ndarray]:
        logits, embs = [], []
        for i in range(0, len(segs), batch):
            wav = torch.from_numpy(np.stack(segs[i:i + batch])).to(self.device)
            with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.device.type == "cuda"):
                out = self.model(wav)
            logits.append(out.float().cpu().numpy())
            embs.append(self._emb.float().cpu().numpy())
        return np.concatenate(logits), np.concatenate(embs)

    def score_segments(self, x: np.ndarray, spans: list[tuple[int, int]]) -> np.ndarray:
        """Spoof logits for arbitrary sample ranges, each prepared exactly like an analysis window
        (padded to the training crop as in training). Used for the ensemble's local timeline."""
        return self._score([self._prep(x[a:b]) for a, b in spans])[0]

    def prob(self, z) -> np.ndarray:
        return self.calibration.prob_fake(z) if self.calibration else sigmoid(np.asarray(z, float))

    # -- analysis -----------------------------------------------------------------------
    def analyze(self, x: np.ndarray) -> dict:
        feats = signal_features(x)
        reasons = check_signal(feats, self.signal_ranges)
        wins = self.windows(x)
        logits, embs = self._score([self._prep(x[a:b]) for a, b in wins])
        z = float(logits.mean())
        p = float(self.prob(z))
        timeline = [Window(a / SR, b / SR, float(l), float(self.prob(l))) for (a, b), l in zip(wins, logits)]
        emb_dist = None
        if self.emb_ood is not None:
            emb_dist = float(self.emb_ood.distance(embs.mean(axis=0))[0])
            if emb_dist > self.emb_ood.threshold:
                reasons.append("the model's internal features are far from anything seen in training")
        cal = self.calibration
        verdict = "Fake" if p >= DECISION_P else "Real"      # two-way: never Uncertain
        suspicious = sorted((w for w in timeline if w.p_fake >= DECISION_P), key=lambda w: -w.p_fake)[:3]
        return {
            "verdict": verdict, "p_fake": p, "logit": z,
            "out_of_distribution": bool(reasons), "ood_reasons": reasons,
            "signal": feats, "embedding_distance": emb_dist,
            "embedding_threshold": self.emb_ood.threshold if self.emb_ood else None,
            "duration_s": len(x) / SR, "analysed_region_s": [r / SR for r in self.region(x)],
            "timeline": [w.__dict__ for w in timeline],
            "suspicious_segments": [w.__dict__ for w in suspicious],
            "thresholds": {"t_real": DECISION_P, "t_fake": DECISION_P, "decision": DECISION_P},
            "calibrated": cal is not None, "model": self.name,
        }

    # -- explanation ----------------------------------------------------------------------
    def saliency(self, seg: np.ndarray) -> dict:
        """Gradient x input on the model's input features for one window.

        Spectral front ends give a (freq, time) map over that feature; SSL and raw
        waveform models give a time profile only (their inputs have no frequency axis).
        """
        wav = torch.from_numpy(self._prep(seg)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.model.frontend(wav).float()
        feat = feat.detach().requires_grad_(True)
        with torch.enable_grad(), torch.backends.cudnn.flags(enabled=False):
            logit = self.model.backbone(feat)
            logit.sum().backward()
        sal = (feat.grad * feat).abs().detach().cpu().numpy()
        out = {"logit": float(logit.item()), "frontend": self.frontend_name}
        if feat.dim() == 4 and self.frontend_name in SPECTRAL_FRONTENDS:
            out["feature"] = feat.detach().cpu().numpy()[0, 0]
            out["map"] = sal[0, 0]
        else:
            prof = sal.reshape(sal.shape[0], -1, sal.shape[-1]).sum(axis=1)[0] if feat.dim() == 4 else \
                np.add.reduceat(sal[0], np.arange(0, sal.shape[-1], 160))
            out["time_profile"] = prof
        return out

"""The app's detector: WavLM-Base+ + CNN-BiLSTM-Att (E7a) on its own.

    an = SingleModelAnalyzer(device="cuda")        # loads runs/e7a_wavlm_crnn_s42 only
    res = an.analyze(load_audio("clip.wav"))       # verdict, P(fake), timeline, segments, explanation

How a file is decided (no other model is loaded or run)
  1. adfd.inference.Analyzer scores the file exactly as before: the speech region, 4-s windows
     (the training crop), hop 1 s, file logit = mean window logit, E7a's own calibration
     (runs/e7a_wavlm_crnn_s42/reliability.json, fitted on dev) -> P(fake);
  2. verdict: FAKE if P(fake) >= 0.5 (DECISION_P), else REAL; P(real) = 1 - P(fake). Input that
     fails the speech (OOD) checks keeps its verdict and carries a warning.
Where (time-local analysis, display only, never changes the verdict)
  3. E7a scores 4-s windows (its training crop, so its calibration applies unchanged) over the whole
     recording with a 0.5-s hop (longer for long files, at most 120 windows); each timeline step gets
     the mean log-odds of the speech windows covering its centre;
  4. consecutive steps >= 50% form the flagged segments, each with acoustic evidence (adfd.segments).
E7a reads WavLM features, which have no frequency axis, so there is no frequency-band saliency.
Segment-level detection (when runs/t1_wavlm_crnn_temporal_s42 has best.pt and temporal_calibration.json)
  5. the temporal model (adfd.temporal, E7a + a frame head trained on PartialSpoof's time-stamped labels) labels
     every 0.16 s REAL / FAKE / UNCERTAIN / NO SPEECH: res["temporal"]. It never changes the E7a verdict.
     Without a calibrated temporal run res["temporal"] is None and the UI falls back to step 3, marked as such.
"""

from __future__ import annotations

import numpy as np

from adfd import forensics, segments
from adfd.inference import DECISION_P, SR, Analyzer
from adfd.paths import load_paths
from adfd.timeline import local_grid, logit, speech_windows, step_logodds, timeline_regions

MODEL_RUN = "e7a_wavlm_crnn_s42"
TEMPORAL_RUN = "t1_wavlm_crnn_temporal_s42"
TEMPORAL_COST_PER_S = 0.5      # progress-bar weight of the segment model per second of audio
MODEL_LABEL = "WavLM + CNN-BiLSTM-Att (E7a)"
LOCAL_WINDOW_S = 4.0          # E7a's training crop: the calibration is valid for these windows
EXPLAIN_COST_PER_S = 0.23


class SingleModelAnalyzer:
    def __init__(self, device: str = "cpu", analyzer: Analyzer | None = None, run_dir=None, label: str = MODEL_LABEL,
                 temporal=None):
        self.analyzer = analyzer or Analyzer(run_dir or load_paths().runs_dir / MODEL_RUN, device)
        self.name = self.analyzer.name
        self.label = label
        self.reference = forensics.load_reference()
        self.temporal = temporal if temporal is not None else (load_temporal(device) if analyzer is None else None)

    # used by the app's internals panel (Grad-CAM, attention, layer weights)
    @property
    def analyzers(self) -> dict:
        return {self.name: self.analyzer}

    def analyze(self, x: np.ndarray, explain: bool = True, progress=None) -> dict:
        """progress: optional callback(fraction 0-1, text) for a UI progress bar."""
        an = self.analyzer
        n = len(x)
        dur = n / SR
        spans, step, win = local_grid(n, LOCAL_WINDOW_S)
        speech = speech_windows(x, spans)
        n_file = len(an.windows(x))
        weights = [n_file, len(spans)] + ([TEMPORAL_COST_PER_S * dur] if self.temporal else []) + (
            [EXPLAIN_COST_PER_S * dur] if explain else [])
        total = sum(weights) or 1.0

        def report(stage: int, text: str, sub: float = 0.0) -> None:
            if progress is not None:
                f = (sum(weights[:stage]) + sub * weights[min(stage, len(weights) - 1)]) / total
                progress(min(f, 1.0), f"Step {min(stage + 1, len(weights))} of {len(weights)} · {text}")

        def tracker(stage: int, text: str):
            return None if progress is None else (lambda f: report(stage, f"{text} ({100 * f:.0f}%)", f))

        txt = f"{self.label}: scoring the whole file"
        report(0, txt + "…")
        r = an.analyze(x, on_batch=tracker(0, txt)) if progress is not None else an.analyze(x)
        txt = f"{self.label}: scoring {len(spans)} local windows"
        report(1, txt + "…")
        raw = np.asarray(an.score_segments(x, spans, on_batch=tracker(1, txt)) if progress is not None
                         else an.score_segments(x, spans), float)
        zl = np.where(np.isfinite(raw), logit(an.prob(np.nan_to_num(raw))), np.nan)
        valid = speech & np.isfinite(zl)
        t0, zs = step_logodds(spans, zl, n, step, valid)
        step_p = 1.0 / (1.0 + np.exp(-zs))
        scored = np.isfinite(step_p)
        p = float(r["p_fake"])
        per_step = [{"start_s": float(t), "end_s": float(min(t + step, dur)),
                     "p_fake": float(q) if ok else None, "speech": bool(ok)}
                    for t, q, ok in zip(t0, step_p, scored)]
        res = {
            "verdict": r["verdict"], "p_fake": p, "p_real": 1.0 - p, "logit": r["logit"], "gate_reason": None,
            "out_of_distribution": r["out_of_distribution"], "ood_reasons": r["ood_reasons"],
            "signal": r["signal"], "embedding_distance": r.get("embedding_distance"),
            "duration_s": dur, "analysed_region_s": r["analysed_region_s"],
            "model": self.name, "model_label": self.label,
            "model_windows": r["timeline"],             # E7a's own file-level windows (4 s, hop 1 s)
            "per_second": per_step,                     # timeline steps (name kept for API compatibility)
            "step_s": step, "local_window_s": win, "n_local_windows": len(spans),
            "n_local_windows_scored": int(valid.sum()), "time_resolved": int(valid.sum()) > 1,
            "thresholds": {"t_real": DECISION_P, "t_fake": DECISION_P, "decision": DECISION_P},
            "calibrated": r["calibrated"],
        }
        res["timeline"] = per_step
        res["regions"] = timeline_regions(t0, step_p, step, dur)
        res["segments"] = [{"start_s": g["start_s"], "end_s": g["end_s"], "duration_s": g["end_s"] - g["start_s"],
                            "p_fake": g["p_fake_mean"],
                            "p_fake_peak": max(w["p_fake"] for w in per_step if w["p_fake"] is not None
                                               and w["start_s"] < g["end_s"] and w["end_s"] > g["start_s"]),
                            "severity": segments.severity(g["p_fake_mean"])}
                           for g in res["regions"] if g["label"] == "Fake"]
        res["suspicious_segments"] = sorted(({"start_s": s["start_s"], "end_s": s["end_s"], "p_fake": s["p_fake"]}
                                             for s in res["segments"]), key=lambda w: -w["p_fake"])[:3]
        res["temporal"] = None
        stage = 2
        if self.temporal:
            report(stage, "Segment model: labelling every 0.16 s…")
            res["temporal"] = self.temporal.analyze(x)
            stage += 1
        if explain:
            report(stage, "Measuring flagged segments and writing the explanation…")
            self._explain(x, res)
        report(len(weights), "Done")
        return res

    def _explain(self, x: np.ndarray, res: dict) -> None:
        meas = forensics.measure(x)
        res["measurements"] = meas
        res["observations"] = forensics.observations(meas, self.reference)
        if res["segments"]:
            ff = segments.frame_features(x)
            for s in res["segments"]:
                s["evidence"] = segments.segment_evidence(x, s, ff, res, self.reference)
        res["explanation"] = forensics.statements(res)


def load_temporal(device: str = "cpu"):
    """The calibrated segment-level model, or None when it is not trained / calibrated on this machine."""
    run = load_paths().runs_dir / TEMPORAL_RUN
    if not ((run / "best.pt").exists() and (run / "temporal_calibration.json").exists()):
        return None
    from adfd.temporal import TemporalAnalyzer

    return TemporalAnalyzer(run, device)

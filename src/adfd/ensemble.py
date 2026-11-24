"""RETIRED merged detector (E7a + VoiceGuard improved + VoiceGuard v2). Not used by the app or the API since
2026-11-23, which run E7a alone (adfd.single); kept only for the offline research scripts that fitted it.

Merged detector: the project's best model plus the two VoiceGuard checkpoints.

    ens = EnsembleAnalyzer()                      # reads runs/ensemble/ensemble.json
    res = ens.analyze(load_audio("clip.wav"))     # verdict, P(fake), members, timeline, segments

How a file is decided (configuration fitted by scripts/ensemble/fit_ensemble_binary.py)
  1. every member scores the file with its own windows (E7a: 4 s on the speech region;
     VoiceGuard: 5 s, as each was trained) and its own calibration -> member P(fake);
  2. fusion of the member log-odds z_i: each clipped to +-clip (an over-confident member
     cannot swamp the others), weighted sum with weights 0.2 <= w_i <= 0.5 (every member
     counts, and two members that agree always outweigh the third), then calibrated:
         P(fake) = sigmoid(A * sum_i w_i * clip(z_i) + B)
  3. two-way verdict: FAKE if P(fake) >= 0.5, else REAL; Realness = 1 - P(fake).
     Input that fails the speech (OOD) checks keeps its verdict and carries a warning.

Where (time-local analysis, separate from the file verdict)
  4. every member also scores short local windows (2 s, hop 0.5 s; longer for long files,
     at most 120 windows) over the whole recording; each timeline step gets the mean member
     log-odds of the windows covering its centre, fused with the same weights and clipping
     but with the window-level calibration A_loc, B_loc (scripts/ensemble/fit_local_calibration.py:
     members score short windows differently from whole files). These are real per-window
     model outputs, never the file score repeated. Windows with (almost) no speech, or with
     non-finite scores, are left out; steps covered only by such windows have no score
     ("No speech"). A clip with a single scored window has one value (`time_resolved` False).
  5. consecutive steps >= 50% form the flagged segments (severity by their mean P(fake)),
     each with model evidence and acoustic measurements (adfd.segments).
Old three-way configurations (with t_real / t_fake / gates) still load, but are decided
two-way at 0.5 as well.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from adfd import forensics, segments
from adfd.calibration import sigmoid
from adfd.inference import SR, Analyzer
from adfd.paths import load_paths
from adfd.timeline import local_grid, logit, speech_windows, step_logodds, timeline_regions  # noqa: F401

DECISION_P = 0.5
LOCAL_S = 2.0            # local analysis window (s), unless ensemble.json says otherwise
MIN_HOP_S = 0.5          # timeline resolution for clips up to ~1 min
MAX_LOCAL = 120          # local windows per member at most (CPU time); the hop grows for long files
# Relative CPU time per analysis window (progress bar only), measured on the project laptop
# (i7-12650H): E7a ~0.54 s, VoiceGuard improved ~0.21 s, VoiceGuard v2 ~0.03 s per window;
# the explanation ~0.12 s per second of audio.
WINDOW_COST = {"e7a": 1.0, "vg_improved": 0.4, "vg_v2": 0.06}
EXPLAIN_COST_PER_S = 0.23


def window_cost(run: str) -> float:
    return next((v for k, v in WINDOW_COST.items() if run.startswith(k)), 1.0)


def ensemble_dir() -> Path:
    return load_paths().runs_dir / "ensemble"


def load_config(path: Path | None = None) -> dict:
    return json.loads((path or ensemble_dir() / "ensemble.json").read_text(encoding="utf-8"))


class EnsembleAnalyzer:
    def __init__(self, config: dict | None = None, device: str = "cpu", analyzers: dict | None = None):
        self.cfg = config or load_config()
        self.name = "ensemble"
        runs = load_paths().runs_dir
        self.members = self.cfg["members"]
        self.analyzers = analyzers or {m["run"]: Analyzer(runs / m["run"], device) for m in self.members}
        self.reference = forensics.load_reference()
        self.local = self.cfg.get("local") or {"window_s": LOCAL_S, "A": self.cfg.get("A", 1.0),
                                               "B": self.cfg.get("B", 0.0)}

    # ---------------------------------------------------------------------------------
    def member_prob(self, m: dict, z) -> np.ndarray:
        """Member logit -> member P(fake), with the member calibration of ensemble.json."""
        return sigmoid(m["a"] * np.asarray(z, float) + m["b"])

    def _weighted(self, member_logodds) -> np.ndarray:
        z = np.asarray(member_logodds, float).reshape(len(self.members), -1)
        w = np.array([m["weight"] for m in self.members])[:, None]
        clip = self.cfg.get("clip")
        return (w * (np.clip(z, -clip, clip) if clip else z)).sum(0)

    def fuse(self, member_logodds: np.ndarray) -> np.ndarray:
        """(members, ...) calibrated member log-odds -> merged P(fake) of a whole file."""
        c = self.cfg
        if "A" in c:                                   # clipped, constrained, calibrated (binary fit)
            return sigmoid(c["A"] * self._weighted(member_logodds) + c["B"])
        return sigmoid(self._weighted(member_logodds) + c.get("bias", 0.0))   # older linear fusion

    def fuse_local(self, member_logodds: np.ndarray) -> np.ndarray:
        """Like fuse(), with the window-level calibration of the local timeline (NaN stays NaN)."""
        return sigmoid(self.local["A"] * self._weighted(member_logodds) + self.local["B"])

    def _fuse_subset(self, member_logodds: np.ndarray, idx: list[int], cal: dict) -> np.ndarray:
        """Fusion over the available members only, weights renormalised to the full-set total."""
        z = np.asarray(member_logodds, float).reshape(len(self.members), -1)[idx]
        w = np.array([self.members[i]["weight"] for i in idx])[:, None]
        w = w * sum(m["weight"] for m in self.members) / w.sum()
        clip = self.cfg.get("clip")
        s = (w * (np.clip(z, -clip, clip) if clip else z)).sum(0)
        return sigmoid(cal.get("A", 1.0) * s + cal.get("B", 0.0))

    @staticmethod
    def vote(p: float) -> str:
        return "Fake" if p >= DECISION_P else "Real"

    def decide(self, p: float, member_p: list[float] | None = None, ood: bool = False) -> tuple[str, None]:
        """Two-way verdict from the merged P(fake). The members enter only through the fusion
        (no gate or veto), and OOD input is flagged with a warning, not a third verdict."""
        return ("Fake" if p >= DECISION_P else "Real"), None

    # ---------------------------------------------------------------------------------
    def analyze(self, x: np.ndarray, explain: bool = True, progress=None) -> dict:
        """progress: optional callback(fraction 0-1, text) for a UI progress bar."""
        n = len(x)
        dur = n / SR
        spans, step, win = local_grid(n, self.local["window_s"])
        speech = speech_windows(x, spans)
        # progress: stages weighted by their expected CPU time (windows x per-window cost)
        weights = []
        for m in self.members:
            an = self.analyzers[m["run"]]
            c = window_cost(m["run"])
            n_file = len(an.windows(x)) if hasattr(an, "windows") else 1
            weights += [c * n_file, c * len(spans)]
        if explain:
            weights.append(EXPLAIN_COST_PER_S * dur)
        total_w = sum(weights) or 1.0
        stage = 0

        def report(text: str, sub: float = 0.0) -> None:
            if progress is not None:
                f = (sum(weights[:stage]) + sub * weights[min(stage, len(weights) - 1)]) / total_w
                progress(min(f, 1.0), f"Step {min(stage + 1, len(weights))} of {len(weights)} · {text}")

        def tracker(text: str):
            return None if progress is None else (lambda f: report(f"{text} ({100 * f:.0f}%)", f))

        mres, mz, mlocal = [], [], []
        for m in self.members:
            an = self.analyzers[m["run"]]
            txt = f"{m['label']}: scoring the whole file"
            report(txt + "…")
            try:
                r = an.analyze(x, on_batch=tracker(txt)) if progress is not None else an.analyze(x)
                stage += 1
                p_m = float(self.member_prob(m, r["logit"]))
                if not np.isfinite(p_m):
                    raise ValueError("non-finite member score")
                tl = [dict(w, p_fake=float(self.member_prob(m, w["logit"]))) for w in r["timeline"]]
                txt = f"{m['label']}: scoring {len(spans)} local windows"
                report(txt + "…")
                raw = np.asarray(an.score_segments(x, spans, on_batch=tracker(txt)) if progress is not None
                                 else an.score_segments(x, spans), float)
                stage += 1
            except Exception as e:  # noqa: BLE001  (a failed member is reported, never counted as REAL)
                if isinstance(e, MemoryError) or type(e).__name__ == "OutOfMemoryError":
                    raise                     # the app retries the whole file on the CPU
                stage = 2 * (len(mres) + 1)
                mres.append({"run": m["run"], "label": m["label"], "p_fake": None, "result": None, "timeline": [],
                             "error": f"{e.__class__.__name__}: {e}"})
                mz.append(np.nan)
                mlocal.append(np.full(len(spans), np.nan))
                continue
            mlocal.append(np.where(np.isfinite(raw), logit(self.member_prob(m, np.nan_to_num(raw))), np.nan))
            mres.append({"run": m["run"], "label": m["label"], "p_fake": p_m, "result": r, "timeline": tl})
            mz.append(logit(p_m))
        ok_members = [mr["p_fake"] is not None for mr in mres]
        if not any(ok_members):
            raise RuntimeError("every ensemble member failed: " + "; ".join(mr["error"] for mr in mres))
        # a local window counts only if it carries speech and every available member scored it
        ok_idx = [i for i, ok in enumerate(ok_members) if ok]
        valid = speech & np.isfinite(np.array(mlocal)[ok_idx]).all(0)
        mstep = []
        for zl in mlocal:
            t0, zs = step_logodds(spans, zl, n, step, valid)
            mstep.append(zs)
        if all(ok_members):
            p = float(self.fuse(np.array(mz))[0])
            step_p = self.fuse_local(np.array(mstep))
        else:   # a member failed: renormalised weights over the rest (uncalibrated; the decision is INCONCLUSIVE)
            p = float(self._fuse_subset(np.array(mz), ok_idx, self.cfg)[0])
            step_p = self._fuse_subset(np.array(mstep), ok_idx, self.local)
        scored = np.isfinite(step_p)
        # OOD: the model-free signal check, or a majority of the members' embedding checks
        first = next(mr for mr in mres if mr["result"] is not None)
        sig_reasons = [s for s in first["result"]["ood_reasons"] if "internal features" not in s]
        emb_flags = [any("internal features" in s for s in mr["result"]["ood_reasons"]) for mr in mres
                     if mr["result"] is not None]
        emb_ood = sum(emb_flags) > len(mres) / 2
        ood = bool(sig_reasons) or emb_ood
        ood_reasons = sig_reasons + (["most models' internal features are far from anything seen in training"]
                                     if emb_ood else [])
        verdict, _ = self.decide(p)
        # member lines: each member's own window-level P(fake) (display and segment evidence only)
        members = [{"run": mr["run"], "label": mr["label"], "p_fake": mr["p_fake"],
                    "vote": None if mr["p_fake"] is None else self.vote(mr["p_fake"]),
                    "available": mr["p_fake"] is not None, "error": mr.get("error"),
                    "weight": m["weight"], "timeline": mr["timeline"],
                    "local": [{"start_s": a / SR, "end_s": b / SR, "p_fake": float(sigmoid(z)) if ok else None}
                              for (a, b), z, ok in zip(spans, zl, valid)],
                    "per_second": [float(v) if np.isfinite(v) else None for v in sigmoid(zs)]}
                   for mr, m, zs, zl in zip(mres, self.members, mstep, mlocal)]
        per_step = [{"start_s": float(t), "end_s": float(min(t + step, dur)),
                     "p_fake": float(q) if ok else None, "speech": bool(ok)}
                    for t, q, ok in zip(t0, step_p, scored)]
        res = {
            "verdict": verdict, "p_fake": p, "p_real": 1.0 - p, "logit": float(logit(p)), "gate_reason": None,
            "out_of_distribution": ood, "ood_reasons": ood_reasons,
            "signal": first["result"]["signal"], "duration_s": dur, "analysed_region_s": [0.0, dur],
            "members": members,
            "per_second": per_step,                 # timeline steps (name kept for API compatibility)
            "step_s": step, "local_window_s": win, "n_local_windows": len(spans),
            "n_local_windows_scored": int(valid.sum()), "time_resolved": int(valid.sum()) > 1,
            "thresholds": {"t_real": DECISION_P, "t_fake": DECISION_P, "decision": DECISION_P},
            "calibrated": True, "model": "ensemble",
        }
        if not all(ok_members):
            res["calibrated"] = False
        res["timeline"] = per_step
        res["regions"] = timeline_regions(t0, step_p, step, dur)
        res["segments"] = [{"start_s": r["start_s"], "end_s": r["end_s"], "duration_s": r["end_s"] - r["start_s"],
                            "p_fake": r["p_fake_mean"],
                            "p_fake_peak": max(w["p_fake"] for w in per_step if w["p_fake"] is not None
                                               and w["start_s"] < r["end_s"] and w["end_s"] > r["start_s"]),
                            "severity": segments.severity(r["p_fake_mean"])}
                           for r in res["regions"] if r["label"] == "Fake"]
        res["suspicious_segments"] = sorted(({"start_s": s["start_s"], "end_s": s["end_s"], "p_fake": s["p_fake"]}
                                             for s in res["segments"]), key=lambda w: -w["p_fake"])[:3]
        if explain:
            report("Measuring flagged segments and writing the explanation…")
            self._explain(x, res)
            stage += 1
        report("Done")
        return res

    def _explain(self, x: np.ndarray, res: dict) -> None:
        # saliency: the VoiceGuard member with the largest weight, on its most decisive window
        vg = [m for m in self.members if m["run"].startswith("vg_")]
        vg = [m for m in vg if any(mm["run"] == m["run"] and mm["available"] for mm in res["members"])]
        if vg:
            m = max(vg, key=lambda m: abs(m["weight"]))
            an = self.analyzers[m["run"]]
            tl = next(mm["timeline"] for mm in res["members"] if mm["run"] == m["run"])
            top = max(tl, key=lambda w: w["p_fake"]) if res["verdict"] == "Fake" else \
                min(tl, key=lambda w: w["p_fake"])
            a, b = int(top["start_s"] * SR), int(top["end_s"] * SR)
            sal = an.saliency(x[a:b])
            res["saliency"] = {"model": m["label"], "start_s": top["start_s"], "end_s": top["end_s"],
                               "source_s": (b - a) / SR, "map": sal.get("map"), "feature": sal.get("feature")}
            if sal.get("map") is not None:
                res["bands"] = forensics.band_shares(sal["map"][:, : max(1, int(np.ceil((b - a) / 256)))])
        meas = forensics.measure(x)
        res["measurements"] = meas
        res["observations"] = forensics.observations(meas, self.reference)
        if res["segments"]:
            ff = segments.frame_features(x)
            for s in res["segments"]:
                s["evidence"] = segments.segment_evidence(x, s, ff, res, self.reference)
        res["explanation"] = forensics.statements(res)

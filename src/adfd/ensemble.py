"""Merged detector: the project's best model plus the two VoiceGuard checkpoints.

    ens = EnsembleAnalyzer()                      # reads runs/ensemble/ensemble.json
    res = ens.analyze(load_audio("clip.wav"))     # verdict, P(fake), members, regions, explanation

How a file is decided
  1. every member scores the file with its own windows (E7a: 4 s on the speech region;
     VoiceGuard: 5 s, as each was trained) and its own calibration;
  2. member logits are fused with fixed weights (scripts/fit_ensemble.py, fitted on
     ASVspoof dev data only) into one P(fake) for the file and one per second;
  3. the verdict is cautious by design, to keep false detections rare:
       - input that does not look like speech (OOD)         -> Uncertain
       - Fake  only if fused P(fake) >= t_fake AND at least `min_agree` members vote Fake
       - Real  only if fused P(fake) <= t_real AND no member votes Fake with P >= `veto_p`
       - everything else                                    -> Uncertain
     So a single over-confident model can never produce a Fake (or a Real) on its own.
  4. explanation: Real / Uncertain / Fake regions over time, mel-spectrogram saliency
     of the most suspicious window with per-band shares, and model-free acoustic
     observations (adfd.forensics).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from adfd import forensics
from adfd.calibration import sigmoid
from adfd.inference import SR, Analyzer
from adfd.paths import load_paths

STEP_S = 1.0


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def ensemble_dir() -> Path:
    return load_paths().runs_dir / "ensemble"


def load_config(path: Path | None = None) -> dict:
    return json.loads((path or ensemble_dir() / "ensemble.json").read_text(encoding="utf-8"))


def _per_second(timeline: list[dict], n_sec: int, field: str = "p_fake") -> np.ndarray:
    """Mean of the member's window scores (as logits) overlapping each 1 s step."""
    out = np.full(n_sec, np.nan)
    if not timeline:
        return out
    starts = np.array([w["start_s"] for w in timeline])
    ends = np.array([w["end_s"] for w in timeline])
    z = logit([w[field] for w in timeline])
    for k in range(n_sec):
        a, b = k * STEP_S, (k + 1) * STEP_S
        m = (starts < b) & (ends > a)
        if m.any():
            out[k] = z[m].mean()
    # seconds no window covers (silence the model skipped): nearest covered value
    if np.isnan(out).all():
        return np.full(n_sec, z.mean())
    idx = np.where(~np.isnan(out))[0]
    for k in np.where(np.isnan(out))[0]:
        out[k] = out[idx[np.argmin(np.abs(idx - k))]]
    return out


class EnsembleAnalyzer:
    def __init__(self, config: dict | None = None, device: str = "cpu", analyzers: dict | None = None):
        self.cfg = config or load_config()
        self.name = "ensemble"
        runs = load_paths().runs_dir
        self.members = self.cfg["members"]
        self.analyzers = analyzers or {m["run"]: Analyzer(runs / m["run"], device) for m in self.members}
        self.reference = forensics.load_reference()

    # ---------------------------------------------------------------------------------
    def member_prob(self, m: dict, z) -> np.ndarray:
        """Member logit -> member P(fake), with the member calibration of ensemble.json."""
        return sigmoid(m["a"] * np.asarray(z, float) + m["b"])

    def fuse(self, member_logits: np.ndarray) -> np.ndarray:
        """(members, ...) calibrated member log-odds -> fused P(fake)."""
        w = np.array([m["weight"] for m in self.members])[:, None]
        return sigmoid((w * member_logits.reshape(len(self.members), -1)).sum(0) + self.cfg["bias"])

    def decide(self, p: float, member_p: list[float], ood: bool) -> tuple[str, str | None]:
        c = self.cfg
        votes = ["Fake" if q >= c["vote_fake"] else "Real" if q <= c["vote_real"] else "Uncertain" for q in member_p]
        n_fake = votes.count("Fake")
        if ood:
            return "Uncertain", "The input failed the speech check, so the models' scores are not meaningful."
        if p >= c["t_fake"]:
            if n_fake >= c["min_agree"]:
                return "Fake", None
            return "Uncertain", (f"The fused score is high, but only {n_fake} of {len(member_p)} models vote fake "
                                 f"(at least {c['min_agree']} must agree before the app says Fake).")
        if p <= c["t_real"]:
            strong = [q for q in member_p if q >= c["veto_p"]]
            if not strong:
                return "Real", None
            return "Uncertain", ("The fused score is low, but one model is confident the audio is fake "
                                 f"(P = {max(strong):.2f}), so the app does not call it Real.")
        return "Uncertain", None

    # ---------------------------------------------------------------------------------
    def analyze(self, x: np.ndarray, explain: bool = True) -> dict:
        dur = len(x) / SR
        n_sec = max(1, int(np.ceil(dur / STEP_S)))
        mres, mz, msec = [], [], []
        for m in self.members:
            an = self.analyzers[m["run"]]
            r = an.analyze(x)
            # raw member logit, then the ensemble's member calibration
            z_raw = r["logit"]
            win_logits = np.array([w["logit"] for w in r["timeline"]])
            p_m = float(self.member_prob(m, z_raw))
            tl = [dict(w, p_fake=float(self.member_prob(m, l))) for w, l in zip(r["timeline"], win_logits)]
            mres.append({"run": m["run"], "label": m["label"], "p_fake": p_m, "result": r, "timeline": tl})
            mz.append(logit(p_m))
            msec.append(_per_second(tl, n_sec))
        p = float(self.fuse(np.array(mz))[0])
        sec_p = self.fuse(np.array(msec))
        ood_reasons = sorted({s for mr in mres for s in mr["result"]["ood_reasons"]})
        # the signal check is model-free and shared; embedding checks are per model and can
        # disagree, so only the signal check and a majority of embedding checks count
        sig_ood = bool(mres[0]["result"]["ood_reasons"] and any(
            "internal features" not in s for s in mres[0]["result"]["ood_reasons"]))
        emb_flags = [any("internal features" in s for s in mr["result"]["ood_reasons"]) for mr in mres]
        ood = sig_ood or sum(emb_flags) > len(mres) / 2
        if not ood:
            ood_reasons = [s for s in ood_reasons if "internal features" not in s] if not sig_ood else ood_reasons
        verdict, gate = self.decide(p, [mr["p_fake"] for mr in mres], ood)
        c = self.cfg
        members = []
        for mr, sec in zip(mres, msec):
            q = mr["p_fake"]
            members.append({"run": mr["run"], "label": mr["label"], "p_fake": q,
                            "vote": "Fake" if q >= c["vote_fake"] else "Real" if q <= c["vote_real"] else "Uncertain",
                            "timeline": mr["timeline"], "per_second": [float(v) for v in sigmoid(sec)]})
        times = np.arange(n_sec) * STEP_S
        res = {
            "verdict": verdict, "p_fake": p, "logit": float(logit(p)), "gate_reason": gate,
            "out_of_distribution": ood, "ood_reasons": ood_reasons if ood else [],
            "signal": mres[0]["result"]["signal"], "duration_s": dur, "analysed_region_s": [0.0, dur],
            "members": members, "per_second": [{"start_s": float(t), "end_s": float(min(t + STEP_S, dur)),
                                                "p_fake": float(q)} for t, q in zip(times, sec_p)],
            "thresholds": {"t_real": c["t_real"], "t_fake": c["t_fake"]}, "calibrated": True, "model": "ensemble",
        }
        res["timeline"] = [dict(w, logit=float(logit(w["p_fake"]))) for w in res["per_second"]]
        res["regions"] = forensics.regions(times, sec_p, c["t_real"], c["t_fake"], STEP_S)
        for r in res["regions"]:
            r["end_s"] = min(r["end_s"], dur)
        res["suspicious_segments"] = sorted(
            ({"start_s": r["start_s"], "end_s": r["end_s"], "p_fake": r["p_fake_mean"]}
             for r in res["regions"] if r["label"] == "Fake"), key=lambda w: -w["p_fake"])[:3]
        if explain:
            self._explain(x, res)
        return res

    def _explain(self, x: np.ndarray, res: dict) -> None:
        # saliency: the VoiceGuard member with the largest weight, on its most suspicious window
        vg = [m for m in self.members if m["run"].startswith("vg_")]
        if vg:
            m = max(vg, key=lambda m: abs(m["weight"]))
            an = self.analyzers[m["run"]]
            tl = next(mm["timeline"] for mm in res["members"] if mm["run"] == m["run"])
            top = max(tl, key=lambda w: w["p_fake"]) if res["verdict"] != "Real" else \
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
        res["explanation"] = forensics.statements(res)

"""Calibrate the merged detector's time-local timeline (short analysis windows).

The file verdict is calibrated on whole files (scripts/ensemble/fit_ensemble_binary.py). The timeline
scores short local windows (adfd.ensemble.LOCAL_S), which the members score differently
(e.g. VoiceGuard improved calls most 2 s windows of real ASVspoof speech fake), so the
file calibration cannot be reused there. This script scores short windows of the TUNING
data with all three members and fits, for the same member weights and clipping:
    local P(fake) = sigmoid(A_loc * sum_i w_i * clip(z_i) + B_loc)
with the threshold (P = 0.5) at the best mean balanced accuracy over the tuning domains and
a domain- and class-balanced Platt slope (as for the file verdict).

Tuning windows (never test data):
  - ASVspoof 2019 LA dev: N clips per class, one random window each, anywhere in the file
    (the app's local windows also cover pauses);
  - DEEP-VOICE non-test persons (deep_voice_xd train + dev): N windows per class, one random
    window inside each 4 s manifest window.
Reports the window-level EER and balanced accuracy for each candidate window length, and
writes the chosen one into runs/ensemble/ensemble.json under "local".

    python scripts/ensemble/fit_local_calibration.py [--n 1500] [--lengths 1.5 2 3] [--device cuda]
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from adfd.calibration import sigmoid
from adfd.data.audio_cache import AudioCache
from adfd.ensemble import EnsembleAnalyzer, ensemble_dir, logit
from adfd.inference import SR
from adfd.metrics import eer_from_labels
from adfd.paths import configure_caches, load_paths


def sample_windows(n: int, length: float, seed: int) -> dict:
    """{domain: (list of float32 windows, labels)} from the tuning data."""
    paths = load_paths()
    rng = np.random.default_rng(seed)
    L = int(length * SR)
    out = {}
    asv = pd.read_csv(paths.manifest("asvspoof2019_la"), keep_default_na=False)
    dev = asv[asv["split"] == "dev"]
    cache = AudioCache(paths.cache_dir, "asvspoof2019_la")
    rows = pd.concat([dev[dev.label == c].sample(min(n, (dev.label == c).sum()), random_state=seed) for c in (0, 1)])
    xs, ys = [], []
    for r in rows.itertuples():
        tot = cache.length(r.recording_id)
        a = int(rng.integers(0, max(1, tot - L + 1)))
        xs.append(cache.read(r.recording_id, a, min(tot, a + L)).astype(np.float32) / 32768.0)
        ys.append(r.label)
    out["ASVspoof dev"] = (xs, np.array(ys))
    dv = pd.read_csv(paths.manifest("deep_voice_xd"), keep_default_na=False)
    dv = dv[dv["split"].isin(["train", "dev"])]
    cache = AudioCache(paths.cache_dir, "deep_voice")
    rows = pd.concat([dv[dv.label == c].sample(min(n, (dv.label == c).sum()), random_state=seed) for c in (0, 1)])
    xs, ys = [], []
    for r in rows.itertuples():
        s, e = int(r.start_s * SR), int(r.end_s * SR)
        a = s + int(rng.integers(0, max(1, e - s - L + 1)))
        xs.append(cache.read(r.recording_id, a, min(e, a + L)).astype(np.float32) / 32768.0)
        ys.append(r.label)
    out["DEEP-VOICE tuning persons"] = (xs, np.array(ys))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=1500, help="windows per class and domain")
    ap.add_argument("--lengths", type=float, nargs="+", default=[1.5, 2.0, 3.0])
    ap.add_argument("--choose", type=float, help="window length to write (default: best mean balanced accuracy)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-write-config", action="store_true")
    args = ap.parse_args()
    configure_caches()
    ens = EnsembleAnalyzer(device=args.device)
    w = np.array([m["weight"] for m in ens.members])
    clip = ens.cfg["clip"]
    results = {}
    for L in args.lengths:
        data = sample_windows(args.n, L, seed=0)
        s_dom = {}
        for dom, (xs, y) in data.items():
            z = []
            for m in ens.members:
                an = ens.analyzers[m["run"]]
                raw = an._score([an._prep(x) for x in xs], batch=32)[0]
                z.append(logit(ens.member_prob(m, raw)))
            z = np.array(z)
            ok = np.isfinite(z).all(0)          # silent windows can give NaN (zero-variance log-mel)
            if (~ok).any():
                print(f"  {dom} {L} s: dropped {(~ok).sum()} windows with non-finite scores (silence)")
            z, y = z[:, ok], y[ok]
            s_dom[dom] = ((w[:, None] * np.clip(z, -clip, clip)).sum(0), y, z)
        thr = np.linspace(-clip, clip, 161)
        bacc = np.zeros_like(thr)
        margin = ens.cfg.get("margin", 0.2)       # same no-single-member rule as the file verdict
        feasible = np.abs(thr) <= clip * (1 - 2 * w.max()) - margin + 1e-9
        for s, y, _ in s_dom.values():
            fake = s[None] >= thr[:, None]
            bacc += 0.5 * (fake[:, y == 1].mean(1) + (~fake[:, y == 0]).mean(1)) / len(s_dom)
        bacc[~feasible] = -1
        t = float(thr[int(np.argmax(bacc))])
        S, Y, W = [], [], []
        for s, y, _ in s_dom.values():
            for c in (0, 1):
                k = y == c
                S.append(s[k]), Y.append(y[k]), W.append(np.full(k.sum(), 1.0 / k.sum()))
        S, Y, W = map(np.concatenate, (S, Y, W))
        A = float(LogisticRegression(C=1e6, max_iter=1000).fit((S - t)[:, None], Y, sample_weight=W).coef_[0, 0])
        rep = {"window_s": L, "A": A, "B": -A * t, "mean_balanced_acc_pct": 100 * float(bacc.max()), "domains": {}}
        for dom, (s, y, z) in s_dom.items():
            p = sigmoid(A * (s - t))
            fake = p >= 0.5
            rep["domains"][dom] = {
                "eer_pct": 100 * float(eer_from_labels(-s, y)[0]),
                "balanced_acc_pct": 50 * float(fake[y == 1].mean() + (~fake)[y == 0].mean()),
                "real_called_fake_pct": 100 * float(fake[y == 0].mean()),
                "fake_called_real_pct": 100 * float((~fake)[y == 1].mean()),
                "member_eer_pct": [100 * float(eer_from_labels(-zi, y)[0]) for zi in z]}
        results[L] = rep
        print(f"L={L} s: mean balanced acc {rep['mean_balanced_acc_pct']:.1f}%", flush=True)
        for dom, d in rep["domains"].items():
            print(f"   {dom:26s} EER {d['eer_pct']:5.1f}  bal.acc {d['balanced_acc_pct']:5.1f}  real->FAKE "
                  f"{d['real_called_fake_pct']:5.1f}  fake->REAL {d['fake_called_real_pct']:5.1f}  member EER "
                  f"{[round(v, 1) for v in d['member_eer_pct']]}", flush=True)
    best = results[args.choose] if args.choose else max(results.values(), key=lambda r: r["mean_balanced_acc_pct"])
    print(f"chosen local window: {best['window_s']} s")
    if not args.no_write_config:
        f = ensemble_dir() / "ensemble.json"
        cfg = json.loads(f.read_text(encoding="utf-8"))
        cfg["local"] = {"window_s": best["window_s"], "A": best["A"], "B": best["B"],
                        "fitted_on": f"{args.n} windows per class: ASVspoof dev + DEEP-VOICE non-test persons",
                        "report": {str(k): v for k, v in results.items()}}
        f.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

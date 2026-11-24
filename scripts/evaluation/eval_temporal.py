"""Fit the temporal model's calibration on validation frames, then evaluate it against E7a's sliding timeline.

    python scripts/evaluation/eval_temporal.py --run runs/t1_wavlm_crnn_temporal_s42 --fit            # val speakers only
    python scripts/evaluation/eval_temporal.py --run runs/t1_wavlm_crnn_temporal_s42 --split test --limit 6000
    python scripts/evaluation/eval_temporal.py --run ... --split test --long 300    # 10-15 s same-speaker mixtures

--fit       scores PartialSpoof dev validation speakers (never trained on) and writes
            <run>/temporal_calibration.json: Platt scaling of frame logits (and, for reference, the validation EER
            threshold). Frames are labelled by argmax as in the app: FAKE if P(fake) > P(real), ties REAL.
--split     val | test (PartialSpoof eval: unseen speakers, unseen attacks A07-A19). Never used for fitting.
--limit     seeded sample stratified by kind (real / partial / fake).
--long N    also builds N recordings of 10-15 s by joining utterances of one speaker from the same split (each
            with at least one real and one partially fake utterance), labels concatenated exactly; seeded.
Baseline: E7a's time-local analysis as the app showed it before (adfd.single: 4-s windows every 0.5 s, step
score for each 0.16-s frame centre), FAKE at its 50% line.
Writes experiments/temporal_t1/<split>[_long].json and prints a summary.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from adfd.data.partialspoof import FRAME_S, HOP, full_utterance, load_seglab
from adfd.metrics import compute_eer
from adfd.paths import REPO_ROOT, load_paths
from adfd.temporal import TemporalAnalyzer, sigmoid
from adfd.temporal_metrics import frame_metrics, segment_counts, segment_summary, short_recall

LAB = str(load_paths().dataset("partialspoof") / "segment_labels" / "{}_seglab_0.16.npy")
OUT = REPO_ROOT / "experiments" / "temporal_t1"


def sample(df: pd.DataFrame, n: int | None, seed: int) -> pd.DataFrame:
    if not n or n >= len(df):
        return df
    return df.groupby("kind", group_keys=False).sample(frac=n / len(df), random_state=seed).reset_index(drop=True)


def long_mixes(df: pd.DataFrame, n: int, seed: int, lo: float = 10.0, hi: float = 15.0) -> list[list[str]]:
    rng = random.Random(seed)
    by = {s: g for s, g in df.groupby("speaker")}
    spk = sorted(by)
    out, used = [], set()
    tries = 0
    while len(out) < n and tries < n * 50:
        tries += 1
        g = by[rng.choice(spk)]
        reals = [u for u in g[g.kind == "real"].utt_id if u not in used]
        parts = [u for u in g[g.kind != "real"].utt_id if u not in used]
        if not reals or not parts:
            continue
        pick = [rng.choice(reals), rng.choice(parts)]
        dur = g.set_index("utt_id").duration_s
        pool = [u for u in g.utt_id if u not in used and u not in pick]
        rng.shuffle(pool)
        while sum(dur[u] for u in pick) < lo and pool:
            pick.append(pool.pop())
        if not lo <= sum(dur[u] for u in pick) <= hi:
            continue
        rng.shuffle(pick)
        used.update(pick)
        out.append(pick)
    return out


def e7a_frames(an, x: np.ndarray, n: int) -> np.ndarray:
    res = an.analyze(x, explain=False)
    steps = res["per_second"]
    p = np.zeros(n)
    for k in range(n):
        c = (k + 0.5) * FRAME_S
        s = next((w for w in steps if w["start_s"] <= c < w["end_s"]), steps[-1])
        p[k] = 0.0 if s["p_fake"] is None else s["p_fake"]
    return p, res


def score_items(items, ta, e7a, labels_of, audio_of):
    rows = []
    t0 = time.time()
    for i, it in enumerate(items, 1):
        x = audio_of(it)
        y = labels_of(it)
        z, clips, n = ta.logits(x)
        k = min(n, len(y))
        row = {"id": it if isinstance(it, str) else "+".join(it), "y": y[:k], "z": z[:k],
               "clip_z": float(np.mean(clips))}
        if e7a is not None:
            pe, res = e7a_frames(e7a, x, k)
            row.update(e7a=pe, e7a_clip=float(res["p_fake"]))
        rows.append(row)
        if i % 250 == 0:
            print(f"  {i}/{len(items)} ({i / (time.time() - t0):.1f}/s)", flush=True)
    return rows


def fit(rows) -> dict:
    z = np.concatenate([r["z"] for r in rows])
    y = np.concatenate([r["y"] for r in rows]).astype(int)
    a, b = 1.0, 0.0                                      # Platt scaling by Newton steps on the 2 parameters
    for _ in range(50):
        p = sigmoid(a * z + b)
        g = np.array([np.sum((p - y) * z), np.sum(p - y)])
        w = p * (1 - p) + 1e-9
        H = np.array([[np.sum(w * z * z), np.sum(w * z)], [np.sum(w * z), np.sum(w)]]) + 1e-6 * np.eye(2)
        step = np.linalg.solve(H, g)
        a, b = a - step[0], b - step[1]
        if np.abs(step).max() < 1e-7:
            break
    p = sigmoid(a * z + b)
    eer, thr = compute_eer(-p[y == 0], -p[y == 1])        # accept-as-real when -p >= thr
    t = float(-thr)
    clip_y = np.array([int(r["y"].max() > 0) for r in rows])
    clip_z = np.array([r["clip_z"] for r in rows])
    ceer, cthr = compute_eer(-clip_z[clip_y == 0], -clip_z[clip_y == 1])
    return {"a": float(a), "b": float(b), "eer_threshold": t, "fitted": True,
            "clip_logit_threshold": float(-cthr),
            "val": {"frames": int(len(y)), "frame_eer": float(eer),
                    "argmax_accuracy": float(np.mean((p > 0.5) == (y == 1))), "clip_eer": float(ceer)}}


def report(rows, cal, with_e7a: bool) -> dict:
    y = np.concatenate([r["y"] for r in rows]).astype(int)
    out = {"utterances": len(rows), "frames": int(len(y)), "fake_frame_share": float(y.mean())}
    systems = {"t1": [sigmoid(cal["a"] * r["z"] + cal["b"]) for r in rows]}
    if with_e7a:
        systems["e7a_sliding"] = [r["e7a"] for r in rows]
    for name, ps in systems.items():
        p = np.concatenate(ps)
        thr = 0.5                                    # argmax: FAKE only if P(fake) > P(real), as in the app
        pred = [q > thr for q in ps]
        res = {"frame_eer": float(compute_eer(-p[y == 0], -p[y == 1])[0]) if 0 < y.mean() < 1 else None,
               "threshold": thr, "frame": frame_metrics(y, p > thr),
               "segment": segment_summary([segment_counts(r["y"], q) for r, q in zip(rows, pred)]),
               "short_fake_recall": short_recall([(r["y"], q) for r, q in zip(rows, pred)])}
        res["threshold_sweep"] = [{"threshold": round(float(t), 2), **{k: v for k, v in frame_metrics(y, p >= t).items()
                                                                     if k in ("precision", "recall", "f1", "fpr", "fnr")}}
                                  for t in np.arange(0.05, 0.96, 0.05)]
        clip_y = np.array([int(r["y"].max() > 0) for r in rows])
        if name == "t1":
            clip_pred = np.array([r["clip_z"] >= cal.get("clip_logit_threshold", 0.0) for r in rows])
            seg_pred = np.array([bool(q.any()) for q in pred])
            res["clip_from_segments_accuracy"] = float(np.mean(seg_pred == clip_y))
        else:
            clip_pred = np.array([r["e7a_clip"] >= 0.5 for r in rows])
        res["clip_accuracy"] = float(np.mean(clip_pred == clip_y))
        res["clip_confusion"] = frame_metrics(clip_y, clip_pred)["confusion"]
        out[name] = res
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--long", type=int, default=0)
    ap.add_argument("--no-e7a", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    paths = load_paths()
    man = pd.read_csv(paths.manifest("partialspoof"), keep_default_na=False)
    split = "val" if args.fit else args.split
    df = man[man["split"] == split].reset_index(drop=True)
    if df.empty:
        print(f"no rows for split {split!r} (eval audio not extracted yet?)")
        return 1
    seglab = load_seglab(LAB.format("dev" if split == "val" else "eval"))
    path = dict(zip(df.utt_id, df.path))
    ta = TemporalAnalyzer(args.run, args.device)
    e7a = None
    if not args.fit and not args.no_e7a:
        from adfd import registry
        e7a = registry.make_analyzer(registry.MODEL, args.device)
    OUT.mkdir(parents=True, exist_ok=True)
    run = Path(args.run)

    if args.fit:
        rows = score_items(list(sample(df, args.limit, args.seed).utt_id), ta, None, lambda u: seglab[u],
                           lambda u: full_utterance(path[u]))
        cal = fit(rows)
        (run / "temporal_calibration.json").write_text(json.dumps(cal, indent=2), encoding="utf-8")
        print(json.dumps(cal, indent=2))
        return 0

    cal = ta.cal
    if not cal.get("fitted"):
        print("run --fit first")
        return 1
    if args.long:
        mixes = long_mixes(df, args.long, args.seed)
        rows = score_items(mixes, ta, e7a, lambda m: np.concatenate([seglab[u] for u in m]),
                           lambda m: np.concatenate([np.pad(full_utterance(path[u]), (0, len(seglab[u]) * HOP -
                                                    len(full_utterance(path[u]))))[: len(seglab[u]) * HOP]
                                                    for u in m]))
        tag = f"{split}_long"
    else:
        rows = score_items(list(sample(df, args.limit, args.seed).utt_id), ta, e7a, lambda u: seglab[u],
                           lambda u: full_utterance(path[u]))
        tag = split
    rep = report(rows, cal, e7a is not None)
    rep.update(split=tag, run=run.name, limit=args.limit, long=args.long, seed=args.seed, calibration=cal)
    (OUT / f"{tag}.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    for name in ("t1", "e7a_sliding"):
        if name in rep:
            r = rep[name]
            print(f"{tag} {name}: frame EER {r['frame_eer'] and r['frame_eer'] * 100:.2f}%  frame F1 "
                  f"{r['frame']['f1']:.3f}  segment F1 {r['segment']['f1']:.3f}  clip acc {r['clip_accuracy']:.3f}  "
                  f"short<0.5s recall {r['short_fake_recall']['< 0.5 s']['recall']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

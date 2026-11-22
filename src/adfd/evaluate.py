"""Score a trained run on the test sets and compute all Phase 1 metrics.

Usage:
    python -m adfd.evaluate --run runs/<run_name>
    python -m adfd.evaluate --run runs/<run_name> --sets in_the_wild:test

Writes per-utterance scores to the run folder and metrics to
runs/<run>/metrics.json, plus experiments/<run>/metrics.json and summary.md
(small files tracked in git).

Decision threshold for accuracy/precision/recall: the dev EER threshold saved
at training time. Test sets never influence any threshold.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from adfd.data.audio_cache import AudioCache
from adfd.data.datasets import CropDataset, make_loader
from adfd.engine import score_loader
from adfd.data.silence import load_silence
from adfd.metrics import auc, bootstrap_eer, eer_from_labels, load_asv_scores, min_tdcf, threshold_metrics
from adfd.models import build_model, count_parameters
from adfd.paths import REPO_ROOT, configure_caches, load_paths

ASV_SCORES = {
    "dev": "ASVspoof2019_LA_asv_scores/ASVspoof2019.LA.asv.dev.gi.trl.scores.txt",
    "eval": "ASVspoof2019_LA_asv_scores/ASVspoof2019.LA.asv.eval.gi.trl.scores.txt",
}


def _pct(x: float) -> float:
    return round(100 * x, 3) if x == x else x  # keep NaN


def common_metrics(scores, labels, dev_threshold) -> dict:
    eer, thr = eer_from_labels(scores, labels)
    return {
        "n": int(len(labels)),
        "n_bonafide": int((labels == 0).sum()),
        "n_spoof": int((labels == 1).sum()),
        "eer_pct": _pct(eer),
        "eer_threshold": thr,
        "auc": auc(scores, labels),
        "at_dev_threshold": threshold_metrics(scores, labels, dev_threshold),
    }


def asvspoof_metrics(df: pd.DataFrame, split: str, dataset_root: Path) -> dict:
    out = {}
    asv = load_asv_scores(dataset_root / ASV_SCORES[split])
    bona = df.loc[df["label"] == 0, "score"].to_numpy()
    spoof = df.loc[df["label"] == 1, "score"].to_numpy()
    out["min_tdcf"] = round(min_tdcf(bona, spoof, asv), 4)
    per_attack = {}
    for attack, part in df[df["label"] == 1].groupby("attack"):
        per_attack[attack] = _pct(eer_from_labels(np.r_[bona, part["score"]], np.r_[np.zeros(len(bona)), np.ones(len(part))])[0])
    out["per_attack_eer_pct"] = dict(sorted(per_attack.items()))
    return out


def in_the_wild_metrics(df: pd.DataFrame) -> dict:
    per_spk = {}
    for spk, part in df.groupby("speaker"):
        if part["label"].nunique() == 2:
            per_spk[spk] = _pct(eer_from_labels(part["score"], part["label"])[0])
    values = np.array(list(per_spk.values()))
    worst = sorted(per_spk.items(), key=lambda kv: -kv[1])[:5]
    return {
        "speakers_with_both_classes": len(per_spk),
        "per_speaker_eer_pct": {"median": float(np.median(values)) if len(values) else None,
                                "mean": float(values.mean()) if len(values) else None,
                                "worst5": dict(worst)},
    }


def deep_voice_metrics(df: pd.DataFrame, dev_threshold: float) -> dict:
    rec = df.groupby("recording_id").agg(score=("score", "mean"), label=("label", "first"),
                                         windows=("score", "size")).reset_index()
    eer, thr = eer_from_labels(rec["score"].to_numpy(), rec["label"].to_numpy())
    flagged = rec[rec["score"] < dev_threshold]
    return {
        "per_recording": {
            "n": len(rec), "n_bonafide": int((rec["label"] == 0).sum()), "n_spoof": int((rec["label"] == 1).sum()),
            "eer_pct": _pct(eer), "auc": auc(rec["score"], rec["label"]),
            "at_dev_threshold": threshold_metrics(rec["score"], rec["label"], dev_threshold),
            "real_recordings_flagged_fake": sorted(flagged[flagged["label"] == 0]["recording_id"]),
        },
        "mean_score_by_recording": {r.recording_id: round(r.score, 3) for r in rec.itertuples()},
    }


def summary_markdown(run: str, results: dict) -> str:
    lines = [f"# Results: {run}", "",
             f"Model: {results['model']} ({results['parameters']:,} parameters), best dev epoch {results['best_epoch']}.",
             "Score threshold for accuracy/precision/recall: dev EER threshold (spoof = positive class).", "",
             "| Test set | n | EER % | EER 95% CI | min t-DCF | AUC | Accuracy | Precision | Recall |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name, m in results["sets"].items():
        t = m["at_dev_threshold"]
        tdcf = m.get("min_tdcf", "")
        ci = m.get("eer_ci95")
        ci_txt = f"{ci['lo_pct']:.2f}–{ci['hi_pct']:.2f}" if ci else ""
        lines.append(f"| {name} | {m['n']:,} | {m['eer_pct']:.2f} | {ci_txt} | {tdcf} | {m['auc']:.4f} | "
                     f"{t['accuracy'] * 100:.2f} | {t['precision'] * 100:.2f} | {t['recall'] * 100:.2f} |")
        if "per_recording" in m:
            pr = m["per_recording"]
            lines.append(f"| {name} (per recording) | {pr['n']} | {pr['eer_pct']:.2f} |  |  | {pr['auc']:.4f} | "
                         f"{pr['at_dev_threshold']['accuracy'] * 100:.2f} | {pr['at_dev_threshold']['precision'] * 100:.2f} | "
                         f"{pr['at_dev_threshold']['recall'] * 100:.2f} |")
    for name, m in results["sets"].items():
        if "per_attack_eer_pct" in m and name.endswith("eval"):
            lines += ["", f"## {name}: EER per attack (%)", "",
                      "| " + " | ".join(m["per_attack_eer_pct"]) + " |",
                      "|" + "---|" * len(m["per_attack_eer_pct"]),
                      "| " + " | ".join(f"{v:.2f}" for v in m["per_attack_eer_pct"].values()) + " |"]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run folder containing best.pt")
    ap.add_argument("--sets", nargs="*", help="dataset:split entries (default: dev + config test_sets)")
    ap.add_argument("--workers", type=int)
    ap.add_argument("--batch-size", type=int, help="default: the config's train.eval_batch_size, else 64")
    ap.add_argument("--trim-silence", choices=["auto", "on", "off"], default="auto",
                    help="trim leading/trailing silence of whole-file test items (auto = as in training)")
    ap.add_argument("--normalize", choices=["auto", "none", "rms", "peak"], default="auto",
                    help="loudness normalisation of every test crop (auto = as in training)")
    ap.add_argument("--tag", help="suffix for output names, e.g. 'trimmed' -> experiments/<run>__trimmed")
    ap.add_argument("--n-boot", type=int, default=500, help="bootstrap resamples for EER CIs (0 = off)")
    args = ap.parse_args()

    paths = load_paths()
    configure_caches(paths)
    run_dir = Path(args.run).resolve()
    ckpt = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    dev_thr = float(ckpt["dev_threshold"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    amp = bool(cfg["train"]["amp"]) and device.type == "cuda"
    workers = args.workers if args.workers is not None else cfg["data"]["num_workers"]
    sr = cfg["audio"]["sample_rate"]
    crop = int(cfg["audio"]["crop_seconds"] * sr)

    sets = args.sets or ["asvspoof2019_la:dev", *cfg["data"]["test_sets"]]
    trim = {"on": True, "off": False}.get(args.trim_silence, bool(cfg["audio"].get("trim_silence")))
    norm = cfg["audio"].get("normalize") if args.normalize == "auto" else args.normalize
    norm = None if norm in (None, "none") else norm
    suffix = f"__{args.tag}" if args.tag else ""
    results = {"run": run_dir.name, "tag": args.tag, "model": cfg["model"]["name"],
               "frontend": cfg["model"].get("frontend", {}).get("name", "mfcc"),
               "parameters": count_parameters(model), "best_epoch": ckpt["epoch"], "dev_threshold": dev_thr,
               "crop_seconds": cfg["audio"]["crop_seconds"], "trim_silence": trim,
               "normalize": norm, "augment": cfg.get("augment"), "sets": {}}
    manifests = {}
    for entry in sets:
        name, split = entry.split(":")
        if name not in manifests:
            manifests[name] = pd.read_csv(paths.manifest(name), keep_default_na=False)
        df = manifests[name][manifests[name]["split"] == split].reset_index(drop=True)
        cache_name = str(df["dataset"].iloc[0]) if len(df) else name   # e.g. in_the_wild_xd -> in_the_wild cache
        cache = AudioCache(paths.cache_dir, cache_name)
        bounds = load_silence(paths.cache_dir, cache_name) if trim else None
        if trim and bounds is None and not name.startswith("deep_voice"):
            raise FileNotFoundError(f"silence bounds for {name} missing: run scripts/compute_silence.py")
        loader = make_loader(CropDataset(df, cache, crop, False, cfg["audio"]["pad_mode"], sr, speech_bounds=bounds,
                                         normalize=norm),
                             args.batch_size or cfg["train"].get("eval_batch_size", 64), False, workers)
        print(f"Scoring {entry}{' (silence trimmed)' if bounds else ''}: {len(df):,} items ...", flush=True)
        scores, labels = score_loader(model, loader, device, amp)
        keep = [c for c in ("utt_id", "recording_id", "speaker", "attack", "start_s", "end_s") if c in df]
        scored = df[keep].assign(label=labels, score=scores)
        scored.to_csv(run_dir / f"scores_{name}_{split}{suffix}.csv", index=False)

        m = common_metrics(scores, labels, dev_thr)
        if args.n_boot:
            groups = df["recording_id"].to_numpy() if name.startswith("deep_voice") else None
            m["eer_ci95"] = bootstrap_eer(scores, labels, groups, n_boot=args.n_boot)
        if name == "asvspoof2019_la":
            m.update(asvspoof_metrics(scored, split, paths.dataset(name)))
        elif name.startswith("asvspoof2019_la_robust_"):
            # Degraded eval subsets (P2.4): eval clips, so use the eval ASV scores.
            m.update(asvspoof_metrics(scored, "eval", paths.dataset("asvspoof2019_la")))
        elif name.startswith("asvspoof2019_la_robustdev_"):
            # Degraded dev subsets: used to choose augmentation, so dev ASV scores.
            m.update(asvspoof_metrics(scored, "dev", paths.dataset("asvspoof2019_la")))
        elif name.startswith("in_the_wild"):
            m.update(in_the_wild_metrics(scored))
        elif name.startswith("deep_voice"):
            m.update(deep_voice_metrics(scored, dev_thr))
        results["sets"][entry] = m
        extra = f"  min t-DCF {m['min_tdcf']}" if "min_tdcf" in m else ""
        print(f"  EER {m['eer_pct']:.2f}%  AUC {m['auc']:.4f}{extra}", flush=True)

    def clean(o):
        if isinstance(o, float) and math.isnan(o):
            return None
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, list):
            return [clean(v) for v in o]
        return o

    text = json.dumps(clean(results), indent=2)
    (run_dir / f"metrics{suffix}.json").write_text(text, encoding="utf-8")
    out = REPO_ROOT / "experiments" / f"{run_dir.name}{suffix}"  # not "results": same folder as v1 Results/ on Windows
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(text, encoding="utf-8")
    for f in ("history.json", "train_summary.json", "run_info.json"):
        if (run_dir / f).exists():
            (out / f).write_text((run_dir / f).read_text(encoding="utf-8"), encoding="utf-8")
    md = summary_markdown(f"{run_dir.name}{suffix}", results)
    (out / "summary.md").write_text(md, encoding="utf-8")
    print("\n" + md)
    print(f"Saved: {run_dir / f'metrics{suffix}.json'} and {out}")


if __name__ == "__main__":
    main()

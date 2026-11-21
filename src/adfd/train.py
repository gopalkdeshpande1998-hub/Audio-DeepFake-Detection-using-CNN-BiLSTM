"""Train a detector on ASVspoof 2019 LA train, early-stopping on dev EER.

Usage:
    python -m adfd.train --config configs/e1_cnn_bilstm_mfcc.yaml [--seed 43]
    python -m adfd.train --config ... --limit 2000 --max-epochs 1   # quick smoke test

Test sets are never touched here; score them afterwards with adfd.evaluate.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from adfd.config import load_config
from adfd.data.audio_cache import AudioCache
from adfd.data.augment import build_augmentation
from adfd.data.datasets import CropDataset, make_loader
from adfd.data.silence import load_silence
from adfd.engine import score_loader, train_one_epoch
from adfd.metrics import eer_from_labels
from adfd.models import build_model, count_parameters
from adfd.paths import configure_caches, load_paths
from adfd.runinfo import describe_device, save_run_info
from adfd.seed import seed_everything


def stratified_limit(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Keep about n rows with the same real/fake ratio (for smoke tests)."""
    if n >= len(df):
        return df
    frac = n / len(df)
    return df.groupby("label", group_keys=False).sample(frac=frac, random_state=seed).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--max-epochs", type=int)
    ap.add_argument("--stop-after", type=int,
                    help="end training after this many epochs but keep the max_epochs LR schedule")
    ap.add_argument("--limit", type=int, help="use only this many train and dev rows (smoke test)")
    ap.add_argument("--workers", type=int)
    ap.add_argument("--run-name", help="run folder name (default: <name>_s<seed>_<timestamp>)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.max_epochs is not None:
        cfg["train"]["max_epochs"] = args.max_epochs
    if args.workers is not None:
        cfg["data"]["num_workers"] = args.workers
    if args.stop_after is not None:
        cfg["train"]["stop_after_epochs"] = args.stop_after
    if args.limit:
        cfg["data"]["limit"] = args.limit

    paths = load_paths()
    configure_caches(paths)
    seed = int(cfg["seed"])
    seed_everything(seed, cfg.get("deterministic", False))
    device = torch.device(cfg["train"]["device"] if torch.cuda.is_available() else "cpu")

    run_name = args.run_name or f"{cfg['name']}_s{seed}_{datetime.now():%Y%m%d-%H%M%S}"
    run_dir = paths.runs_dir / run_name
    save_run_info(run_dir, cfg, paths.manifests_dir / "summary.json")
    print(f"Run folder: {run_dir}")
    print(f"Device: {json.dumps(describe_device())}")

    # --- data: ASVspoof train for training, dev for model selection -----------------
    asv = pd.read_csv(paths.manifest("asvspoof2019_la"), keep_default_na=False)
    train_df = asv[asv["split"] == cfg["data"]["train_split"]].reset_index(drop=True)
    dev_df = asv[asv["split"] == cfg["data"]["val_split"]].reset_index(drop=True)
    if cfg["data"].get("limit"):
        train_df = stratified_limit(train_df, cfg["data"]["limit"], seed)
        dev_df = stratified_limit(dev_df, cfg["data"]["limit"], seed)
    cache = AudioCache(paths.cache_dir, "asvspoof2019_la")
    sr = cfg["audio"]["sample_rate"]
    crop = int(cfg["audio"]["crop_seconds"] * sr)
    pad = cfg["audio"]["pad_mode"]
    tcfg = cfg["train"]
    workers = cfg["data"]["num_workers"]
    bounds = None
    if cfg["audio"].get("trim_silence"):
        bounds = load_silence(paths.cache_dir, "asvspoof2019_la")
        if bounds is None:
            raise FileNotFoundError("trim_silence is on but silence bounds are missing: run scripts/compute_silence.py")
        print("Trimming leading/trailing silence (train and dev)")
    augment = build_augmentation(cfg.get("augment"), sr, seed)
    if augment is not None:
        print(f"Augmentation: {augment}")
    norm = cfg["audio"].get("normalize")
    train_ds = CropDataset(train_df, cache, crop, True, pad, sr, speech_bounds=bounds, transform=augment, normalize=norm)
    dev_ds = CropDataset(dev_df, cache, crop, False, pad, sr, speech_bounds=bounds, normalize=norm)
    train_loader = make_loader(train_ds, tcfg["batch_size"], True, workers, seed)
    dev_loader = make_loader(dev_ds, tcfg.get("eval_batch_size", 64), False, workers, seed)
    n_real, n_fake = int((train_df["label"] == 0).sum()), int((train_df["label"] == 1).sum())
    print(f"Train: {len(train_df):,} ({n_real:,} real / {n_fake:,} fake)   Dev: {len(dev_df):,}")

    # --- model, loss, optimiser ---------------------------------------------------------
    model = build_model(cfg).to(device)
    print(f"Model: {cfg['model']['name']}  parameters: {count_parameters(model):,}")
    pos_weight = None
    if cfg["data"].get("balance") == "class_weights":
        # Spoof is the positive class and the majority: weight it down to balance the classes.
        pos_weight = torch.tensor(n_real / n_fake, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    steps_per_epoch = len(train_loader) // tcfg["grad_accum"]
    sched = None
    if tcfg.get("scheduler") == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps_per_epoch * tcfg["max_epochs"])
    amp = bool(tcfg["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    writer = SummaryWriter(run_dir / "tb")

    # --- training loop with early stopping on dev EER ----------------------------------
    best_eer, best_epoch, bad_epochs, step = float("inf"), -1, 0, 0
    history = []
    t_start = time.time()
    last_epoch = min(tcfg["max_epochs"], tcfg.get("stop_after_epochs") or tcfg["max_epochs"])
    for epoch in range(1, last_epoch + 1):
        loss, step, t_train = train_one_epoch(model, train_loader, opt, sched, scaler, loss_fn, device, amp,
                                              tcfg["grad_accum"], tcfg.get("grad_clip", 5.0), writer, step)
        t0 = time.time()
        dev_scores, dev_labels = score_loader(model, dev_loader, device, amp)
        dev_eer, dev_thr = eer_from_labels(dev_scores, dev_labels)
        t_eval = time.time() - t0
        writer.add_scalar("dev/eer", dev_eer, epoch)
        writer.add_scalar("train/epoch_loss", loss, epoch)
        improved = dev_eer < best_eer
        history.append({"epoch": epoch, "train_loss": loss, "dev_eer": dev_eer, "dev_threshold": dev_thr,
                        "train_s": round(t_train, 1), "eval_s": round(t_eval, 1)})
        peak = torch.cuda.max_memory_allocated() / 1024**3 if device.type == "cuda" else 0.0
        print(f"epoch {epoch:3d}  loss {loss:.4f}  dev EER {dev_eer * 100:6.2f}%  "
              f"({t_train:.0f}s train, {t_eval:.0f}s dev, peak VRAM {peak:.2f} GB){'  *best*' if improved else ''}",
              flush=True)
        if improved:
            best_eer, best_epoch, bad_epochs = dev_eer, epoch, 0
            torch.save({"model": model.state_dict(), "config": cfg, "epoch": epoch,
                        "dev_eer": dev_eer, "dev_threshold": dev_thr}, run_dir / "best.pt")
            pd.DataFrame({"utt_id": dev_df["utt_id"], "label": dev_labels, "score": dev_scores}).to_csv(
                run_dir / "scores_asvspoof2019_la_dev.csv", index=False)
        else:
            bad_epochs += 1
            if bad_epochs >= tcfg["early_stop_patience"]:
                print(f"Early stopping: no dev improvement for {bad_epochs} epochs")
                break
        (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    writer.close()
    summary = {"run": run_name, "best_epoch": best_epoch, "best_dev_eer": best_eer,
               "epochs_run": len(history), "minutes": round((time.time() - t_start) / 60, 1)}
    (run_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Done: best dev EER {best_eer * 100:.2f}% at epoch {best_epoch}. Next: python -m adfd.evaluate --run {run_dir}")


if __name__ == "__main__":
    main()

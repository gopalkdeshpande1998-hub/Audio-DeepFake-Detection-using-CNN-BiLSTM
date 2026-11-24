"""Train the temporal (segment-level) detector: frame labels every 0.16 s plus the clip label.

    python -m adfd.train_temporal --config configs/experiments/t1_wavlm_crnn_temporal.yaml
    python -m adfd.train_temporal --config ... --limit 400 --max-epochs 1     # smoke test

Data: PartialSpoof dev, train speakers (partially fake, with 0.16-s labels) + ASVspoof 2019 LA train fully real
and fully fake clips. Validation: PartialSpoof dev, held-out speakers. PartialSpoof eval (the test set) is never
read here. Model selection: validation frame EER. Loss: class-weighted frame BCE + clip_loss_weight * clip BCE.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from adfd.config import load_config
from adfd.data.audio_cache import AudioCache
from adfd.data.augment import build_augmentation
from adfd.data.partialspoof import TemporalCropDataset, load_seglab
from adfd.metrics import eer_from_labels
from adfd.models import build_model, count_parameters
from adfd.paths import configure_caches, load_paths
from adfd.runinfo import describe_device, save_run_info
from adfd.seed import seed_everything, worker_init_fn

SEGLAB = load_paths().dataset("partialspoof") / "segment_labels" / "dev_seglab_0.16.npy"   # configs/paths.yaml


def loader(ds, bs, train, workers, seed):
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(ds, batch_size=bs, shuffle=train, drop_last=train, num_workers=workers,
                      pin_memory=torch.cuda.is_available(), persistent_workers=workers > 0,
                      worker_init_fn=worker_init_fn, generator=g)


def build_rows(cfg: dict, paths, seed: int, limit: int | None):
    t = cfg["temporal"]
    ps = pd.read_csv(paths.manifest(t["manifest"]), keep_default_na=False)
    ps = ps[ps["set"] == "dev"].assign(source="ps", recording_id="")
    tr, va = ps[ps["split"] == "train"], ps[ps["split"] == "val"]
    asv = pd.read_csv(paths.manifest("asvspoof2019_la"), keep_default_na=False)
    asv = asv[asv["split"] == "train"]
    real = asv[asv["label"] == 0].sample(n=min(t["asv_real"], (asv["label"] == 0).sum()), random_state=seed)
    fake = asv[asv["label"] == 1].sample(n=min(t["asv_fake"], (asv["label"] == 1).sum()), random_state=seed)
    extra = pd.concat([real, fake]).assign(source="asv", path="")[["utt_id", "recording_id", "label", "source", "path"]]
    tr = pd.concat([tr[["utt_id", "recording_id", "label", "source", "path"]], extra], ignore_index=True)
    va = va[["utt_id", "recording_id", "label", "source", "path"]].reset_index(drop=True)
    if limit:
        tr = tr.sample(n=min(limit, len(tr)), random_state=seed)
        va = va.sample(n=min(max(limit // 4, 50), len(va)), random_state=seed)
    return tr.reset_index(drop=True), va.reset_index(drop=True)


@torch.no_grad()
def validate(model, dl, device, amp):
    model.eval()
    fz, fy, cz, cy = [], [], [], []
    for wav, y, clip, _ in dl:
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            c, f = model.forward_frames(wav.to(device))
        k = min(f.shape[1], y.shape[1])
        fz.append(f[:, :k].float().cpu().numpy().ravel())
        fy.append(y[:, :k].numpy().ravel())
        cz.append(c.float().cpu().numpy())
        cy.append(clip.numpy())
    fz, fy, cz, cy = map(np.concatenate, (fz, fy, cz, cy))
    frame_eer = eer_from_labels(-fz, fy.astype(int))[0]
    clip_eer = eer_from_labels(-cz, cy.astype(int))[0]
    return frame_eer, clip_eer, fz, fy


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--max-epochs", type=int)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--run-name")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.max_epochs:
        cfg["train"]["max_epochs"] = args.max_epochs
    paths = load_paths()
    configure_caches(paths)
    seed = int(cfg["seed"])
    seed_everything(seed, cfg.get("deterministic", False))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run = args.run_name or f"{cfg['name']}_s{seed}_{datetime.now():%Y%m%d-%H%M%S}"
    run_dir = paths.runs_dir / run
    save_run_info(run_dir, cfg, paths.manifests_dir / "summary.json")
    print(f"Run folder: {run_dir}\nDevice: {json.dumps(describe_device())}", flush=True)

    t, tc = cfg["temporal"], cfg["train"]
    tr, va = build_rows(cfg, paths, seed, args.limit)
    seglab = load_seglab(SEGLAB)
    cache = AudioCache(paths.cache_dir, "asvspoof2019_la")
    augment = build_augmentation(cfg.get("augment"), 16000, seed)
    k = int(t["crop_frames"])
    train_ds = TemporalCropDataset(tr, seglab, k, True, cache, augment)
    val_ds = TemporalCropDataset(va, seglab, k, False, cache)
    train_dl = loader(train_ds, tc["batch_size"], True, args.workers, seed)
    # 0 workers: train workers + persistent val workers exceed the Windows commit limit and hang (2026-11-24)
    val_dl = loader(val_ds, tc["eval_batch_size"], False, 0, seed)

    # frame class balance from the training labels (fake share), ASVspoof rows counted at their clip label
    ps_tr = tr[tr.source == "ps"]
    n_fake = sum(int(seglab[u].sum()) for u in ps_tr.utt_id) + int((tr[tr.source == "asv"].label == 1).sum() * k)
    n_all = sum(len(seglab[u]) for u in ps_tr.utt_id) + int((tr.source == "asv").sum() * k)
    frame_pw = torch.tensor((n_all - n_fake) / max(n_fake, 1), device=device)
    n_clip_fake = int(tr.label.sum())
    clip_pw = torch.tensor((len(tr) - n_clip_fake) / max(n_clip_fake, 1), device=device)
    print(f"Train {len(tr):,} clips (PartialSpoof {len(ps_tr):,}, ASVspoof {len(tr) - len(ps_tr):,}); "
          f"val {len(va):,}; fake frame share {n_fake / n_all:.3f}; pos_weight frame {frame_pw.item():.3f} "
          f"clip {clip_pw.item():.3f}", flush=True)

    model = build_model(cfg).to(device)
    init = torch.load(t["init_from"], map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(init["model"], strict=False)
    assert all(m.startswith("backbone.frame_head") for m in missing) and not unexpected, (missing, unexpected)
    print(f"Warm start from {t['init_from']} (new: {len(missing)} frame-head tensors); parameters "
          f"{count_parameters(model, False):,}", flush=True)

    ssl = [p for p in model.frontend.ssl_parameters() if p.requires_grad]
    head = list(model.backbone.frame_head.parameters())
    ids = {id(p) for p in ssl + head}
    rest = [p for p in model.parameters() if p.requires_grad and id(p) not in ids]
    opt = torch.optim.AdamW([{"params": rest, "lr": tc["lr"]}, {"params": head, "lr": tc["head_lr"]},
                             {"params": ssl, "lr": tc["ssl_lr"]}], weight_decay=tc["weight_decay"])
    steps = len(train_dl) // tc["grad_accum"] * tc["max_epochs"]
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[tc["lr"], tc["head_lr"], tc["ssl_lr"]],
                                                total_steps=max(steps, 1), pct_start=0.05, anneal_strategy="cos")
    amp = device.type == "cuda" and bool(tc["amp"])
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    frame_loss = nn.BCEWithLogitsLoss(pos_weight=frame_pw)
    clip_loss = nn.BCEWithLogitsLoss(pos_weight=clip_pw)
    wc = float(t["clip_loss_weight"])

    best, bad, history, t_start = math.inf, 0, [], time.time()
    for epoch in range(1, tc["max_epochs"] + 1):
        model.train()
        t0, tot, cnt = time.time(), 0.0, 0
        opt.zero_grad(set_to_none=True)
        for step, (wav, y, clip, _) in enumerate(train_dl, 1):
            wav, y, clip = wav.to(device, non_blocking=True), y.to(device), clip.to(device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                c, f = model.forward_frames(wav)
            n = min(f.shape[1], y.shape[1])
            loss = frame_loss(f[:, :n].float(), y[:, :n]) + wc * clip_loss(c.float(), clip)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {step}")
            scaler.scale(loss / tc["grad_accum"]).backward()
            if step % tc["grad_accum"] == 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                if sched.last_epoch + 1 < sched.total_steps:
                    sched.step()
            tot += loss.item() * len(y)
            cnt += len(y)
            if step % 200 == 0:
                rate = step / (time.time() - t0)
                print(f"  epoch {epoch} step {step}/{len(train_dl)} loss {tot / cnt:.4f} "
                      f"({rate:.1f} it/s, ~{(len(train_dl) - step) / rate / 60:.0f} min left)", flush=True)
        t_train = time.time() - t0
        frame_eer, clip_eer, _, _ = validate(model, val_dl, device, amp)
        peak = torch.cuda.max_memory_allocated() / 1024**3 if device.type == "cuda" else 0
        improved = frame_eer < best
        history.append({"epoch": epoch, "loss": tot / max(cnt, 1), "val_frame_eer": frame_eer,
                        "val_clip_eer": clip_eer, "train_s": round(t_train)})
        print(f"epoch {epoch} loss {tot / max(cnt, 1):.4f}  val frame EER {frame_eer * 100:.2f}%  clip EER "
              f"{clip_eer * 100:.2f}%  ({t_train / 60:.1f} min, peak VRAM {peak:.2f} GB){'  *best*' if improved else ''}",
              flush=True)
        if improved:
            best, bad = frame_eer, 0
            tmp = run_dir / "best.pt.partial"
            torch.save({"model": model.state_dict(), "config": cfg, "epoch": epoch, "val_frame_eer": frame_eer,
                        "val_clip_eer": clip_eer}, tmp)
            tmp.replace(run_dir / "best.pt")                     # never leave a half-written checkpoint
        else:
            bad += 1
        (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if bad >= tc["early_stop_patience"]:
            print("Early stopping")
            break
    summary = {"run": run, "best_val_frame_eer": best, "epochs_run": len(history),
               "minutes": round((time.time() - t_start) / 60, 1)}
    (run_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Done: {summary}")


if __name__ == "__main__":
    main()

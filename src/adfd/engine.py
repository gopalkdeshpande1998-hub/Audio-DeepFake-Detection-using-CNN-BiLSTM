"""Training and scoring loops shared by train.py and evaluate.py."""

from __future__ import annotations

import math
import time

import numpy as np
import torch
from torch import nn


def _autocast(device: torch.device, amp: bool):
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp and device.type == "cuda")


@torch.no_grad()
def score_loader(model: nn.Module, loader, device: torch.device, amp: bool = True):
    """Return (scores, labels) in dataset order. Score = -spoof_logit (higher = bona fide)."""
    model.eval()
    n = len(loader.dataset)
    scores = np.full(n, np.nan, dtype=np.float64)
    labels = np.full(n, -1, dtype=np.int64)
    for wav, label, idx in loader:
        wav = wav.to(device, non_blocking=True)
        with _autocast(device, amp):
            logit = model(wav)
        idx = idx.numpy()
        scores[idx] = -logit.float().cpu().numpy()
        labels[idx] = label.numpy().astype(np.int64)
    if np.isnan(scores).any():
        raise RuntimeError("some items were not scored")
    return scores, labels


def train_one_epoch(model, loader, optimizer, scheduler, scaler, loss_fn, device, amp=True,
                    grad_accum=1, grad_clip=5.0, writer=None, global_step=0, log_every=50):
    """One pass over the training loader. Returns (mean loss, new global_step)."""
    model.train()
    total, count = 0.0, 0
    t0 = time.time()
    optimizer.zero_grad(set_to_none=True)
    for step, (wav, label, _) in enumerate(loader, 1):
        wav = wav.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        with _autocast(device, amp):
            logit = model(wav)
        loss = loss_fn(logit.float(), label) / grad_accum
        scaler.scale(loss).backward()
        if step % grad_accum == 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()
            global_step += 1
        value = loss.item() * grad_accum
        if not math.isfinite(value):
            raise RuntimeError(f"non-finite loss at step {step}")
        total += value * len(label)
        count += len(label)
        if writer is not None and step % log_every == 0:
            writer.add_scalar("train/loss", value, global_step)
            writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
    return total / max(count, 1), global_step, time.time() - t0

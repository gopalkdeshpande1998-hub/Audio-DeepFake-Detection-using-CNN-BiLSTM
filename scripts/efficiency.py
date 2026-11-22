"""P3.5 efficiency: size and inference speed of every model (seed-42 checkpoint).

  params          all parameters (front end + back end), and the trainable ones
  checkpoint MB   size of best.pt
  GPU speed       batch 32 x 4 s clips, fp16 autocast: seconds of audio scored per second
  CPU speed       batch 1 x 4 s clip, fp32, all CPU threads: seconds of audio per second
                  (> 1 = faster than real time); and milliseconds per clip
Run it when no other GPU job is running.

Usage:
    python scripts/efficiency.py [--models e5a_lcnn e7a_wavlm_crnn ...]
Writes experiments/efficiency/efficiency.json and .md
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import torch

from adfd.models import build_model, count_parameters
from adfd.paths import REPO_ROOT, configure_caches, load_paths

EXP = REPO_ROOT / "experiments"
MODELS = [
    ("E1 CNN-BiLSTM, MFCC (v1)", "e1_cnn_bilstm_mfcc"),
    ("E4a CNN-BiLSTM, LFCC + RawBoost", "e4a_rawboost"),
    ("E5a LCNN, LFCC", "e5a_lcnn"),
    ("E5b RawNet2", "e5b_rawnet2"),
    ("E5c AASIST", "e5c_aasist"),
    ("E6a SE-res CNN-BiLSTM", "e6a_crnn_seres"),
    ("E6b SE-res CNN-BiLSTM-Att", "e6b_crnn_seres_asp"),
    ("E6c light CNN-BiGRU-Att", "e6c_crnn_light_gru"),
    ("E7a WavLM-Base+ + CNN-BiLSTM-Att", "e7a_wavlm_crnn"),
]
CLIP_S = 4.0


def timed(fn, reps: int, sync) -> float:
    fn()
    sync()
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    sync()
    return (time.perf_counter() - t) / reps


def measure(run_dir, gpu: bool = True, sr=16000) -> dict:
    ckpt = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    model = build_model(ckpt["config"])
    model.load_state_dict(ckpt["model"])
    model.eval()
    n = int(CLIP_S * sr)
    res = {"params": count_parameters(model, False), "trainable_params": count_parameters(model),
           "checkpoint_mb": round((run_dir / "best.pt").stat().st_size / 2**20, 1)}
    with torch.no_grad():
        x = torch.randn(1, n) * 0.1
        cpu_s = timed(lambda: model(x), 10, lambda: None)
        res["cpu_ms_per_clip"] = round(1000 * cpu_s, 1)
        res["cpu_audio_s_per_s"] = round(CLIP_S / cpu_s, 1)
        if gpu and torch.cuda.is_available():
            model.cuda()
            xg = torch.randn(32, n, device="cuda") * 0.1

            def run():
                with torch.autocast("cuda", dtype=torch.float16):
                    model(xg)
            gpu_s = timed(run, 10, torch.cuda.synchronize)
            res["gpu_audio_s_per_s"] = round(32 * CLIP_S / gpu_s, 0)
            model.cpu()
            torch.cuda.empty_cache()
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="*", help="model prefixes (default: all)")
    ap.add_argument("--no-gpu", action="store_true", help="CPU timing only")
    args = ap.parse_args()
    paths = load_paths()
    configure_caches(paths)
    torch.backends.cudnn.benchmark = True
    models = [m for m in MODELS if not args.models or m[1] in args.models]
    out, lines = {}, ["# P3.5 Model size and inference speed", "",
                      f"Seed-42 checkpoints, {CLIP_S:.0f} s clips. GPU: RTX 3050 Laptop (4 GB), batch 32, fp16. "
                      f"CPU: batch 1, fp32, {torch.get_num_threads()} threads. Speed = seconds of audio scored per second.",
                      "", "| Model | parameters | trainable | checkpoint (MB) | GPU speed (× real time) | "
                      "CPU speed (× real time) | CPU ms per clip |", "|---|---|---|---|---|---|---|"]
    for label, prefix in models:
        run_dir = paths.runs_dir / f"{prefix}_s42"
        if not (run_dir / "best.pt").exists():
            continue
        r = measure(run_dir, gpu=not args.no_gpu)
        out[prefix] = {"label": label, **r}
        lines.append(f"| {label} | {r['params']:,} | {r['trainable_params']:,} | {r['checkpoint_mb']} | "
                     f"{r.get('gpu_audio_s_per_s', float('nan')):,.0f} | {r['cpu_audio_s_per_s']:,.1f} | "
                     f"{r['cpu_ms_per_clip']:,.1f} |")
        print(lines[-1], flush=True)
    d = EXP / "efficiency"
    d.mkdir(parents=True, exist_ok=True)
    (d / "efficiency.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    (d / "efficiency.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

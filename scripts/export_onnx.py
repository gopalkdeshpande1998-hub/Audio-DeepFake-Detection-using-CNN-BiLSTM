"""P3.5 ONNX export of a trained model (back end), with optional INT8 dynamic quantization.

The spectral front end (STFT -> LFCC/MFCC) cannot be exported by torch's TorchScript ONNX
exporter (complex STFT), so the exported graph takes the front-end features
(B, 1, F, T) and returns the spoof logit; compute the features with the PyTorch front
end (cheap) or re-implement them. Raw-waveform models (RawNet2, AASIST) export whole.
The script checks the ONNX outputs against PyTorch and times both on the CPU.

Usage:
    python scripts/export_onnx.py --run runs/e4a_rawboost_s42 [--int8]
Writes <run>/model.onnx (and model.int8.onnx) plus <run>/onnx_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch

from adfd.models import build_model


def timed(fn, reps=20):
    fn()
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t) / reps


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--int8", action="store_true", help="also write an INT8 dynamically quantized model")
    args = ap.parse_args()
    import onnxruntime as ort

    run = Path(args.run)
    ck = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    model = build_model(ck["config"])
    model.load_state_dict(ck["model"])
    model.eval()
    wav = torch.randn(4, int(ck["config"]["audio"]["crop_seconds"] * 16000)) * 0.1
    raw = isinstance(model.frontend, torch.nn.Identity)
    with torch.no_grad():
        inp = wav if raw else model.frontend(wav)
        ref = model.backbone(inp).numpy()
    out = run / "model.onnx"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(model.backbone, (inp,), out, input_names=["wav" if raw else "features"],
                          output_names=["spoof_logit"], dynamic_axes={("wav" if raw else "features"): {0: "batch"}},
                          opset_version=18, dynamo=False)
    report = {"input": "waveform (B, N)" if raw else f"front-end features {tuple(inp.shape[1:])}",
              "frontend_in_pytorch": not raw}
    files = {"fp32": out}
    if args.int8:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        files["int8"] = run / "model.int8.onnx"
        quantize_dynamic(str(out), str(files["int8"]), weight_type=QuantType.QInt8)
    x1 = inp[:1].numpy()
    with torch.no_grad():
        report["pytorch_cpu_ms_per_clip"] = round(1000 * timed(lambda: model.backbone(inp[:1])), 2)
    for k, f in files.items():
        s = ort.InferenceSession(str(f), providers=["CPUExecutionProvider"])
        name = s.get_inputs()[0].name
        y = s.run(None, {name: inp.numpy()})[0]
        report[k] = {"file": f.name, "size_mb": round(f.stat().st_size / 2**20, 2),
                     "max_abs_diff_vs_pytorch": float(np.abs(y - ref).max()),
                     "same_decision_pct": float(100 * np.mean((y > 0) == (ref > 0))),
                     "cpu_ms_per_clip": round(1000 * timed(lambda: s.run(None, {name: x1})), 2)}
    (run / "onnx_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Detection models and the registry that builds them from a config."""

from __future__ import annotations

from torch import nn

from adfd.features.mfcc import MFCCFrontend
from adfd.models.cnn_bilstm import CNNBiLSTM


class Detector(nn.Module):
    """Front end (waveform -> features) followed by a classifier -> spoof logit."""

    def __init__(self, frontend: nn.Module, backbone: nn.Module):
        super().__init__()
        self.frontend = frontend
        self.backbone = backbone

    def forward(self, wav):
        return self.backbone(self.frontend(wav))


def build_frontend(cfg: dict, sample_rate: int) -> nn.Module:
    name = cfg.get("name", "mfcc")
    kwargs = {k: v for k, v in cfg.items() if k != "name"}
    if name == "mfcc":
        return MFCCFrontend(sample_rate=sample_rate, **kwargs)
    raise ValueError(f"Unknown front end {name!r}")


def build_model(cfg: dict) -> Detector:
    """cfg is the full experiment config (uses cfg['model'] and cfg['audio'])."""
    mcfg = cfg["model"]
    frontend = build_frontend(mcfg.get("frontend", {"name": "mfcc"}), cfg["audio"]["sample_rate"])
    name = mcfg["name"]
    kwargs = mcfg.get("kwargs", {})
    if name == "cnn_bilstm":
        backbone = CNNBiLSTM(**kwargs)
    else:
        raise ValueError(f"Unknown model {name!r}")
    return Detector(frontend, backbone)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

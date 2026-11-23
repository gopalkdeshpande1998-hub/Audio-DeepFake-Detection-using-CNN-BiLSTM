"""Model-internal explanations (plan P4.5): Grad-CAM, attention over time, SSL layer weights.

Complements the gradient x input saliency in adfd.inference.Analyzer.saliency:

  grad_cam(an, seg)        Grad-CAM on the last 2-D convolution of the model's back end,
                           upsampled to the input feature map (freq x time for spectral
                           front ends; feature x time for the WavLM front end).
  attention(an, seg)       the attentive-statistics-pooling weights over time (CRNN with
                           pooling "asp", e.g. E6b and E7a): where in the window the model
                           put its weight when summarising the utterance.
  layer_weights(an)        the learned softmax weights over the WavLM hidden layers (E7a):
                           which depth of the pretrained model the detector relies on.

All three show where a model looked, not proof of manipulation.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def _last_conv(module: nn.Module) -> nn.Conv2d | None:
    convs = [m for m in module.modules() if isinstance(m, nn.Conv2d)]
    return convs[-1] if convs else None


def _features(an, seg: np.ndarray) -> torch.Tensor:
    wav = torch.from_numpy(an._prep(seg)).unsqueeze(0).to(an.device)
    with torch.no_grad():
        return an.model.frontend(wav).float()


def grad_cam(an, seg: np.ndarray) -> dict | None:
    """{'cam': (F, T) in [0, 1], 'feature': (F, T), 'logit', 'layer'} or None when the back end has no conv."""
    conv = _last_conv(an.model.backbone)
    if conv is None:
        return None
    feat = _features(an, seg)
    if feat.dim() != 4:
        return None
    store: dict = {}
    h1 = conv.register_forward_hook(lambda m, i, o: store.__setitem__("act", o))
    h2 = conv.register_full_backward_hook(lambda m, gi, go: store.__setitem__("grad", go[0]))
    try:
        with torch.enable_grad(), torch.backends.cudnn.flags(enabled=False):
            x = feat.detach().requires_grad_(True)
            logit = an.model.backbone(x)
            an.model.zero_grad(set_to_none=True)
            logit.sum().backward()
    finally:
        h1.remove()
        h2.remove()
    act, grad = store["act"].detach().float(), store["grad"].detach().float()       # (1, C, f, t)
    w = grad.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * act).sum(dim=1, keepdim=True))                                  # towards "fake"
    cam = F.interpolate(cam, size=feat.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam = cam.cpu().numpy()
    cam = cam / (cam.max() + 1e-12)
    name = next((n for n, m in an.model.backbone.named_modules() if m is conv), "conv")
    return {"cam": cam, "feature": feat[0, 0].cpu().numpy(), "logit": float(logit.item()), "layer": name}


def attention(an, seg: np.ndarray) -> dict | None:
    """{'weights': (T',) summing to 1, 'times_s': (T',), 'logit'} for back ends with attentive pooling."""
    pool = getattr(an.model.backbone, "pool", None)
    att = getattr(pool, "att", None)
    if att is None:
        return None
    feat = _features(an, seg)
    store: dict = {}
    h = att.register_forward_hook(lambda m, i, o: store.__setitem__("e", o))
    try:
        with torch.no_grad():
            logit = an.model.backbone(feat)
    finally:
        h.remove()
    w = torch.softmax(store["e"].float(), dim=1)[0, :, 0].cpu().numpy()
    dur = len(an._prep(seg)) / 16000
    return {"weights": w, "times_s": (np.arange(len(w)) + 0.5) * dur / len(w), "logit": float(logit.item())}


def layer_weights(an) -> np.ndarray | None:
    """Softmax weights over the SSL hidden layers (index 0 = CNN encoder output), or None."""
    lw = getattr(getattr(an.model, "frontend", None), "layer_weights", None)
    if lw is None:
        return None
    return torch.softmax(lw.detach().float(), dim=0).cpu().numpy()


def plot_grad_cam(gc: dict, title: str = "", spectral: bool = True):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.imshow(gc["feature"], origin="lower", aspect="auto", cmap="gray_r")
    ax.imshow(gc["cam"], origin="lower", aspect="auto", cmap="inferno", alpha=0.55)
    ax.set_ylabel("frequency band" if spectral else "WavLM feature")
    ax.set_xlabel("time frame")
    ax.set_title(title or f"Grad-CAM ({gc['layer']})", fontsize=10)
    fig.tight_layout()
    return fig


def plot_attention(att: dict, title: str = "Attention over time"):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 2.2))
    ax.fill_between(att["times_s"], att["weights"], color="#2563eb", alpha=0.35, step="mid")
    ax.plot(att["times_s"], att["weights"], color="#2563eb", lw=1.2, drawstyle="steps-mid")
    ax.axhline(1 / len(att["weights"]), color="#78716c", ls="--", lw=1)
    ax.set_xlabel("time in the analysed window (s)")
    ax.set_ylabel("weight")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return fig


def plot_layer_weights(w: np.ndarray, title: str = "Which WavLM layers the detector uses"):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 2.4))
    ax.bar(np.arange(len(w)), w, color="#2563eb")
    ax.axhline(1 / len(w), color="#78716c", ls="--", lw=1)
    ax.set_xticks(np.arange(len(w)), ["CNN"] + [str(i) for i in range(1, len(w))])
    ax.set_xlabel("WavLM layer (0 = convolutional encoder output)")
    ax.set_ylabel("weight")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return fig

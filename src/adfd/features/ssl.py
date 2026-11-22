"""P3.3 pretrained self-supervised (SSL) front end: WavLM / wav2vec 2.0 / XLS-R.

Waveform (B, N) at 16 kHz -> hidden states of every transformer layer -> learned
softmax-weighted sum -> linear projection to `proj_dim` -> (B, 1, proj_dim, T)
(T = 50 frames per second), so any (B, 1, F, T) back end can follow.

Modes:
  finetune  the transformer is trained (its convolutional feature encoder stays frozen);
            gradient checkpointing keeps a Base model inside 4 GB of VRAM.
  frozen    the SSL model only runs forward (no gradients, fp16); only the layer weights
            and the projection are trained. Used for the Large models.
The transformer can also be frozen for the first epochs (train.freeze_ssl_epochs);
train.py calls set_ssl_trainable().
Weights come from the Hugging Face hub (cached under .cache/huggingface).
"""

from __future__ import annotations

import torch
from torch import nn

MODELS = {
    "wavlm_base_plus": "microsoft/wavlm-base-plus",
    "wav2vec2_base": "facebook/wav2vec2-base",
    "xlsr_300m": "facebook/wav2vec2-xls-r-300m",
    "wavlm_large": "microsoft/wavlm-large",
}
# Whether each model's own feature extractor zero-mean/unit-variance normalises the waveform.
NORMALIZE = {"wavlm_base_plus": False, "wav2vec2_base": True, "xlsr_300m": True, "wavlm_large": True}


def load_ssl(name: str, config=None) -> nn.Module:
    """Pretrained model by short name, or a randomly initialised one from `config` (tests)."""
    from transformers import AutoModel

    if config is not None:
        return AutoModel.from_config(config)
    return AutoModel.from_pretrained(MODELS.get(name, name))


class SSLFrontend(nn.Module):
    def __init__(self, sample_rate: int = 16000, model: str = "wavlm_base_plus", mode: str = "finetune",
                 proj_dim: int = 128, grad_checkpointing: bool = True, normalize: bool | None = None, hf_config=None):
        super().__init__()
        if sample_rate != 16000:
            raise ValueError("SSL models expect 16 kHz audio")
        self.ssl = load_ssl(model, hf_config)
        self.ssl.config.layerdrop = 0.0                      # stable gradients; also makes eval deterministic
        if hasattr(self.ssl, "freeze_feature_encoder"):
            self.ssl.freeze_feature_encoder()
        self.mode = mode
        self.normalize = NORMALIZE.get(model, True) if normalize is None else normalize
        if mode == "frozen":
            for p in self.ssl.parameters():
                p.requires_grad_(False)
        elif grad_checkpointing:
            self.ssl.activation_checkpointing_enable(activation_checkpointing_kwargs={"use_reentrant": False})
        n_layers = self.ssl.config.num_hidden_layers + 1     # + the CNN/embedding output
        dim = self.ssl.config.hidden_size
        self.layer_weights = nn.Parameter(torch.zeros(n_layers))
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, proj_dim)

    def ssl_parameters(self):
        return [p for n, p in self.ssl.named_parameters() if "feature_extractor" not in n]

    def set_ssl_trainable(self, trainable: bool):
        """Freeze or unfreeze the transformer (the CNN feature encoder always stays frozen)."""
        if self.mode == "frozen":
            return
        for p in self.ssl_parameters():
            p.requires_grad_(trainable)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.mode == "frozen":
            self.ssl.eval()                                  # no dropout in a frozen model
        return self

    def forward(self, wav):
        wav = wav.float()
        if self.normalize:
            wav = (wav - wav.mean(dim=1, keepdim=True)) / (wav.std(dim=1, keepdim=True) + 1e-5)
        if self.mode == "frozen":
            with torch.no_grad():
                hs = self.ssl(wav, output_hidden_states=True).hidden_states
        else:
            hs = self.ssl(wav, output_hidden_states=True).hidden_states
        w = torch.softmax(self.layer_weights, dim=0)
        x = sum(wi * h.float() for wi, h in zip(w, hs))      # (B, T, D)
        x = self.proj(self.norm(x))                          # (B, T, proj_dim)
        return x.transpose(1, 2).unsqueeze(1)                # (B, 1, proj_dim, T)

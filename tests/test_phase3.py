"""Phase 3 models: shapes, parameter counts and SSL front-end behaviour (CPU, no downloads)."""

import pytest
import torch

from adfd.models import build_model, count_parameters

SR = 16000


def _build(model: dict):
    return build_model({"model": model, "audio": {"sample_rate": SR}})


@pytest.mark.parametrize("model", [
    {"name": "lcnn", "frontend": {"name": "lfcc"}},
    {"name": "rawnet2", "frontend": {"name": "raw"}},
    {"name": "aasist", "frontend": {"name": "raw"}},
    {"name": "crnn", "frontend": {"name": "lfcc"}, "kwargs": {"cnn": "v1", "pooling": "last"}},
    {"name": "crnn", "frontend": {"name": "lfcc"}, "kwargs": {"cnn": "se_res", "pooling": "asp"}},
    {"name": "crnn", "frontend": {"name": "lfcc"}, "kwargs": {"cnn": "light", "rnn": "gru", "pooling": "asp"}},
])
def test_models_give_one_logit_per_clip(model):
    m = _build(model).eval()
    with torch.no_grad():
        out = m(torch.randn(2, 4 * SR) * 0.1)
    assert out.shape == (2,) and torch.isfinite(out).all()


def test_aasist_matches_paper_size():
    # The AASIST paper reports 297k parameters; one logit instead of two removes 33.
    assert count_parameters(_build({"name": "aasist", "frontend": {"name": "raw"}})) == 297_705


def test_aasist_on_feature_maps():
    from adfd.models.aasist import AASIST

    m = AASIST(input="features", in_freq=128).eval()
    with torch.no_grad():
        assert m(torch.randn(2, 1, 128, 199)).shape == (2,)


def test_asp_is_permutation_invariant_over_time():
    from adfd.models.crnn import AttentiveStatsPool

    pool = AttentiveStatsPool(8)
    h = torch.randn(3, 10, 8)
    perm = torch.randperm(10)
    assert torch.allclose(pool(h), pool(h[:, perm]), atol=1e-5)
    assert pool(h).shape == (3, 16)


def _tiny_ssl(mode="finetune"):
    from transformers import WavLMConfig

    from adfd.features.ssl import SSLFrontend

    cfg = WavLMConfig(hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64,
                      conv_dim=(32, 32), conv_stride=(5, 64), conv_kernel=(10, 64), num_conv_pos_embeddings=16,
                      num_conv_pos_embedding_groups=2)
    return SSLFrontend(model="wavlm_base_plus", mode=mode, proj_dim=16, grad_checkpointing=False, hf_config=cfg)


def test_ssl_frontend_shape_and_layer_weights():
    fe = _tiny_ssl().eval()
    with torch.no_grad():
        x = fe(torch.randn(2, SR))
    assert x.shape[:3] == (2, 1, 16) and x.shape[3] > 0
    assert fe.layer_weights.numel() == 3                      # 2 transformer layers + embedding output


def test_ssl_freeze_and_unfreeze():
    fe = _tiny_ssl()
    encoder = [p for n, p in fe.ssl.named_parameters() if "feature_extractor" in n]
    assert encoder and not any(p.requires_grad for p in encoder)    # CNN encoder always frozen
    fe.set_ssl_trainable(False)
    assert not any(p.requires_grad for p in fe.ssl_parameters())
    fe.set_ssl_trainable(True)
    assert all(p.requires_grad for p in fe.ssl_parameters())
    assert not any(p.requires_grad for p in encoder)


def test_frozen_ssl_trains_only_the_head():
    fe = _tiny_ssl("frozen")
    fe.train()
    assert not fe.ssl.training                                      # frozen model stays in eval mode
    trainable = {n.split(".")[0] for n, p in fe.named_parameters() if p.requires_grad}
    assert trainable == {"layer_weights", "norm", "proj"}

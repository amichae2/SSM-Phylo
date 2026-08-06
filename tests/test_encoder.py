"""Encoder tests: config-driven build_encoder (single from-scratch path).

CI runs WITHOUT `kernels` and WITHOUT network: build_encoder always builds a
from-scratch transformers MambaForCausalLM with random weights.
"""
import torch

from ssm_phylo.models.encoder import MambaEncoder, build_encoder, make_position_ids

TINY_MAMBA = {"state_size": 4, "time_step_rank": 8, "conv_kernel": 3, "expand": 2}


def tiny_cfg():
    from types import SimpleNamespace

    return SimpleNamespace(
        d_model=32,
        n_layer=2,
        vocab_size=38,
        encoder=SimpleNamespace(mamba=dict(TINY_MAMBA)),
    )


def test_from_scratch_builds_and_forwards():
    encoder = build_encoder(tiny_cfg(), device="cpu")
    assert isinstance(encoder, MambaEncoder)
    assert encoder.d_model == 32
    hidden = encoder(torch.randint(0, 38, (2, 64)))
    assert hidden.shape == (2, 64, 32)
    assert torch.isfinite(hidden).all()
    assert encoder.supports_positions is False


def test_from_scratch_layer_selection():
    encoder = build_encoder(tiny_cfg(), device="cpu")
    tokens = torch.randint(0, 38, (1, 16))
    h_last = encoder(tokens, layer=-1)
    h_first = encoder(tokens, layer=0)
    assert h_last.shape == h_first.shape == (1, 16, 32)
    assert not torch.allclose(h_last, h_first)


def test_freeze_toggles_requires_grad():
    encoder = build_encoder(tiny_cfg(), device="cpu")
    assert all(p.requires_grad for p in encoder.backbone.parameters())
    encoder.freeze(True)
    assert all(not p.requires_grad for p in encoder.backbone.parameters())
    encoder.freeze(False)
    assert all(p.requires_grad for p in encoder.backbone.parameters())


def test_make_position_ids():
    spans = torch.tensor([[[0, 5], [6, 11], [12, 17]], [[0, 3], [4, 7], [8, 9]]])
    pos, seq_pos = make_position_ids(spans, max_pos=10, max_seq_pos=2)
    assert pos.shape == seq_pos.shape == (2, 17)
    assert pos[0, 0] == 1 and pos[0, 16] == 10  # 1-indexed, clipped at max_pos
    assert torch.all(seq_pos[0, 0:5] == 0)
    assert torch.all(seq_pos[0, 6:11] == 1)
    assert torch.all(seq_pos[0, 12:17] == 2)
    assert torch.all(seq_pos[1, 8:9] == 2)  # clipped at max_seq_pos? no: 2 <= 2


def test_make_position_ids_single_sample():
    spans = torch.tensor([[0, 4], [5, 9]])
    pos, seq_pos = make_position_ids(spans)
    assert pos.shape == (1, 9)
    assert torch.all(seq_pos[0, 0:4] == 0) and torch.all(seq_pos[0, 5:9] == 1)

"""Encoder tests: config-driven build_encoder in all three modes.

CI runs WITHOUT mamba-ssm and WITHOUT network: from_scratch is the CI path;
degraded_protmamba is tested against a SYNTHESIZED checkpoint that mimics the
broken ProtMamba v1.0 release (ckpt_layer prefixes, mismatched embedding,
lying config.json); ptm_mamba must raise a clear "not available" error.
"""
import json

import pytest
import torch

from ssm_phylo.models import checkpoint_compat as cc
from ssm_phylo.models.encoder import ProtMambaEncoder, build_encoder, make_position_ids

TINY_MAMBA = {"state_size": 4, "time_step_rank": 8, "conv_kernel": 3, "expand": 2}


def tiny_cfg(kind="from_scratch", checkpoint_dir=None, ptm_model_id="programmablebio/ptm-mamba"):
    from types import SimpleNamespace

    return SimpleNamespace(
        d_model=32,
        n_layer=2,
        vocab_size=38,
        encoder=SimpleNamespace(
            kind=kind,
            checkpoint_dir=checkpoint_dir,
            mamba=dict(TINY_MAMBA),
            ptm_model_id=ptm_model_id,
        ),
    )


def synthesize_fake_checkpoint(dest, corrupt_backbone=False):
    """Mimic the ProtMamba v1.0 release anomalies on a tiny real model."""
    from transformers import MambaConfig, MambaForCausalLM

    src = MambaForCausalLM(
        MambaConfig(
            vocab_size=40, hidden_size=32, num_hidden_layers=2,
            state_size=4, time_step_rank=8, conv_kernel=3, expand=2,
            tie_word_embeddings=False,
        )
    )
    fake = {}
    for k, v in src.state_dict().items():
        if ".mixer." in k:  # checkpoint_mixer naming from the real release
            k = k.replace(".mixer.", ".mixer.ckpt_layer.")
        fake[k] = v
    fake["backbone.embedding.weight"] = torch.randn(40, 16)   # mismatched dims
    fake["lm_head.weight"] = torch.randn(40, 16)
    fake["backbone.position_embedding.weight"] = torch.randn(2048, 16)
    if corrupt_backbone:
        fake["backbone.layers.0.mixer.ckpt_layer.in_proj.weight"] = torch.randn(999, 32)
    dest.mkdir(parents=True, exist_ok=True)
    torch.save(fake, dest / "pytorch_model.bin")
    # config.json deliberately LIES about shapes, like the real release
    with open(dest / "config.json", "w") as fh:
        json.dump({"d_model": 1024, "n_layer": 16, "vocab_size": 38}, fh)
    return src


# --------------------------------------------------------------------------- #
# from_scratch (CI path)
# --------------------------------------------------------------------------- #
def test_from_scratch_builds_and_forwards():
    encoder = build_encoder(tiny_cfg(), device="cpu")
    assert isinstance(encoder, ProtMambaEncoder)
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


# --------------------------------------------------------------------------- #
# degraded_protmamba
# --------------------------------------------------------------------------- #
def test_degraded_loads_backbone_and_reinits_embedding(tmp_path):
    src = synthesize_fake_checkpoint(tmp_path / "ckpt")
    encoder = build_encoder(tiny_cfg("degraded_protmamba", str(tmp_path / "ckpt")), device="cpu")
    assert encoder.d_model == 32
    emb = encoder.backbone.backbone.embeddings.weight
    assert tuple(emb.shape) == (38, 32)  # re-initialized, NOT the (40,16) junk
    loaded = encoder.backbone.backbone.layers[0].mixer.in_proj.weight
    assert torch.allclose(loaded, src.backbone.layers[0].mixer.in_proj.weight)
    hidden = encoder(torch.randint(0, 38, (1, 64)))
    assert hidden.shape == (1, 64, 32) and torch.isfinite(hidden).all()


def test_degraded_backbone_mismatch_fails_loudly(tmp_path):
    synthesize_fake_checkpoint(tmp_path / "ckpt", corrupt_backbone=True)
    with pytest.raises(cc.ProtMambaWeightsError, match="backbone shape validation failed"):
        build_encoder(tiny_cfg("degraded_protmamba", str(tmp_path / "ckpt")), device="cpu")


def test_degraded_missing_checkpoint_prints_download_hint(monkeypatch, tmp_path):
    monkeypatch.setenv("PROT_MAMBA_CKPT", str(tmp_path / "nope"))
    with pytest.raises(cc.ProtMambaWeightsError, match="download_weights.sh"):
        build_encoder(tiny_cfg("degraded_protmamba"), device="cpu")


def test_degraded_non_mamba_checkpoint_fails(tmp_path):
    (tmp_path / "ckpt").mkdir()
    torch.save({"some.other.model.weight": torch.randn(4, 4)}, tmp_path / "ckpt" / "pytorch_model.bin")
    with pytest.raises(cc.ProtMambaWeightsError, match="no backbone.layers"):
        build_encoder(tiny_cfg("degraded_protmamba", str(tmp_path / "ckpt")), device="cpu")


# --------------------------------------------------------------------------- #
# ptm_mamba (dormant)
# --------------------------------------------------------------------------- #
def test_ptm_mamba_not_available_without_local_checkout(monkeypatch, tmp_path):
    monkeypatch.delenv("SSM_PHYLO_PTM_MAMBA_DIR", raising=False)
    monkeypatch.setenv("SSM_PHYLO_PTM_MAMBA_TIMEOUT", "10")
    with pytest.raises(RuntimeError, match="ptm_mamba not available"):
        build_encoder(tiny_cfg("ptm_mamba"), device="cpu")


def test_ptm_mamba_bogus_local_checkout_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("SSM_PHYLO_PTM_MAMBA_DIR", str(tmp_path / "empty_checkout"))
    (tmp_path / "empty_checkout").mkdir()
    monkeypatch.setenv("SSM_PHYLO_PTM_MAMBA_TIMEOUT", "10")
    with pytest.raises(RuntimeError, match="ptm_mamba not available"):
        build_encoder(tiny_cfg("ptm_mamba"), device="cpu")


def test_unknown_kind_raises():
    with pytest.raises(ValueError, match="unknown encoder.kind"):
        build_encoder(tiny_cfg("quantum_duck"), device="cpu")

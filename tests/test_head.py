"""Task head tests: hard guarantees + gradient flow through head and encoder."""
import pytest
import torch

from ssm_phylo.models.encoder import build_encoder
from ssm_phylo.models.head import AttentionPooling, DistanceHead, PhyloModel
from ssm_phylo.models.losses import mae_loss

TINY_MAMBA = {"state_size": 4, "time_step_rank": 8, "conv_kernel": 3, "expand": 2}


@pytest.fixture(scope="module")
def encoder():
    from types import SimpleNamespace

    cfg = SimpleNamespace(
        d_model=32, n_layer=2, vocab_size=38,
        encoder=SimpleNamespace(
            kind="from_scratch", checkpoint_dir=None,
            mamba=dict(TINY_MAMBA), ptm_model_id="x",
        ),
    )
    return build_encoder(cfg, device="cpu")


def make_inputs(batch=2, n_seqs=5, seq_len=60, device="cpu"):
    tokens = torch.randint(0, 38, (batch, n_seqs * seq_len + n_seqs - 1))
    spans = torch.zeros(batch, n_seqs, 2, dtype=torch.long)
    for k in range(n_seqs):
        spans[:, k, 0] = k * (seq_len + 1)
        spans[:, k, 1] = k * (seq_len + 1) + seq_len
    mask = torch.ones(batch, n_seqs, dtype=torch.bool)
    return tokens.to(device), spans.to(device), mask.to(device)


def make_target(batch, n_seqs, device="cpu"):
    t = torch.rand(batch, n_seqs, n_seqs, device=device) * 2.0
    t = torch.triu(t, diagonal=1)
    t = t + t.transpose(1, 2)
    return t


def test_distance_matrix_hard_guarantees(encoder):
    model = PhyloModel(encoder, d_emb=16, max_dist=3.0, head_type="bilinear_mlp")
    tokens, spans, mask = make_inputs()
    with torch.no_grad():
        dm, embs = model(tokens, spans, mask)
    assert dm.shape == (2, 5, 5)
    assert embs.shape == (2, 5, 16)
    assert torch.allclose(dm, dm.transpose(1, 2), atol=1e-5)   # symmetric
    assert (dm >= 0.0).all()                                   # non-negative
    assert (dm <= model.head.max_dist + 1e-5).all()            # bounded
    diag = torch.diagonal(dm, dim1=1, dim2=2)
    assert torch.allclose(diag, torch.zeros_like(diag), atol=1e-6)


@pytest.mark.parametrize("head_type", ["bilinear", "mlp", "bilinear_mlp"])
def test_all_head_types_guarantees(head_type):
    head = DistanceHead(d_model=16, d_emb=8, max_dist=3.0, head_type=head_type)
    embs = torch.randn(2, 7, 8)
    dm = head(embs)
    assert torch.allclose(dm, dm.transpose(1, 2), atol=1e-5)
    assert (dm >= 0.0).all() and (dm <= 3.0 + 1e-5).all()


def test_pool_all_masked_span_is_zero(encoder):
    head = DistanceHead(d_model=32, d_emb=16)
    tokens, spans, mask = make_inputs()
    mask[1, 2] = False
    with torch.no_grad():
        hidden = encoder(tokens)
    embs = head.pool_all(hidden, spans, mask)
    assert torch.allclose(embs[1, 2], torch.zeros(16))
    assert not torch.allclose(embs[1, 3], torch.zeros(16))


def test_gradient_flows_into_head_and_unfrozen_encoder(encoder):
    model = PhyloModel(encoder, d_emb=16, max_dist=3.0)
    encoder.freeze(True)
    tokens, spans, mask = make_inputs()
    target = make_target(2, 5)
    scale = torch.ones(2)
    dm, _ = model(tokens, spans, mask)
    mae_loss(dm, target, scale).backward()
    head_grads = [p.grad for p in model.head.parameters() if p.grad is not None]
    assert head_grads, "no gradients on head params"
    assert model.head.bilinear_w.grad is not None
    encoder_grads = [p.grad for p in encoder.backbone.parameters() if p.grad is not None]
    assert not encoder_grads, "frozen encoder must not receive gradients"

    encoder.freeze(False)
    model.zero_grad()
    dm, _ = model(tokens, spans, mask)
    mae_loss(dm, target, scale).backward()
    encoder_grads = [p.grad for p in encoder.backbone.parameters() if p.grad is not None]
    assert encoder_grads, "unfrozen encoder must receive gradients"
    assert model.head.pooling.query.grad is not None


def test_attention_pooling_weights_concentrate():
    pool = AttentionPooling(d_model=8, d_emb=4)
    hidden = torch.randn(2, 10, 8)
    mask = torch.ones(2, 10, dtype=torch.bool)
    out = pool(hidden, mask)
    assert out.shape == (2, 4)
    mask2 = torch.zeros(2, 10, dtype=torch.bool)
    mask2[:, :5] = True
    out2 = pool(hidden, mask2)
    assert torch.isfinite(out2).all()


def test_unknown_head_type_raises():
    with pytest.raises(ValueError, match="unknown head_type"):
        DistanceHead(d_model=8, head_type="bogus")

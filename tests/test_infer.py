"""Inference tests: predict_distances guarantees + CLI + blockwise equivalence."""
import numpy as np
import pytest
import torch

from ssm_phylo.data.datasets import get_tokenizer
from ssm_phylo.infer import _load_phylo_model, predict_distances
from ssm_phylo.infer import main as infer_main
from ssm_phylo.models.encoder import build_encoder
from ssm_phylo.models.head import PhyloModel

TINY_MAMBA = {"state_size": 4, "time_step_rank": 8, "conv_kernel": 3, "expand": 2}


def tiny_cfg():
    from types import SimpleNamespace

    return SimpleNamespace(
        d_model=32, n_layer=2, vocab_size=38,
        encoder=SimpleNamespace(
            kind="from_scratch", checkpoint_dir=None,
            mamba=dict(TINY_MAMBA), ptm_model_id="ChatterjeeLab/PTM-Mamba",
        ),
    )


@pytest.fixture(scope="module")
def model():
    encoder = build_encoder(tiny_cfg(), device="cpu")
    m = PhyloModel(encoder, d_emb=16, max_dist=3.0)
    m.eval()
    return m


def random_seqs(n=7, length=40, seed=0):
    rng = np.random.default_rng(seed)
    aas = "ACDEFGHIKLMNPQRSTVWY"
    return ["".join(rng.choice(list(aas), size=length)) for _ in range(n)]


def test_predict_distances_guarantees(model):
    seqs = random_seqs()
    dm = predict_distances(model, seqs, get_tokenizer(), device=torch.device("cpu"))
    assert dm.shape == (len(seqs), len(seqs))
    assert dm.dtype == np.float32
    assert np.allclose(dm, dm.T, atol=1e-4)          # symmetric
    assert (dm >= 0).all()                           # non-negative
    assert (dm <= 3.0 + 1e-5).all()                  # bounded by max_dist
    np.fill_diagonal(dm, 1.0)
    assert (dm[np.triu_indices(len(seqs), 1)] > 0).all()


def test_predict_blockwise_equals_full(model):
    seqs = random_seqs(n=10, length=30)
    tok = get_tokenizer()
    device = torch.device("cpu")
    full = predict_distances(model, seqs, tok, device=device, chunk=64)
    block = predict_distances(model, seqs, tok, device=device, chunk=3)
    assert np.allclose(full, block, atol=1e-5)


def test_predict_blockwise_partial_blocks_match_full(model):
    """Blockwise scatter must equal the single-block path for chunk sizes
    that do not divide n (partial final j-block), including chunk=1 and
    chunk >= n. Regression guard for the clamped-slice scatter."""
    tok = get_tokenizer()
    device = torch.device("cpu")
    for n, chunk in [(10, 3), (11, 4), (5, 2), (7, 7), (8, 5), (6, 1), (9, 10**6)]:
        seqs = random_seqs(n=n, length=30, seed=n)
        full = predict_distances(model, seqs, tok, device=device, chunk=10**6)
        block = predict_distances(model, seqs, tok, device=device, chunk=chunk)
        assert np.allclose(full, block, atol=1e-5), f"n={n} chunk={chunk}"


def test_predict_partial_last_block_no_column_duplication(model):
    """The partial final j-block must not silently duplicate a column.

    n=10, chunk=3 -> j-blocks of width 3/3/3/1: the last (partial) block
    writes only column 9. If the scatter over-wrote with a repeated column,
    dm[:, 9] would equal dm[:, 8]; they are different pairs and must differ.
    """
    seqs = random_seqs(n=10, length=30, seed=3)
    tok = get_tokenizer()
    device = torch.device("cpu")
    dm = predict_distances(model, seqs, tok, device=device, chunk=3)
    assert dm.shape == (10, 10)
    assert not np.allclose(dm[:, 9], dm[:, 8], atol=1e-5), (
        "partial last block duplicated column 8 into column 9"
    )
    assert not np.allclose(dm[9, :], dm[8, :], atol=1e-5)


def test_predict_matches_model_forward(model):
    seqs = random_seqs(n=5, length=20)
    tok = get_tokenizer()
    device = torch.device("cpu")
    dm = predict_distances(model, seqs, tok, device=device, scale=1.0)
    from ssm_phylo.data.datasets import _tokenize_with_spans

    tokens, spans = _tokenize_with_spans(seqs, tok, 32768)
    with torch.no_grad():
        pred, _ = model(
            torch.from_numpy(tokens).unsqueeze(0),
            torch.tensor(spans).unsqueeze(0),
            torch.ones(1, len(spans), dtype=torch.bool),
        )
    assert np.allclose(dm, pred[0].numpy(), atol=1e-5)


def test_predict_scale_multiplies(model):
    seqs = random_seqs(n=4, length=20)
    tok = get_tokenizer()
    device = torch.device("cpu")
    base = predict_distances(model, seqs, tok, device=device, scale=1.0)
    scaled = predict_distances(model, seqs, tok, device=device, scale=2.5)
    assert np.allclose(scaled, base * 2.5, atol=1e-4)


def test_cli_writes_dm_npy(model, tmp_path):
    ckpt_path = tmp_path / "ckpt.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": {
                "d_model": 32, "n_layer": 2, "vocab_size": 38,
                "encoder": {"kind": "from_scratch", "checkpoint_dir": None,
                            "mamba": dict(TINY_MAMBA), "ptm_model_id": "x"},
                "d_emb": 16, "max_dist": 3.0, "head": "bilinear_mlp",
            },
            "scale": 1.0,
        },
        ckpt_path,
    )
    fasta = tmp_path / "seqs.fasta"
    with open(fasta, "w") as fh:
        fh.writelines(f">s{i + 1}\n{s}\n" for i, s in enumerate(random_seqs(n=4, length=20)))
    out = tmp_path / "dm.npy"
    rc = infer_main([
        "--checkpoint", str(ckpt_path),
        "--input", str(fasta),
        "--output", str(out),
        "--device", "cpu",
    ])
    assert rc == 0
    assert out.exists()
    dm = np.load(out)
    assert dm.shape == (4, 4)
    assert np.allclose(dm, dm.T, atol=1e-4)


def test_load_phylo_model_from_checkpoint(model, tmp_path):
    ckpt_path = tmp_path / "ckpt.pt"
    torch.save({"model_state": model.state_dict(), "config": {
        "d_model": 32, "n_layer": 2, "vocab_size": 38,
        "encoder": {"kind": "from_scratch", "checkpoint_dir": None,
                    "mamba": dict(TINY_MAMBA), "ptm_model_id": "x"},
        "d_emb": 16, "max_dist": 3.0, "head": "bilinear_mlp",
    }}, ckpt_path)
    loaded, ckpt = _load_phylo_model(str(ckpt_path), torch.device("cpu"))
    assert loaded.encoder.d_model == 32
    assert float(ckpt.get("scale", 1.0)) == 1.0

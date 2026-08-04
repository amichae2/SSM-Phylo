"""Evaluation tests: RF metric + end-to-end evaluate_checkpoint on tiny data."""
import csv
import os

import pytest
import torch

from ssm_phylo.data.simulation import (
    consolidate_to_parquet,
    simulate_alignments,
    simulate_trees,
)
from ssm_phylo.evaluate import (
    evaluate_checkpoint,
    tree_distances,
)
from ssm_phylo.evaluate import (
    main as evaluate_main,
)
from ssm_phylo.models.encoder import build_encoder
from ssm_phylo.models.head import PhyloModel

STEMS = [f"t004_{i:06d}" for i in range(3)]


@pytest.fixture()
def tiny_parquet(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    for stem, tips in zip(STEMS, (4, 5, 6)):
        simulate_trees(tips, 1, str(raw), seed=int(stem[-4:]), stems=[stem])
    simulate_alignments(str(raw), str(raw), length=40, seed=7, engine="python", stems=STEMS)
    out = str(tmp_path / "tiny.parquet")
    consolidate_to_parquet(str(raw), out, seed=42)
    return out


@pytest.fixture()
def tiny_checkpoint(tiny_parquet, tmp_path):
    """A checkpoint in the exact training format (config + model_state + scale)."""
    from types import SimpleNamespace

    cfg = SimpleNamespace(
        d_model=32, n_layer=2, vocab_size=38,
        encoder=SimpleNamespace(
            kind="from_scratch", checkpoint_dir=None,
            mamba={"state_size": 4, "time_step_rank": 8, "conv_kernel": 3, "expand": 2},
            ptm_model_id="ChatterjeeLab/PTM-Mamba",
        ),
    )
    model = PhyloModel(build_encoder(cfg, device="cpu"), d_emb=16, max_dist=3.0)
    path = tmp_path / "ckpt.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": {
                "d_model": 32, "n_layer": 2, "vocab_size": 38,
                "encoder": {"kind": "from_scratch", "checkpoint_dir": None,
                            "mamba": {"state_size": 4, "time_step_rank": 8,
                                      "conv_kernel": 3, "expand": 2},
                            "ptm_model_id": "x"},
                "d_emb": 16, "max_dist": 3.0, "head": "bilinear_mlp",
            },
            "scale": 1.0,
        },
        path,
    )
    return str(path)


def _write_newick(path, newick):
    with open(path, "w") as fh:
        fh.write(newick if newick.endswith(";") else newick + ";\n")


# --------------------------------------------------------------------------- #
# tree_distances
# --------------------------------------------------------------------------- #
def test_tree_distances_rf_zero_identical(tmp_path):
    p = tmp_path / "a.nwk"
    _write_newick(str(p), "((s1,s2),(s3,s4),(s5,s6));")
    assert tree_distances(str(p), str(p)) == 0.0


def test_tree_distances_rf_positive_permuted(tmp_path):
    a = tmp_path / "a.nwk"
    b = tmp_path / "b.nwk"
    _write_newick(str(a), "((s1,s2),(s3,(s4,(s5,s6))));")
    _write_newick(str(b), "((s1,s6),(s5,(s4,(s3,s2))));")
    rf = tree_distances(str(a), str(b))
    assert 0.0 < rf <= 1.0


def test_tree_distances_unsupported_metric(tmp_path):
    p = tmp_path / "a.nwk"
    _write_newick(str(p), "((s1,s2),(s3,s4));")
    with pytest.raises(NotImplementedError, match="baselines phase"):
        tree_distances(str(p), str(p), metric="quartet")


# --------------------------------------------------------------------------- #
# evaluate_checkpoint
# --------------------------------------------------------------------------- #
def test_evaluate_checkpoint_end_to_end(tiny_checkpoint, tiny_parquet, tmp_path):
    out_dir = str(tmp_path / "results")
    summary = evaluate_checkpoint(
        tiny_checkpoint, tiny_parquet, out_dir,
        max_alignments=3, device=torch.device("cpu"),
    )
    assert summary["count"] == 3
    assert set(summary["bins"]) == {4, 5, 6}
    results_csv = os.path.join(out_dir, "results.csv")
    assert os.path.exists(results_csv)
    with open(results_csv) as fh:
        rows = list(csv.DictReader(fh))
    assert set(rows[0]) == {
        "index", "n_tips", "seq_len", "rf_pred",
        "rf_true_dist_upper_bound", "elapsed_s",
    }
    # the upper bound (NJ on TRUE distances) must recover the tree: RF ~ 0
    assert all(float(r["rf_true_dist_upper_bound"]) < 1e-6 for r in rows)
    # rf_pred is a valid probability
    assert all(0.0 <= float(r["rf_pred"]) <= 1.0 for r in rows)


def test_evaluate_cli(tiny_checkpoint, tiny_parquet, tmp_path):
    out_dir = str(tmp_path / "cli_results")
    rc = evaluate_main([
        "--checkpoint", tiny_checkpoint,
        "--test-parquet", tiny_parquet,
        "--out-dir", out_dir,
        "--max-alignments", "2",
        "--device", "cpu",
    ])
    assert rc == 0
    assert os.path.exists(os.path.join(out_dir, "results.csv"))

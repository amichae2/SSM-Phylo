"""Loss tests: four-point condition, scale behavior, masking."""
import numpy as np
import pytest
import torch

from ssm_phylo.models.losses import (
    combined_loss,
    four_point_penalty,
    mae_loss,
    mre_loss,
)


def additive_tree_matrix(n=6, seed=0):
    """Patristic distances of a random dendropy tree (a perfect tree metric)."""
    import random

    from dendropy.simulate import treesim

    random.seed(seed)
    tree = treesim.birth_death_tree(birth_rate=1.0, death_rate=0.5, num_extant_tips=n)
    leaves = tree.leaf_nodes()
    ndm = tree.node_distance_matrix()
    m = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            m[i, j] = ndm(leaves[i], leaves[j])
    return torch.from_numpy(m)


def random_matrix(n=8, seed=1):
    rng = np.random.default_rng(seed)
    m = rng.random((n, n))
    m = (m + m.T) / 2
    np.fill_diagonal(m, 0.0)
    return torch.from_numpy(m.astype(np.float32))


def valid_target(n=5, batch=2):
    t = torch.rand(batch, n, n) * 2.0
    t = torch.triu(t, diagonal=1)
    return t + t.transpose(1, 2)


def test_four_point_zero_for_additive_tree_metric():
    dm = additive_tree_matrix(n=6)
    p = four_point_penalty(dm)
    assert float(p) == pytest.approx(0.0, abs=1e-5)


def test_four_point_positive_for_random_matrix():
    dm = random_matrix(n=8)
    assert float(four_point_penalty(dm)) > 1e-4


def test_four_point_sampled_for_large_n():
    dm = random_matrix(n=60)
    p = four_point_penalty(dm)  # n > 40 -> seeded sample of <= 4096 quartets
    assert float(p) > 1e-4
    small = four_point_penalty(dm, sample_quartets=64)
    assert torch.isfinite(small)


def test_four_point_ignores_padded_entries():
    dm = random_matrix(n=6)
    padded = dm.clone()
    padded[1:, :] = -1.0  # kill every quartet touching taxa 1..5 -> none valid
    padded[:, 1:] = -1.0
    p = four_point_penalty(padded.unsqueeze(0))
    assert float(p) == 0.0


def test_mae_scale_dependent():
    pred = torch.rand(2, 4, 4) * 1.5
    target = valid_target(4)
    ones = torch.ones(2)
    twice = torch.full((2,), 2.0)
    l1 = mae_loss(pred, target, ones)
    l2 = mae_loss(pred, target, twice)
    assert float(l2) == pytest.approx(2.0 * float(l1))


def test_mre_scale_invariant():
    pred = torch.rand(2, 4, 4) * 1.5 + 0.5
    target = valid_target(4) + 0.5
    k = 2.0
    assert float(mre_loss(pred * k, target * k)) == pytest.approx(
        float(mre_loss(pred, target)), rel=1e-3
    )


def test_mae_counts_only_ij_pairs_with_mask():
    n = 4
    pred = torch.full((1, n, n), 1.0)
    target = valid_target(n, batch=1)
    target[0, 3, :] = -1.0
    target[0, :, 3] = -1.0  # taxon 3 padded -> pairs (0,3),(1,3),(2,3) invalid
    mask = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1) & (target[0] >= 0)
    expected = ((pred[0] - target[0]).abs()[mask]).mean()
    got = mae_loss(pred, target, torch.ones(1))
    assert float(got) == pytest.approx(float(expected))
    # without the padding this would be the full 6 pairs
    full = mae_loss(pred, valid_target(n, batch=1), torch.ones(1))
    assert float(got) != float(full)


def test_combined_loss_dict_and_composition():
    pred = (torch.rand(2, 5, 5) * 1.5).requires_grad_(True)
    target = valid_target(5)
    scale = torch.tensor([1.0, 2.0])
    out = combined_loss(pred, target, scale, loss_type="mae", lambda_fp=0.01)
    assert set(out) == {"loss", "primary", "four_point", "mae_norm", "mae_raw"}
    assert torch.allclose(out["loss"], out["primary"] + 0.01 * out["four_point"])
    assert float(out["mae_raw"]) == pytest.approx(float(out["primary"]))
    assert float(out["mae_norm"]) <= float(out["mae_raw"]) + 1e-6  # scale >= 1
    assert out["loss"].requires_grad


def test_combined_loss_mre_primary():
    pred = torch.rand(2, 5, 5) * 1.5 + 0.5
    target = valid_target(5) + 0.5
    out = combined_loss(pred, target, torch.ones(2), loss_type="mre", lambda_fp=0.0)
    assert float(out["primary"]) == pytest.approx(
        float(mre_loss(pred, target))
    )
    assert torch.allclose(out["loss"], out["primary"])

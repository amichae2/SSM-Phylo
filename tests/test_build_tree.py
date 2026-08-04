"""Tree-building tests: PHYLIP writer, FastME->NJ fallback, perfect-matrix recovery."""
import logging

import dendropy
import numpy as np
import pytest

from ssm_phylo.build_tree import _neighbor_joining, fastme, write_phylip
from ssm_phylo.evaluate import _patristic_matrix, tree_distances


# --------------------------------------------------------------------------- #
# write_phylip
# --------------------------------------------------------------------------- #
def test_write_phylip_roundtrip(tmp_path):
    dm = np.array([[0.0, 1.5, 2.0], [1.5, 0.0, 0.5], [2.0, 0.5, 0.0]], dtype=np.float64)
    path = str(tmp_path / "dm.phy")
    write_phylip(dm, path)
    with open(path) as fh:
        lines = fh.read().splitlines()
    assert lines[0].strip() == "3"
    assert len(lines) == 4
    row0 = lines[1].split()
    assert row0[0] == "s1"
    assert np.allclose([float(x) for x in row0[1:]], dm[0])


def test_write_phylip_rejects_non_square(tmp_path):
    with pytest.raises(ValueError, match="square"):
        write_phylip(np.zeros((2, 3)), str(tmp_path / "bad.phy"))


# --------------------------------------------------------------------------- #
# fastme fallback
# --------------------------------------------------------------------------- #
def test_fastme_falls_back_to_nj_with_bogus_binary(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("FASTME_BIN", "/nonexistent/fastme")
    dm = np.array([[0.0, 1.0, 1.5], [1.0, 0.0, 1.0], [1.5, 1.0, 0.0]])
    out = str(tmp_path / "tree.nwk")
    with caplog.at_level(logging.WARNING, logger="ssm_phylo.build_tree"):
        result = fastme(dm, out, binary="/nonexistent/fastme")
    assert result == out
    assert "FastME not found" in caplog.text
    assert "NJ fallback" in caplog.text
    tree = dendropy.Tree.get(path=out, schema="newick")
    assert len(tree.leaf_nodes()) == 3


def test_perfect_matrix_recovers_topology(tmp_path):
    """NJ on the exact patristic matrix of a 10-tip tree recovers it (RF 0)."""
    import random

    from dendropy.simulate import treesim

    random.seed(42)
    tree = treesim.birth_death_tree(birth_rate=1.0, death_rate=0.5, num_extant_tips=10)
    for i, leaf in enumerate(tree.leaf_nodes()):
        leaf.taxon.label = f"s{i + 1}"
    true_path = str(tmp_path / "true.nwk")
    tree.write(path=true_path, schema="newick", suppress_rooting=True)

    dm = _patristic_matrix(tree.as_string(schema="newick"))
    pred_path = str(tmp_path / "pred.nwk")
    fastme(dm, pred_path)  # no fastme binary in CI -> NJ fallback
    assert tree_distances(true_path, pred_path) == pytest.approx(0.0, abs=1e-6)


def test_neighbor_joining_direct(tmp_path):
    dm = np.array([[0.0, 2.0, 2.0], [2.0, 0.0, 2.0], [2.0, 2.0, 0.0]])
    out = str(tmp_path / "t.nwk")
    _neighbor_joining(dm, out)
    assert len(dendropy.Tree.get(path=out, schema="newick").leaf_nodes()) == 3

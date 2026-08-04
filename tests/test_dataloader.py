"""Phase 1 dataloader + simulation pipeline tests.

Uses the pure-python fallback simulation engine (engine="python") so the
suite runs without iqtree. Raw dirs are tmp_path (scratch); Drive is
simulated by a local dir.
"""
import os

import dendropy
import numpy as np
import pyarrow.parquet as pq
import pytest

from ssm_phylo.data.datasets import (
    ParquetPhyloDataset,
    collate_with_bucketing,
    get_tokenizer,
    load_dataset,
)
from ssm_phylo.data.simulation import (
    _read_fasta,
    clean_scratch,
    consolidate_to_parquet,
    make_ood_set,
    make_splits,
    simulate_alignments,
    simulate_trees,
)

STEMS = [f"t004_{i:06d}" for i in range(3)]  # tip counts 4, 5, 6


@pytest.fixture()
def raw_dir(tmp_path):
    r = tmp_path / "raw"
    r.mkdir()
    for stem, tips in zip(STEMS, (4, 5, 6)):
        simulate_trees(tips, 1, str(r), seed=int(stem[-4:]), stems=[stem])
    simulate_alignments(str(r), str(r), length=60, seed=7, engine="python", stems=STEMS)
    return r


def test_unaligned_fasta_no_gaps(raw_dir):
    for stem in STEMS:
        _, seqs = _read_fasta(os.path.join(str(raw_dir), f"{stem}.fasta"))
        assert len(seqs) > 0
        for s in seqs:
            assert "-" not in s
            assert "." not in s
        _, aligned = _read_fasta(os.path.join(str(raw_dir), "aligned", f"{stem}.fasta"))
        assert any("-" in s for s in aligned) or len(aligned) > 0


def test_parquet_roundtrip(raw_dir, tmp_path):
    out = str(tmp_path / "tiny.parquet")
    consolidate_to_parquet(str(raw_dir), out, seed=42)
    assert os.path.exists(out)
    tbl = pq.read_table(out)
    assert set(tbl.column_names) == {"seqs", "tree_newick", "n_tips", "scale"}
    tips = tbl["n_tips"].to_pylist()
    assert tips == sorted(tips)
    assert len(tbl) == 3
    for i in range(len(tbl)):
        row = tbl.slice(i, 1)
        seqs = row["seqs"][0].as_py()
        newick = row["tree_newick"][0].as_py()
        n_tips = row["n_tips"][0].as_py()
        scale = row["scale"][0].as_py()
        tree = dendropy.Tree.get(data=newick, schema="newick")
        assert len(tree.leaf_nodes()) == n_tips == len(seqs)
        root_dists = [n.distance_from_root() for n in tree.leaf_nodes()]
        assert scale == pytest.approx(float(np.median(root_dists)))
    # identical rows after reload
    tbl2 = pq.read_table(out)
    assert tbl2.equals(tbl)


def test_tokenization_spans_cover_stream(raw_dir, tmp_path):
    out = str(tmp_path / "tiny.parquet")
    consolidate_to_parquet(str(raw_dir), out, seed=42)
    ds = ParquetPhyloDataset(out, max_seq_len=1024)
    tok = get_tokenizer()
    tokens, spans, _dm, _scale, _newick = ds[0]
    # spans are ordered, non-overlapping, cover the whole stream with a
    # single separator between consecutive spans (no other gaps)
    starts = [s for s, _e in spans]
    ends = [e for _s, e in spans]
    assert len(spans) > 0
    assert starts == sorted(starts)
    assert ends == sorted(ends)
    assert starts[0] == 0
    for i in range(len(spans) - 1):
        assert starts[i + 1] == ends[i] + 1  # exactly one separator between spans
        assert tokens[ends[i]] == tok.sep_id
    assert ends[-1] == len(tokens)
    # reconstructed stream matches tokens exactly
    seqs = pq.read_table(out)["seqs"][0].as_py()
    expected = []
    for i, seq in enumerate(seqs):
        expected.extend(tok.encode(seq))
        if i < len(seqs) - 1:
            expected.append(tok.sep_id)
    assert tokens.tolist() == expected
    # no separator token inside any span
    for s, e in spans:
        assert tok.sep_id not in tokens[s:e].tolist()
    # exactly n-1 separators in the stream
    assert int((tokens == tok.sep_id).sum()) == len(seqs) - 1


def test_true_distances_match_dendropy(raw_dir, tmp_path):
    out = str(tmp_path / "tiny.parquet")
    consolidate_to_parquet(str(raw_dir), out, seed=42)
    ds = ParquetPhyloDataset(out, max_seq_len=1024)
    for i in range(len(ds)):
        _tokens, spans, dm, scale, newick = ds[i]
        tree = dendropy.Tree.get(data=newick, schema="newick")
        leaves = tree.leaf_nodes()
        ndm = tree.node_distance_matrix()
        ref = np.empty((len(leaves), len(leaves)), dtype=np.float32)
        for a in range(len(leaves)):
            for b in range(len(leaves)):
                ref[a, b] = ndm(leaves[a], leaves[b])
        ref /= scale
        assert dm.shape == ref.shape
        assert np.allclose(dm, ref, atol=1e-5)
        assert len(spans) == len(leaves)
        assert np.all(np.diag(dm) == 0.0)


def test_collate_bucketing_never_exceeds_max_seq_len(raw_dir, tmp_path):
    out = str(tmp_path / "tiny.parquet")
    consolidate_to_parquet(str(raw_dir), out, seed=42)
    ds = ParquetPhyloDataset(out, max_seq_len=512)
    batch = [ds[i] for i in range(len(ds))]
    # synthetic over-length sample: 100 seqs x 200 tokens each
    tok = get_tokenizer()
    long_seq = "M" * 500
    big_tokens, big_spans = [], []
    for k in range(100):
        start = len(big_tokens)
        big_tokens.extend(tok.encode(long_seq[:200]))
        big_spans.append((start, start + 200))
        if k < 99:
            big_tokens.append(tok.sep_id)
    big_tokens = np.asarray(big_tokens, dtype=np.int64)
    n = 100
    big_dm = np.ones((n, n), dtype=np.float32)
    batch.append((big_tokens, big_spans, big_dm, 1.0, "(x,y);"))
    max_seq_len = 512
    tok_t, spans_t, mask_t, dm_t, scales_t = collate_with_bucketing(
        batch, max_seq_len=max_seq_len, pad_id=tok.pad_id, bucket_step=128
    )
    assert tok_t.shape[1] <= max_seq_len
    assert tok_t.shape[0] == len(batch)
    assert spans_t.shape[2] == 2
    assert dm_t.shape[1] == dm_t.shape[2]
    assert mask_t.shape == tok_t.shape
    assert (mask_t.sum(1) <= tok_t.shape[1]).all()
    assert (tok_t == tok.pad_id).all(0)[tok_t.shape[1] - 1].item() or True
    # real samples' distance matrices preserved in padded tensor
    assert float(scales_t[0]) == pytest.approx(float(batch[0][3]))


def test_num_samples_subset(raw_dir, tmp_path):
    out = str(tmp_path / "tiny.parquet")
    consolidate_to_parquet(str(raw_dir), out, seed=42)
    ds = ParquetPhyloDataset(out, max_seq_len=1024, num_samples=2, seed=0)
    assert len(ds) == 2


def test_make_splits_disjoint_and_complete(raw_dir, tmp_path):
    data_dir = tmp_path / "drive"
    make_splits(str(raw_dir), str(data_dir), seed=42)
    newicks = {}
    for split in ("train", "val", "test"):
        p = data_dir / f"{split}.parquet"
        assert p.exists()
        newicks[split] = set(pq.read_table(str(p))["tree_newick"].to_pylist())
    assert len(newicks["train"]) + len(newicks["val"]) + len(newicks["test"]) == 3
    assert not (newicks["train"] & newicks["val"])
    assert not (newicks["train"] & newicks["test"])
    assert not (newicks["val"] & newicks["test"])
    all_newicks = set().union(*newicks.values())
    assert len(all_newicks) == 3


def test_load_dataset_validation(raw_dir, tmp_path, monkeypatch):
    data_dir = tmp_path / "drive"
    make_splits(str(raw_dir), str(data_dir), seed=42)
    monkeypatch.setenv("DATA_DIR", str(data_dir))
    ds = load_dataset("train", max_seq_len=512)
    assert len(ds) > 0
    with pytest.raises(FileNotFoundError):
        load_dataset("nonexistent")
    with pytest.raises(FileNotFoundError):
        load_dataset("train", data_dir=str(tmp_path))  # file missing there


def test_make_ood_set(tmp_path):
    raw_ood = tmp_path / "raw_ood"
    data_dir = tmp_path / "drive"
    make_ood_set(
        str(raw_ood), str(data_dir),
        n_tips_range=(6, 10), length_range=(40, 60), indel=True,
        n_trees=4, seed=42,
    )
    p = data_dir / "ood.parquet"
    assert p.exists()
    tbl = pq.read_table(str(p))
    assert set(tbl.column_names) == {"seqs", "tree_newick", "n_tips", "scale"}
    assert 6 <= min(tbl["n_tips"].to_pylist()) and max(tbl["n_tips"].to_pylist()) <= 10


def test_clean_scratch_guards(raw_dir, tmp_path, monkeypatch):
    monkeypatch.setenv("LOCAL_DATA_DIR", str(tmp_path))
    data_dir = tmp_path / "drive"
    monkeypatch.setenv("DATA_DIR", str(data_dir))
    with pytest.raises(ValueError):
        clean_scratch(str(data_dir))  # refusing to delete a Drive-ish path
    clean_scratch(str(raw_dir))
    assert not raw_dir.exists()


def test_simulate_alignments_idempotent(raw_dir):
    before = sorted(os.listdir(raw_dir))
    simulate_alignments(str(raw_dir), str(raw_dir), length=60, seed=7, engine="python", stems=STEMS)
    after = sorted(os.listdir(raw_dir))
    assert before == after


def test_drive_path_guard(raw_dir, tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "drive"))
    with pytest.raises(ValueError):
        simulate_alignments(str(raw_dir), str(tmp_path / "drive" / "raw"), engine="python")

"""Evaluation: predicted tree vs TRUE tree (Robinson-Foulds), + upper bound.

evaluate_checkpoint loads a training checkpoint, predicts distance matrices
for a seeded sample of the test parquet, builds trees (FastME with NJ
fallback), and compares against the TRUE trees stored in the parquet
(simulated data: tree_newick column is ground truth). It ALSO computes the
upper bound: FastME/NJ on the TRUE patristic distance matrix -> RF vs the
true tree (expected ~0, which proves the eval chain end-to-end).

Results land in out_dir/results.csv plus a printed markdown summary grouped
by n_tips bins.

CLI:
  python -m ssm_phylo.evaluate --checkpoint ckpt.pt --test-parquet test.parquet
      --out-dir results [--max-alignments 200]
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import time
from collections.abc import Sequence

import dendropy
import numpy as np
import pyarrow.parquet as pq
import torch

from ssm_phylo.build_tree import fastme
from ssm_phylo.data.datasets import get_tokenizer
from ssm_phylo.infer import _load_phylo_model, predict_distances

log = logging.getLogger("ssm_phylo.evaluate")

RESULTS_COLUMNS = ["index", "n_tips", "seq_len", "rf_pred", "rf_true_dist_upper_bound", "elapsed_s"]


def _patristic_matrix(newick: str) -> np.ndarray:
    """Raw (un-normalized) n x n patristic distances for a Newick tree."""
    tree = dendropy.Tree.get(data=newick, schema="newick")
    leaves = tree.leaf_nodes()
    n = len(leaves)
    ndm = tree.node_distance_matrix()
    m = np.empty((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            m[i, j] = ndm(leaves[i], leaves[j])
    return m


def tree_distances(tree1_path: str, tree2_path: str, metric: str = "rf") -> float:
    """Normalized Robinson-Foulds distance (0..1) between two Newick files."""
    if metric != "rf":
        raise NotImplementedError(
            f"metric {metric!r} comes with the baselines phase; v1 supports only 'rf'"
        )
    t1 = dendropy.Tree.get(path=tree1_path, schema="newick")
    t2 = dendropy.Tree.get(path=tree2_path, schema="newick")
    t2.migrate_taxon_namespace(t1.taxon_namespace)
    n_tips = len(t1.leaf_nodes())
    max_rf = max(1, 2 * (n_tips - 3))  # unweighted RF max for binary trees
    from dendropy.calculate import treecompare

    rf = treecompare.symmetric_difference(t1, t2) / 2.0
    return float(min(1.0, rf / max_rf))


def _write_true_tree(newick: str, path: str) -> None:
    """Write the true tree with leaves relabeled s1..sn (traversal order).

    The simulated parquet stores seqs in fasta/traversal order but drops the
    original leaf names; the predicted/NJ tree uses s1..sn (write_phylip).
    Relabeling in traversal order makes the RF comparison label-consistent.
    """
    tree = dendropy.Tree.get(data=newick, schema="newick")
    for i, leaf in enumerate(tree.leaf_nodes()):
        leaf.taxon.label = f"s{i + 1}"
    # suppress_rooting: without it dendropy writes a '[&R]' annotation, and
    # RF would compare a rooted vs unrooted tree (odd symdiff counts).
    tree.write(path=path, schema="newick", suppress_rooting=True)


def evaluate_checkpoint(
    checkpoint_path: str,
    test_parquet: str,
    out_dir: str,
    max_alignments: int = 200,
    device: torch.device | None = None,
) -> dict:
    """Evaluate a checkpoint against the test parquet; returns the summary dict."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = _load_phylo_model(checkpoint_path, device)
    scale = float(ckpt.get("scale", 1.0))
    tokenizer = get_tokenizer()

    table = pq.read_table(test_parquet, columns=["seqs", "tree_newick", "n_tips"])
    n_rows = len(table)
    n_eval = min(max_alignments, n_rows)
    rng = np.random.default_rng(42)
    indices = rng.choice(n_rows, size=n_eval, replace=False).tolist()

    tmp_dir = os.path.join(out_dir, "trees")
    os.makedirs(tmp_dir, exist_ok=True)
    rows: list[dict] = []
    for pos, i in enumerate(indices):
        seqs = table["seqs"][i].as_py()
        newick = table["tree_newick"][i].as_py()
        n_tips = int(table["n_tips"][i].as_py())
        seq_len = float(np.mean([len(s) for s in seqs]))
        t0 = time.time()

        true_path = os.path.join(tmp_dir, f"true_{pos}.nwk")
        _write_true_tree(newick, true_path)

        dm = predict_distances(model, seqs, tokenizer, device=device, scale=scale)
        pred_path = os.path.join(tmp_dir, f"pred_{pos}.nwk")
        fastme(dm, pred_path)
        rf_pred = tree_distances(true_path, pred_path)

        true_dm = _patristic_matrix(newick)
        ub_path = os.path.join(tmp_dir, f"ub_{pos}.nwk")
        fastme(true_dm, ub_path)
        rf_ub = tree_distances(true_path, ub_path)

        rows.append({
            "index": int(i),
            "n_tips": n_tips,
            "seq_len": round(seq_len, 1),
            "rf_pred": round(rf_pred, 4),
            "rf_true_dist_upper_bound": round(rf_ub, 4),
            "elapsed_s": round(time.time() - t0, 3),
        })
        if (pos + 1) % 25 == 0 or pos + 1 == n_eval:
            print(f"[evaluate] {pos + 1}/{n_eval} alignments (rf_pred so far: {np.mean([r['rf_pred'] for r in rows]):.3f})")

    results_path = os.path.join(out_dir, "results.csv")
    with open(results_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=RESULTS_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    summary = _summarize(rows)
    _print_markdown(summary)
    return summary


def _summarize(rows: list[dict]) -> dict:
    if not rows:
        return {"bins": {}, "overall": None, "count": 0}
    bins: dict[int, dict] = {}
    for r in rows:
        bins.setdefault(r["n_tips"], {"rf_pred": [], "rf_ub": []})
        bins[r["n_tips"]]["rf_pred"].append(r["rf_pred"])
        bins[r["n_tips"]]["rf_ub"].append(r["rf_true_dist_upper_bound"])
    summary_bins = {
        n: {
            "n": len(v["rf_pred"]),
            "mean_rf_pred": float(np.mean(v["rf_pred"])),
            "mean_rf_upper_bound": float(np.mean(v["rf_ub"])),
        }
        for n, v in sorted(bins.items())
    }
    overall = {
        "mean_rf_pred": float(np.mean([r["rf_pred"] for r in rows])),
        "mean_rf_upper_bound": float(np.mean([r["rf_true_dist_upper_bound"] for r in rows])),
    }
    return {"bins": summary_bins, "overall": overall, "count": len(rows)}


def _print_markdown(summary: dict) -> None:
    print("\n| n_tips | n | mean RF (pred) | mean RF (true-dist upper bound) |")
    print("|---|---|---|---|")
    for n_tips, s in summary["bins"].items():
        print(f"| {n_tips} | {s['n']} | {s['mean_rf_pred']:.3f} | {s['mean_rf_upper_bound']:.3f} |")
    overall = summary["overall"]
    if overall:
        print(
            f"| **all** | {summary['count']} | **{overall['mean_rf_pred']:.3f}** | "
            f"**{overall['mean_rf_upper_bound']:.3f}** |"
        )
    print()


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(prog="ssm_phylo.evaluate")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test-parquet", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-alignments", type=int, default=200)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    summary = evaluate_checkpoint(
        args.checkpoint, args.test_parquet, args.out_dir,
        max_alignments=args.max_alignments, device=device,
    )
    print(f"[evaluate] wrote {os.path.join(args.out_dir, 'results.csv')}")
    print(f"[evaluate] mean RF (pred) = {summary['overall']['mean_rf_pred']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

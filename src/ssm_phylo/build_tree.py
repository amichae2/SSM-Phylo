"""Tree building: distance matrix -> Newick tree (FastME, Biopython NJ fallback).

fastme() resolves the binary as: env FASTME_BIN -> shutil.which("fastme") ->
scripts/vendor/bin_linux/fastme (if a bundled binary exists). When no binary
is found, a clear warning is logged and Biopython's neighbor-joining is used
(topology may differ — FastME is the production tree builder; NJ keeps the
V1 chain working on CPU-only boxes).

CLI:
  python -m ssm_phylo.build_tree --input dm.npy --output tree.nwk
"""
from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np

log = logging.getLogger("ssm_phylo.build_tree")

REPO_ROOT = Path(__file__).resolve().parents[2]
BUNDLED_FASTME = REPO_ROOT / "scripts" / "vendor" / "bin_linux" / "fastme"
NJ_WARNING = "FastME not found; using Biopython NJ fallback — topology may differ"


def _resolve_fastme(binary: str | None = None) -> str | None:
    if binary:
        return binary if os.path.exists(binary) or shutil.which(binary) else None
    env_bin = os.environ.get("FASTME_BIN")
    if env_bin:
        return env_bin if os.path.exists(env_bin) or shutil.which(env_bin) else None
    found = shutil.which("fastme")
    if found:
        return found
    if BUNDLED_FASTME.exists():
        return str(BUNDLED_FASTME)
    return None


def write_phylip(dm: np.ndarray, path: str) -> None:
    """Write a simple PHYLIP distance matrix (count on line 1, then rows)."""
    dm = np.asarray(dm, dtype=np.float64)
    n = dm.shape[0]
    if dm.shape != (n, n):
        raise ValueError(f"distance matrix must be square, got {dm.shape}")
    with open(path, "w") as fh:
        fh.write(f"{n}\n")
        for i in range(n):
            row = " ".join(f"{dm[i, j]:.6f}" for j in range(n))
            fh.write(f"s{i + 1} {row}\n")


def _neighbor_joining(dm: np.ndarray, out_tree: str) -> None:
    """Biopython NJ fallback (CPU-only boxes without a fastme binary)."""
    from Bio.Phylo.TreeConstruction import DistanceMatrix, DistanceTreeConstructor

    n = dm.shape[0]
    names = [f"s{i + 1}" for i in range(n)]
    lower = [dm[i, : i + 1].tolist() for i in range(n)]
    matrix = DistanceMatrix(names, lower)
    tree = DistanceTreeConstructor(method="nj").nj(matrix)
    from Bio import Phylo

    Phylo.write(tree, out_tree, "newick")


def fastme(dm: np.ndarray, out_tree: str, binary: str | None = None,
           nni: bool = True, spr: bool = True) -> str:
    """Build a tree from a distance matrix; FastME if available else NJ.

    Returns the output tree path.
    """
    dm = np.asarray(dm, dtype=np.float64)
    fastme_bin = _resolve_fastme(binary)
    if fastme_bin is None:
        log.warning(NJ_WARNING)
        _neighbor_joining(dm, out_tree)
        return out_tree
    with tempfile.TemporaryDirectory(prefix="fastme_") as tmp:
        phy = os.path.join(tmp, "dm.phy")
        write_phylip(dm, phy)
        cmd = [fastme_bin, "-i", phy, "-o", out_tree]
        if nni:
            cmd.append("--nni")
        if spr:
            cmd.append("--spr")
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            log.warning(
                "fastme failed (rc=%d): %s; %s",
                proc.returncode, (proc.stderr or proc.stdout)[-300:], NJ_WARNING,
            )
            _neighbor_joining(dm, out_tree)
            return out_tree
    if not os.path.exists(out_tree):
        log.warning("fastme produced no output; %s", NJ_WARNING)
        _neighbor_joining(dm, out_tree)
    return out_tree


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(prog="ssm_phylo.build_tree")
    parser.add_argument("--input", required=True, help="distance matrix .npy")
    parser.add_argument("--output", required=True, help="output tree .nwk")
    parser.add_argument("--binary", default=None, help="fastme binary override")
    parser.add_argument("--no-nni", action="store_true")
    parser.add_argument("--no-spr", action="store_true")
    args = parser.parse_args(argv)

    dm = np.load(args.input)
    if dm.ndim == 3:
        dm = dm[0]  # stacked matrices: use the first
    fastme(dm, args.output, binary=args.binary, nni=not args.no_nni, spr=not args.no_spr)
    print(f"[build_tree] wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

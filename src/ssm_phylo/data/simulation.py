"""Simulation and consolidation pipeline for ssm-phylo.

Storage rules (AGENTS.md):
- Simulated raw data is written to LOCAL scratch ONLY (never $DATA_DIR on
  Drive). simulate_alignments raises ValueError if out_dir/tree_dir is inside
  $DATA_DIR.
- Raw scratch is consolidated into ONE snappy parquet per split
  (columns: seqs list[str], tree_newick str, n_tips int, scale float), which
  is then copied to Drive ATOMICALLY (tmp file + os.replace).

Engines:
- Trees: vendored Phyloformer simulate_trees.py when PHYLOFORMER_DIR is set
  and the vendor script exists (or can be copied), else internal dendropy
  (birth_death_tree / yule).
- Alignments: AliSim via `iqtree --alisim` (engine "auto" resolves
  $IQTREE_BIN or shutil.which("iqtree")); the pure-python engine
  (engine="python") is a deterministic dev/smoke fallback with no external
  binary.

All gap characters ('-', '.') are STRIPPED from the unaligned FASTA; an
aligned/ subdirectory copy is kept for baseline experiments only.
Every function is resumable/idempotent and logs progress every 100 files.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path

import dendropy
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from dendropy.simulate import treesim

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
REPO_ROOT = Path(__file__).resolve().parents[3]
VENDOR_DIR = REPO_ROOT / "scripts" / "vendor"
VENDOR_TREE_SCRIPT = "simulate_trees.py"
VENDOR_DATA = ("hogenom_diams.txt", "raxml_diams.txt")

SCHEMA = pa.schema(
    [
        pa.field("seqs", pa.list_(pa.string())),
        pa.field("tree_newick", pa.string()),
        pa.field("n_tips", pa.int64()),
        pa.field("scale", pa.float64()),
    ]
)

_TREE_EXTS = (".nwk", ".newick")
_ALN_EXTS = (".fasta", ".fa")
LOG_EVERY = 100


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _env(name: str, default: str | None = None) -> str | None:
    v = os.environ.get(name)
    return v if v not in (None, "") else default


def _guard_not_drive(path: str, what: str = "output") -> None:
    """Raise ValueError if `path` is inside $DATA_DIR (simulation writes scratch only)."""
    target = os.path.abspath(os.path.expanduser(path))
    data_dir = _env("DATA_DIR")
    if data_dir:
        drive = os.path.abspath(os.path.expanduser(data_dir))
        if target == drive or target.startswith(drive + os.sep):
            raise ValueError(
                f"{what} path {path!r} is inside $DATA_DIR ({drive}); simulated "
                "data must be written to LOCAL scratch only"
            )


def _hash_frac(seed: int, salt: str) -> float:
    h = hashlib.sha256(f"{seed}:{salt}".encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def _hash_int(seed: int, salt: str) -> int:
    h = hashlib.sha256(f"{seed}:{salt}".encode()).hexdigest()
    return int(h[:16], 16)


def _log(stem: str, idx: int, total: int) -> None:
    if idx % LOG_EVERY == 0:
        print(f"[simulation] {stem} ({idx}/{total})", flush=True)


def _read_fasta(path: str) -> tuple[list[str], list[str]]:
    """Return (headers, sequences) in file order; tolerant of wrapped lines."""
    names: list[str] = []
    seqs: list[str] = []
    cur_name: str | None = None
    cur: list[str] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur_name is not None:
                    seqs.append("".join(cur))
                names.append(line[1:].split()[0])
                cur_name = line[1:].split()[0]
                cur = []
            else:
                cur.append(line)
    if cur_name is not None:
        seqs.append("".join(cur))
    return names, seqs


def _write_fasta(path: str, seqs: Sequence[str], names: Sequence[str] | None = None) -> None:
    with open(path, "w") as fh:
        for i, s in enumerate(seqs):
            label = names[i] if names and i < len(names) else f"s{i + 1}"
            fh.write(f">{label}\n{s}\n")


def _strip_gaps(seq: str) -> str:
    return seq.replace("-", "").replace(".", "")


def _has_duplicates(seqs: Sequence[str]) -> bool:
    seen = set()
    for s in seqs:
        if s in seen:
            return True
        seen.add(s)
    return False


def _atomic_copy(src: str, dst: str) -> None:
    """Single-file atomic copy: temp file next to dst, then os.replace."""
    tmp = dst + ".tmp"
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)


def _find_iqtree(iqtree_bin: str | None) -> str | None:
    if iqtree_bin:
        return iqtree_bin if os.path.exists(iqtree_bin) or shutil.which(iqtree_bin) else None
    return _env("IQTREE_BIN") or shutil.which("iqtree")


# --------------------------------------------------------------------------- #
# tree simulation
# --------------------------------------------------------------------------- #
def _target_diameter(rng: np.random.Generator) -> float:
    """Sample a tree diameter in expected-substitutions-per-site units."""
    return float(np.clip(rng.normal(1.2, 0.3), 0.1, 3.0))


def _rescale_diameter(tree: dendropy.Tree, target: float) -> dendropy.Tree:
    root_dists = [n.distance_from_root() for n in tree.leaf_nodes()]
    est_diam = 2.0 * max(root_dists) if root_dists else 1.0
    if est_diam <= 0.0:
        return tree
    factor = target / est_diam
    for node in tree.preorder_node_iter():
        if node is not tree.seed_node and node.edge_length:
            node.edge_length *= factor
    return tree


def _vendor_phyloformer(phyloformer_dir: str) -> Path:
    """Idempotently vendor-copy Phyloformer's simulate_trees.py + data files."""
    src = Path(phyloformer_dir)
    VENDOR_DIR.mkdir(parents=True, exist_ok=True)
    if not (VENDOR_DIR / VENDOR_TREE_SCRIPT).exists():
        shutil.copy2(src / VENDOR_TREE_SCRIPT, VENDOR_DIR / VENDOR_TREE_SCRIPT)
    data_src = src / "data"
    if data_src.is_dir():
        for name in VENDOR_DATA:
            dst = VENDOR_DIR / name
            if not dst.exists() and (data_src / name).exists():
                shutil.copy2(data_src / name, dst)
    return VENDOR_DIR


def simulate_trees(
    ntips: int,
    ntrees: int,
    out_dir: str,
    seed: int = 42,
    stems: Sequence[str] | None = None,
    tree_model: str = "birth-death",
) -> list[Path]:
    """Simulate ntrees Newick trees (each with ntips leaves) into out_dir.

    Writes one '{stem}.nwk' per tree; stems auto-generate as
    't{ntips:03d}_{i:06d}' when not given. When PHYLOFORMER_DIR is set the
    vendored Phyloformer simulate_trees.py is called; otherwise an internal
    dendropy implementation is used (birth_death_tree / yule). Idempotent:
    existing '{stem}.nwk' files are skipped. Returns the nwk paths.
    """
    if stems is not None and len(stems) != ntrees:
        raise ValueError("stems (if given) must have length == ntrees")
    stems = list(stems) if stems is not None else [f"t{ntips:03d}_{i:06d}" for i in range(ntrees)]
    os.makedirs(out_dir, exist_ok=True)
    _guard_not_drive(out_dir, "tree output")

    phyloformer_dir = _env("PHYLOFORMER_DIR")
    vendored = VENDOR_DIR / VENDOR_TREE_SCRIPT
    if phyloformer_dir and tree_model in ("birth-death", "uniform") and vendored.exists():
        _vendor_phyloformer(phyloformer_dir)
        vendored_outputs = [
            Path(out_dir) / f"{i}_{ntips}_tips.nwk" for i in range(ntrees)
        ]
        missing = [p for p in vendored_outputs if not p.exists()]
        if missing:
            subprocess.run(
                [
                    "python",
                    str(VENDOR_DIR / VENDOR_TREE_SCRIPT),
                    "-n", str(ntrees),
                    "-t", str(ntips),
                    "--type", "birth-death" if tree_model == "birth-death" else "uniform",
                    "-o", out_dir,
                ],
                check=True,
            )
        print(f"[simulation] trees: {len(vendored_outputs)} present in {out_dir} (vendored)")
        return vendored_outputs

    if tree_model not in ("birth-death", "yule"):
        raise ValueError(f"unknown tree_model '{tree_model}' (birth-death | yule)")

    done = 0
    for i, stem in enumerate(stems):
        out_path = Path(out_dir) / f"{stem}.nwk"
        _log(stem, i, len(stems))
        if out_path.exists():
            done += 1
            continue
        rng = np.random.default_rng(seed + _hash_int(seed, f"tree:{stem}"))
        random.seed(seed + _hash_int(seed, f"tree:{stem}"))
        tree: dendropy.Tree | None = None
        for _attempt in range(10):  # birth-death can occasionally fail
            try:
                tree = treesim.birth_death_tree(
                    birth_rate=1.0,
                    death_rate=0.0 if tree_model == "yule" else 0.5,
                    num_extant_tips=ntips,
                )
                if len(tree.leaf_nodes()) == ntips:
                    break
            except Exception:  # noqa: BLE001 - retryable draw
                tree = None
                random.seed(seed + _hash_int(seed, f"tree:{stem}") + _attempt)
        if tree is None or len(tree.leaf_nodes()) != ntips:
            print(f"[simulation][warn] failed to simulate tree for {stem}; skipping")
            continue
        _rescale_diameter(tree, _target_diameter(rng))
        with open(out_path, "w") as fh:
            fh.write(tree.as_string(schema="newick", suppress_rooting=True).strip() + "\n")
        done += 1
    print(f"[simulation] trees: {done}/{len(stems)} present in {out_dir}")
    return [Path(out_dir) / f"{stem}.nwk" for stem in stems]


# --------------------------------------------------------------------------- #
# alignment simulation
# --------------------------------------------------------------------------- #
def _mutate(seq: np.ndarray, branch_len: float, rates: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply a uniform 20-state substitution process along one branch."""
    if branch_len <= 0.0:
        return seq
    p = 1.0 - np.exp(-branch_len * rates)
    change = rng.random(seq.shape[0]) < p
    n_ch = int(change.sum())
    if n_ch == 0:
        return seq
    out = seq.copy()
    out[change] = rng.integers(0, len(AMINO_ACIDS), size=n_ch)
    return out


def _simulate_alignment_python(
    newick: str, length: int, seed: int, indels: bool
) -> tuple[list[str], list[str]]:
    """Dev/smoke fallback: deterministic uniform 20-state substitution sampler.

    Not a substitute for AliSim quality (uniform exchangeabilities and
    frequencies); used only when no iqtree binary is available.
    """
    rng = np.random.default_rng(seed)
    tree = dendropy.Tree.get(data=newick, schema="newick")
    leaves = tree.leaf_nodes()
    rates = rng.gamma(1.0, 1.0, size=length)
    root_seq = rng.integers(0, len(AMINO_ACIDS), size=length)
    seqs: dict[dendropy.Node, np.ndarray] = {}
    for node in tree.preorder_node_iter():
        if node is tree.seed_node:
            parent_seq = root_seq
            branch_len = 0.0
        else:
            parent_seq = seqs[node.parent_node]
            branch_len = node.edge_length or 0.0
        seqs[node] = _mutate(parent_seq, branch_len, rates, rng)
    aligned = [seqs[leaf] for leaf in leaves]
    names = [
        leaf.taxon.label if leaf.taxon and leaf.taxon.label else f"s{i + 1}"
        for i, leaf in enumerate(leaves)
    ]
    if indels:
        n_del = int(length * (0.05 + 0.05 * rng.random()))
        del_cols = rng.choice(length, size=n_del, replace=False)
        col_mask = np.isin(np.arange(length), del_cols)
        for i in range(len(aligned)):
            aligned[i] = np.where(col_mask, -1, aligned[i])
    aa = np.array(list(AMINO_ACIDS))
    aligned_str = ["".join(aa[s] if s >= 0 else "-" for s in row) for row in aligned]
    return names, aligned_str


def _simulate_alignment_alisim(
    stem: str,
    nwk_path: Path,
    out_dir: str,
    aligned_dir: str,
    subst_model: str,
    gamma: str,
    length: int,
    indels: bool,
    iqtree: str,
    seed: int,
) -> None:
    """AliSim (iqtree --alisim) in a scratch tmp dir; writes aligned + stripped."""
    tmpd = tempfile.mkdtemp(prefix=f"alisim_{stem}_", dir=out_dir)
    try:
        cmd = [
            iqtree,
            "--alisim", os.path.join(tmpd, stem),
            "-t", str(nwk_path),
            "-m", f"{subst_model}+{gamma}",
            "-af", "fasta",
            "--seqtype", "AA",
            "--length", str(length),
            "--threads", "1",
            "-seed", str(seed),
        ]
        if indels:
            cmd += ["--indel", "0.01,0.01", "--indel-size", "GEO{5},GEO{4}"]
        proc = subprocess.run(  # noqa: PLW1510 - returncode handled below
            cmd, capture_output=True, text=True, timeout=1800
        )
        if proc.returncode != 0:
            raise RuntimeError(f"AliSim failed for {stem}: {proc.stderr[-500:]}")
        fa_path = os.path.join(tmpd, f"{stem}.fa")
        if not os.path.exists(fa_path):
            raise RuntimeError(f"AliSim produced no output for {stem}")
        names, seqs = _read_fasta(fa_path)
        if indels:
            seqs = [s[:length] for s in seqs]
        os.makedirs(aligned_dir, exist_ok=True)
        _write_fasta(os.path.join(aligned_dir, f"{stem}.fasta"), seqs, names)
        _write_fasta(os.path.join(out_dir, f"{stem}.fasta"), [_strip_gaps(s) for s in seqs], names)
    finally:
        shutil.rmtree(tmpd, ignore_errors=True)


def simulate_alignments(
    tree_dir: str,
    out_dir: str,
    subst_model: str = "LG",
    gamma: str = "GC",
    length: int = 300,
    indels: bool = False,
    iqtree_bin: str | None = None,
    seed: int = 42,
    allow_duplicates: bool = False,
    max_attempts: int = 3,
    engine: str = "auto",
    stems: Sequence[str] | None = None,
) -> list[Path]:
    """Simulate one UNALIGNED FASTA per tree (gap-stripped) + aligned copy.

    engine "auto": iqtree --alisim when a binary is found ($IQTREE_BIN or
    shutil.which("iqtree")), else the pure-python engine. Writes ONLY to
    local scratch — raises ValueError if tree_dir/out_dir is inside
    $DATA_DIR. Idempotent: existing '{stem}.fasta' outputs are skipped.
    Returns the unaligned FASTA paths.
    """
    _guard_not_drive(tree_dir, "tree input")
    _guard_not_drive(out_dir, "alignment output")
    os.makedirs(out_dir, exist_ok=True)
    aligned_dir = os.path.join(out_dir, "aligned")

    iqtree = _find_iqtree(iqtree_bin)
    if engine == "auto":
        engine = "alisim" if iqtree else "python"
    engine = engine.lower()
    if engine == "alisim" and not iqtree:
        raise ValueError("engine='alisim' requires an iqtree binary ($IQTREE_BIN or PATH)")

    if stems is None:
        stems = sorted(
            {
                Path(e.path).stem
                for e in os.scandir(tree_dir)
                if e.is_file() and Path(e.name).suffix in _TREE_EXTS
            }
        )

    out_paths: list[Path] = []
    done = skipped = 0
    for i, stem in enumerate(stems):
        nwk_path = Path(tree_dir) / f"{stem}.nwk"
        if not nwk_path.exists():
            for ext in _TREE_EXTS:
                if (Path(tree_dir) / f"{stem}{ext}").exists():
                    nwk_path = Path(tree_dir) / f"{stem}{ext}"
                    break
            else:
                continue
        out_path = Path(out_dir) / f"{stem}.fasta"
        _log(stem, i, len(stems))
        if out_path.exists():
            skipped += 1
            out_paths.append(out_path)
            continue
        ok = False
        for attempt in range(1, max_attempts + 1):
            try:
                if engine == "alisim":
                    assert iqtree is not None  # guaranteed by the raise above
                    _simulate_alignment_alisim(
                        stem, nwk_path, out_dir, aligned_dir,
                        subst_model, gamma, length, indels, iqtree,
                        seed + _hash_int(seed, f"alisim:{stem}:{attempt}"),
                    )
                else:
                    with open(nwk_path) as fh:
                        newick = fh.read()
                    names, aligned = _simulate_alignment_python(
                        newick, length,
                        seed + _hash_int(seed, f"alisim:{stem}:{attempt}"),
                        indels,
                    )
                    os.makedirs(aligned_dir, exist_ok=True)
                    _write_fasta(os.path.join(aligned_dir, f"{stem}.fasta"), aligned, names)
                    unaligned = [_strip_gaps(s) for s in aligned]
                    _write_fasta(str(out_path), unaligned, names)
                unaligned, _ = _read_fasta(str(out_path))
                if not unaligned or any(not s for s in unaligned):
                    raise RuntimeError("empty sequence after gap-stripping")
                if not allow_duplicates and _has_duplicates(unaligned) and attempt < max_attempts:
                    os.remove(out_path)
                    continue
                ok = True
                break
            except Exception as exc:  # noqa: BLE001 - resumable pipeline
                print(f"[simulation][warn] attempt {attempt}/{max_attempts} failed for {stem}: {exc}")
                if out_path.exists():
                    os.remove(out_path)
        if ok:
            done += 1
            out_paths.append(out_path)
        else:
            print(f"[simulation][warn] giving up on {stem} after {max_attempts} attempts")
    print(f"[simulation] alignments: {done} simulated, {skipped} already present ({engine} engine)")
    return out_paths


# --------------------------------------------------------------------------- #
# consolidation -> parquet
# --------------------------------------------------------------------------- #
def _approx_n_tips(stem: str, nwk_path: Path) -> int:
    m = re.match(r"^t(\d+)_", stem)
    if m:
        return int(m.group(1))
    try:
        text = nwk_path.read_text()
        if ":" in text:
            return text.count(":")
        return text.count(",") + 1
    except OSError:
        return 0


def _iter_pairs(
    raw_dir: str, want_stems: set[str] | None = None
) -> list[tuple[str, Path, Path]]:
    """Return (stem, nwk_path, fasta_path) sorted by n_tips then stem."""
    by_stem: dict[str, dict[str, Path]] = {}
    with os.scandir(raw_dir) as it:
        for e in it:
            if not e.is_file():
                continue
            p = Path(e.path)
            if p.suffix in _TREE_EXTS:
                by_stem.setdefault(p.stem, {})["nwk"] = p
            elif p.suffix in _ALN_EXTS:
                by_stem.setdefault(p.stem, {})["aln"] = p
    items = []
    for stem, files in by_stem.items():
        if "nwk" in files and "aln" in files and (want_stems is None or stem in want_stems):
            items.append((stem, files["nwk"], files["aln"]))
    return sorted(items, key=lambda it: (_approx_n_tips(it[0], it[1]), it[0]))


def _make_row(stem: str, nwk_path: Path, aln_path: Path) -> dict | None:
    """One parquet row: seqs, tree_newick, n_tips, scale (median root-to-tip)."""
    try:
        newick = nwk_path.read_text().strip()
        tree = dendropy.Tree.get(data=newick, schema="newick")
        n_tips = len(tree.leaf_nodes())
        _, seqs = _read_fasta(str(aln_path))
        seqs = [s.upper() for s in seqs]
        if not seqs or len(seqs) != n_tips:
            print(f"[simulation][warn] {stem}: {len(seqs)} seqs != {n_tips} tips; skipping row")
            return None
        if any(not s for s in seqs):
            print(f"[simulation][warn] {stem}: empty sequence after stripping; skipping row")
            return None
        root_dists = [node.distance_from_root() for node in tree.leaf_nodes()]
        scale = float(np.median(root_dists))
        if not scale > 0.0:
            scale = 1.0
        return {"seqs": seqs, "tree_newick": newick, "n_tips": n_tips, "scale": scale}
    except Exception as exc:  # noqa: BLE001 - resumable pipeline
        print(f"[simulation][warn] {stem}: skipping row ({exc})")
        return None


def _open_writer(path: str) -> pq.ParquetWriter:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    return pq.ParquetWriter(path + ".part", SCHEMA, compression="snappy")


def _flush_writer(writer: pq.ParquetWriter, pending: list[dict], path: str) -> None:
    if not pending:
        return
    df = pd.DataFrame(pending)
    table = pa.Table.from_pandas(df, schema=SCHEMA, preserve_index=False)
    writer.write_table(table)
    del table
    del df  # free the DataFrame between chunks
    pending.clear()


def consolidate_to_parquet(
    raw_dir: str,
    out_parquet: str,
    seed: int = 42,
    chunk_size: int = 5000,
) -> str:
    """Consolidate a raw scratch dir into ONE sorted (by n_tips) parquet file.

    Columns: seqs (list[str]), tree_newick (str), n_tips (int),
    scale (float, median root-to-tip). Chunked and snappy-compressed.
    Idempotent: existing out_parquet is skipped.
    """
    _ = seed  # kept for API symmetry / future deterministic ordering
    if os.path.exists(out_parquet):
        print(f"[simulation] {out_parquet} exists; skipping consolidation")
        return out_parquet
    pairs = _iter_pairs(raw_dir)
    writer = _open_writer(out_parquet)
    pending: list[dict] = []
    count = 0
    total = len(pairs)
    for idx, (stem, nwk_path, aln_path) in enumerate(pairs):
        _log(stem, idx, total)
        row = _make_row(stem, nwk_path, aln_path)
        if row is None:
            continue
        pending.append(row)
        count += 1
        if len(pending) >= chunk_size:
            _flush_writer(writer, pending, out_parquet)
    _flush_writer(writer, pending, out_parquet)
    writer.close()
    os.replace(out_parquet + ".part", out_parquet)
    print(f"[simulation] {count} rows -> {out_parquet}")
    return out_parquet


def _split_route(seed: int, train: float, val: float) -> Callable[[str], str]:
    def route(stem: str) -> str:
        f = _hash_frac(seed, f"split:{stem}")
        if f < train:
            return "train"
        if f < train + val:
            return "val"
        return "test"
    return route


def make_splits(
    raw_dir: str,
    data_dir: str,
    train: float = 0.8,
    val: float = 0.1,
    test: float = 0.1,
    seed: int = 42,
    local_tmp_dir: str | None = None,
) -> dict[str, Path]:
    """Consolidate raw scratch into train/val/test parquet in data_dir.

    Splits are assigned per-tree-stem via a seeded hash (no tree appears in
    two splits). Parquet is written to LOCAL scratch first, then copied into
    data_dir ATOMICALLY (tmp file + os.replace), so a killed session never
    leaves a half-written parquet on Drive. Returns {split: parquet path}.
    """
    total = train + val + test
    if not (0.0 < train < 1.0 and 0.0 < val < 1.0 and 0.0 < test < 1.0) or abs(total - 1.0) > 1e-9:
        raise ValueError(f"train/val/test must be in (0,1) and sum to 1, got {train}/{val}/{test}")
    os.makedirs(data_dir, exist_ok=True)
    route = _split_route(seed, train, val)
    all_pairs = _iter_pairs(raw_dir)
    out: dict[str, Path] = {}
    for split in ("train", "val", "test"):
        drive_path = Path(data_dir) / f"{split}.parquet"
        if drive_path.exists():
            print(f"[simulation] {split}: parquet already on Drive; skipping")
            out[split] = drive_path
            continue
        tmp_dir = local_tmp_dir or os.path.join(
            _env("LOCAL_DATA_DIR") or tempfile.gettempdir(), "parquet_tmp"
        )
        os.makedirs(tmp_dir, exist_ok=True)
        local_path = os.path.join(tmp_dir, f"{split}.parquet")
        pairs = [(s, n, a) for (s, n, a) in all_pairs if route(s) == split]
        writer = _open_writer(local_path)
        pending: list[dict] = []
        count = 0
        for idx, (stem, nwk_path, aln_path) in enumerate(pairs):
            _log(stem, idx, len(pairs))
            row = _make_row(stem, nwk_path, aln_path)
            if row is None:
                continue
            pending.append(row)
            count += 1
            if len(pending) >= 5000:
                _flush_writer(writer, pending, local_path)
        _flush_writer(writer, pending, local_path)
        writer.close()
        os.replace(local_path + ".part", local_path)
        _atomic_copy(local_path, str(drive_path))
        print(f"[simulation] {split}: {count} rows -> {drive_path}")
        out[split] = drive_path
    return out


def make_ood_set(
    raw_dir: str,
    data_dir: str,
    n_tips_range: tuple[int, int] = (100, 300),
    length_range: tuple[int, int] = (1000, 2000),
    indel: bool = True,
    subst_model: str = "WAG",
    n_trees: int = 20,
    seed: int = 42,
) -> Path:
    """Generate an out-of-distribution set and consolidate to ood.parquet.

    Raw OOD data lives in its own scratch dir (never Drive); the final
    ood.parquet is copied to data_dir atomically. Resumable.
    """
    _guard_not_drive(raw_dir, "OOD raw")
    os.makedirs(raw_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    tips_min, tips_max = n_tips_range
    len_min, len_max = length_range
    for k in range(n_trees):
        stem = f"t000_{k:06d}_ood"
        if os.path.exists(os.path.join(raw_dir, f"{stem}.fasta")):
            continue
        tips = int(rng.integers(tips_min, tips_max + 1))
        length = int(rng.integers(len_min, len_max + 1))
        seed_k = seed + k
        simulate_trees(tips, 1, raw_dir, seed=seed_k, stems=[stem])
        simulate_alignments(
            raw_dir, raw_dir,
            subst_model=subst_model, length=length, indels=indel,
            seed=seed_k, stems=[stem],
        )
        if k % LOG_EVERY == 0:
            print(f"[simulation][ood] {k}/{n_trees} pairs", flush=True)
    tmp_dir = os.path.join(_env("LOCAL_DATA_DIR") or tempfile.gettempdir(), "parquet_tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    local = os.path.join(tmp_dir, "ood.parquet")
    if not os.path.exists(local):
        pairs = _iter_pairs(raw_dir)
        writer = _open_writer(local)
        pending: list[dict] = []
        count = 0
        for idx, (stem, nwk_path, aln_path) in enumerate(pairs):
            _log(stem, idx, len(pairs))
            row = _make_row(stem, nwk_path, aln_path)
            if row is None:
                continue
            pending.append(row)
            count += 1
            if len(pending) >= 5000:
                _flush_writer(writer, pending, local)
        _flush_writer(writer, pending, local)
        writer.close()
        os.replace(local + ".part", local)
        print(f"[simulation] ood: {count} rows -> {local}")
    drive = Path(data_dir) / "ood.parquet"
    if not drive.exists():
        os.makedirs(data_dir, exist_ok=True)
        _atomic_copy(local, str(drive))
        print(f"[simulation] ood: atomic copy -> {drive}")
    return drive


def clean_scratch(scratch_dir: str) -> None:
    """Delete a scratch dir tree; refuses anything touching $DATA_DIR/$CKPT_DIR."""
    target = os.path.abspath(os.path.expanduser(scratch_dir))
    for var in ("DATA_DIR", "CKPT_DIR"):
        val = _env(var)
        if not val:
            continue
        guarded = os.path.abspath(os.path.expanduser(val))
        if target == guarded or target.startswith(guarded + os.sep) or guarded.startswith(target + os.sep):
            raise ValueError(
                f"refusing to delete {target}: it equals or contains ${var} ({guarded})"
            )
    if os.path.isdir(target):
        shutil.rmtree(target)
        print(f"[simulation] removed {target}")
    else:
        print(f"[simulation] nothing to clean at {target}")


# --------------------------------------------------------------------------- #
# orchestration CLI (thin; used by scripts/simulate_big.sh and docs)
# --------------------------------------------------------------------------- #
def cmd_trees(args: argparse.Namespace) -> int:
    simulate_trees(args.tips, args.count, args.out, seed=args.seed, tree_model=args.model)
    return 0


def cmd_alisim(args: argparse.Namespace) -> int:
    simulate_alignments(
        args.tree_dir, args.out,
        subst_model=args.model, gamma=args.gamma, length=args.length,
        indels=args.indels, iqtree_bin=args.iqtree, seed=args.seed,
        allow_duplicates=args.allow_duplicates, max_attempts=args.max_attempts,
        engine=args.engine,
    )
    return 0


def cmd_consolidate(args: argparse.Namespace) -> int:
    consolidate_to_parquet(args.raw, args.out, seed=args.seed)
    return 0


def cmd_splits(args: argparse.Namespace) -> int:
    make_splits(args.raw, args.data_dir, train=args.train, val=args.val,
                test=args.test, seed=args.seed)
    return 0


def cmd_ood(args: argparse.Namespace) -> int:
    make_ood_set(
        args.raw, args.data_dir,
        n_tips_range=(args.n_tips_min, args.n_tips_max),
        length_range=(args.len_min, args.len_max),
        indel=args.indels, subst_model=args.model, n_trees=args.n_trees,
        seed=args.seed,
    )
    return 0


def cmd_clean(args: argparse.Namespace) -> int:
    clean_scratch(args.raw)
    return 0


# --------------------------------------------------------------------------- #
# big grid (scripts/simulate_big.sh) + shared parallel helpers
# --------------------------------------------------------------------------- #
TIP_COUNTS = [10, 20, 40, 80, 160, 320, 640, 1280]
TREES_PER_BIN = 200
LENGTHS = [150, 300, 600]
REPLICATES = 21


def _chunk(seq: Sequence, n: int) -> list[list]:
    """Split seq into n roughly equal disjoint chunks (non-empty)."""
    k, m = divmod(len(seq), n)
    out: list[list] = []
    i = 0
    for j in range(n):
        size = k + (1 if j < m else 0)
        out.append(list(seq[i:i + size]))
        i += size
    return [c for c in out if c]


def _sim_alignment_task(task: tuple) -> int:
    """Worker entry for ProcessPoolExecutor: simulate one stem subset.

    Task = (tree_dir, out_dir, length, seed, engine, indels, stems).
    simulate_alignments is per-stem idempotent (skips existing .fasta) and
    uses per-stem tmp dirs, so parallel calls over DISJOINT stem subsets are
    safe. Seeded deterministically per stem, so results are reproducible
    regardless of worker/partition layout.
    """
    tree_dir, out_dir, length, seed, engine, indels, stems = task
    simulate_alignments(
        tree_dir, out_dir, length=length, seed=seed, engine=engine,
        indels=indels, stems=stems,
    )
    return len(stems)


def _run_alignment_phase(raw: str, stems: list[str], length: int,
                         seed: int, engine: str, indels: bool, workers: int) -> None:
    """simulate_alignments over `stems`, split into disjoint worker chunks."""
    if workers > 1 and len(stems) > 1:
        tasks = [(raw, raw, length, seed, engine, indels, ch)
                 for ch in _chunk(stems, workers)]
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
            for _ in ex.map(_sim_alignment_task, tasks):
                pass
    else:
        simulate_alignments(raw, raw, length=length, seed=seed,
                            engine=engine, indels=indels, stems=stems)


def _print_split_rows(data_dir: str) -> None:
    print("[big] split row counts (real):")
    for name in ("train.parquet", "val.parquet", "test.parquet"):
        path = os.path.join(data_dir, name)
        if os.path.exists(path):
            print(f"  {name}: {pq.read_table(path).num_rows} rows")


def cmd_big(args: argparse.Namespace) -> int:
    """The full simulation grid: 8 tip counts x 200 trees/bin x 3 lengths x 21
    replicates = 100,800 (tree, alignment) pairs, written to LOCAL scratch
    ONLY, then consolidated + split onto Drive atomically (see
    scripts/simulate_big.sh)."""
    local = os.path.abspath(args.local_data_dir or _env("LOCAL_DATA_DIR") or "/content/data")
    data_dir = args.data_dir or _env("DATA_DIR")
    if not data_dir:
        raise SystemExit("no data dir: set DATA_DIR or pass --data-dir")
    raw = os.path.join(local, "raw")
    os.makedirs(raw, exist_ok=True)

    tip_counts = [int(x) for x in str(args.tip_counts).split(",") if x.strip()]
    lengths = [int(x) for x in str(args.lengths).split(",") if x.strip()]
    if not tip_counts or not lengths or args.trees_per_bin < 1 or args.replicates < 1:
        raise SystemExit("bad grid: need >=1 tip count, >=1 length, trees-per-bin >= 1, replicates >= 1")
    bins: list[tuple[int, int, list[str]]] = []
    for tips in tip_counts:
        for length in lengths:
            for rep in range(args.replicates):
                stems = [f"t{tips:03d}_l{length}_r{rep:02d}_{i:06d}"
                         for i in range(args.trees_per_bin)]
                bins.append((tips, length, stems))
    total = len(bins) * args.trees_per_bin
    print(f"[big] grid: {len(tip_counts)} tip counts x {len(lengths)} lengths x "
          f"{args.replicates} replicates x {args.trees_per_bin} trees "
          f"= {total} (tree, alignment) pairs")
    if args.resume:
        print("[big] resume mode: already-present stems are skipped (idempotent)")

    for i, (tips, _length, stems) in enumerate(bins):
        print(f"[big] trees bin {i + 1}/{len(bins)}: tips={tips}")
        simulate_trees(tips, args.trees_per_bin, raw, seed=args.seed, stems=stems)

    for i, (_tips, length, stems) in enumerate(bins):
        print(f"[big] alignments bin {i + 1}/{len(bins)}: length={length} "
              f"({args.workers} worker(s))")
        _run_alignment_phase(raw, stems, length, args.seed, args.engine,
                             args.indels, args.workers)

    print("[big] consolidating + splitting -> Drive")
    make_splits(raw, data_dir, seed=args.seed)
    _print_split_rows(data_dir)
    return 0


def cmd_smoke(args: argparse.Namespace) -> int:
    """200 alignments end-to-end: trees -> alignments -> splits -> 'Drive'."""
    local = os.path.abspath(args.local_data_dir or _env("LOCAL_DATA_DIR") or "/content/data")
    data_dir = args.data_dir or _env("DATA_DIR")
    if not data_dir:
        raise SystemExit("no data dir: set DATA_DIR or pass --data-dir")
    raw = os.path.join(local, "raw")
    os.makedirs(raw, exist_ok=True)
    stems = [f"t020_{i:06d}" for i in range(200)]
    print(f"[smoke] simulating {len(stems)} trees (tips=20, length=150) into {raw}")
    simulate_trees(20, 200, raw, seed=args.seed, stems=stems)
    _run_alignment_phase(raw, stems, 150, args.seed, args.engine,
                         getattr(args, "indels", False), getattr(args, "workers", 1))
    print("[smoke] consolidating + splitting -> Drive")
    make_splits(raw, data_dir, seed=args.seed)
    _print_parquet_sizes(data_dir)
    return 0


def _print_parquet_sizes(data_dir: str) -> None:
    print("[smoke] parquet files:")
    for name in ("train.parquet", "val.parquet", "test.parquet", "ood.parquet"):
        path = os.path.join(data_dir, name)
        if os.path.exists(path):
            print(f"  {name}: {os.path.getsize(path) / 1e6:.1f} MB")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ssm_phylo.data.simulation",
        description="Simulate unaligned training data, consolidate to parquet, split to Drive.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("trees", help="simulate trees only (Newick)")
    t.add_argument("--tips", type=int, required=True)
    t.add_argument("--count", type=int, required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--model", default="birth-death", choices=["birth-death", "yule"])
    t.add_argument("--seed", type=int, default=42)
    t.set_defaults(func=cmd_trees)

    a = sub.add_parser("alisim", help="simulate alignments for existing trees")
    a.add_argument("--tree-dir", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--length", type=int, default=300)
    a.add_argument("--model", default="LG")
    a.add_argument("--gamma", default="GC")
    a.add_argument("--indels", action="store_true")
    a.add_argument("--iqtree", default=None)
    a.add_argument("--engine", default="auto", choices=["auto", "alisim", "python"])
    a.add_argument("--seed", type=int, default=42)
    a.add_argument("--allow-duplicates", action="store_true")
    a.add_argument("--max-attempts", type=int, default=3)
    a.set_defaults(func=cmd_alisim)

    c = sub.add_parser("consolidate", help="consolidate raw scratch -> one parquet")
    c.add_argument("--raw", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--seed", type=int, default=42)
    c.set_defaults(func=cmd_consolidate)

    s = sub.add_parser("splits", help="consolidate + split train/val/test onto Drive")
    s.add_argument("--raw", required=True)
    s.add_argument("--data-dir", default=None)
    s.add_argument("--train", type=float, default=0.8)
    s.add_argument("--val", type=float, default=0.1)
    s.add_argument("--test", type=float, default=0.1)
    s.add_argument("--seed", type=int, default=42)
    s.set_defaults(func=cmd_splits)

    o = sub.add_parser("ood", help="generate OOD set -> ood.parquet on Drive")
    o.add_argument("--raw", required=True)
    o.add_argument("--data-dir", default=None)
    o.add_argument("--n-trees", type=int, default=20)
    o.add_argument("--n-tips-min", type=int, default=100)
    o.add_argument("--n-tips-max", type=int, default=300)
    o.add_argument("--len-min", type=int, default=1000)
    o.add_argument("--len-max", type=int, default=2000)
    o.add_argument("--indels", action="store_true")
    o.add_argument("--model", default="WAG")
    o.add_argument("--seed", type=int, default=42)
    o.set_defaults(func=cmd_ood)

    cl = sub.add_parser("clean", help="delete a raw scratch dir (LOCAL only)")
    cl.add_argument("--raw", required=True)
    cl.set_defaults(func=cmd_clean)

    sm = sub.add_parser("smoke", help="200-alignment end-to-end smoke run")
    sm.add_argument("--data-dir", default=None)
    sm.add_argument("--local-data-dir", default=None)
    sm.add_argument("--workers", type=int, default=1,
                    help="parallelize the 200 alignments across N processes (default 1)")
    sm.add_argument("--engine", default="auto", choices=["auto", "alisim", "python"])
    sm.add_argument("--seed", type=int, default=42)
    sm.set_defaults(func=cmd_smoke)

    bg = sub.add_parser("big", help="full simulation grid (scripts/simulate_big.sh)")
    bg.add_argument("--data-dir", default=None)
    bg.add_argument("--local-data-dir", default=None)
    bg.add_argument("--workers", type=int, default=4)
    bg.add_argument("--resume", action="store_true",
                    help="skip stems already present in scratch (idempotent)")
    bg.add_argument("--engine", default="auto", choices=["auto", "alisim", "python"])
    bg.add_argument("--tip-counts", default=",".join(str(x) for x in TIP_COUNTS),
                    help="comma-separated tip counts (default 10,20,...,1280)")
    bg.add_argument("--trees-per-bin", type=int, default=TREES_PER_BIN)
    bg.add_argument("--lengths", default=",".join(str(x) for x in LENGTHS),
                    help="comma-separated alignment lengths (default 150,300,600)")
    bg.add_argument("--replicates", type=int, default=REPLICATES)
    bg.add_argument("--indels", action="store_true")
    bg.add_argument("--seed", type=int, default=42)
    bg.set_defaults(func=cmd_big)

    return p


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--smoke" in argv:
        argv = ["smoke"] + [a for a in argv if a != "--smoke"]
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

"""Dataloaders for parquet-consolidated simulated data.

ParquetPhyloDataset memory-maps $DATA_DIR/{split}.parquet (pyarrow
read_table with memory_map=True, pre_buffer=False for Google Drive FUSE
friendliness) and yields per-sample:
    (tokens, seq_spans, true_distances, scale, tree_newick)
where tokens is the int-encoded concatenation of all sequences, each
PREFIXED by <cls> (ProtMamba convention), separated by a dedicated <sep>
token, with NO trailing sep. seq_spans lists the (start, end) token spans
per sequence (covering the stream exactly, one sep between consecutive
spans). true_distances is the n x n patristic distance matrix normalized by
scale (median root-to-tip).

collate_with_bucketing returns (tokens, seq_spans, spans_mask, dm, scales)
where spans_mask is a PER-SPAN validity mask of shape (B, N) — True for
real sequences, False for padded spans (NOT a per-token mask).
"""
from __future__ import annotations

import os
from collections.abc import Sequence
from functools import lru_cache

import dendropy
import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

# 38-token vocab: 4 control tokens + 20 amino acids + 14 specials. Ids only
# need to be self-consistent (and match what the model's embedding expects).
_SPECIALS = [
    "<pad>", "<sep>", "<cls>", "<unk>",
]
_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
_FILL_SPECIALS = [
    "<mask-1>", "<mask-2>", "<mask-3>", "<mask-4>", "<mask-5>",
    "<eop>", "<empty>", "<end>", "<gap>", "<start>", "<stop>",
    "<mask-6>", "<mask-7>", "<mask-8>",
]
_VOCAB = _SPECIALS + list(_AMINO_ACIDS) + _FILL_SPECIALS  # 4 + 20 + 14 = 38


class PhyloTokenizer:
    """Int encoder over a 38-token vocab; no separator/cls added by encode()."""

    def __init__(self, vocab: Sequence[str] = _VOCAB) -> None:
        self.vocab = list(vocab)
        self.vocab_size = len(self.vocab)
        self.aa_to_id = {tok: i for i, tok in enumerate(self.vocab)}
        self.id_to_aa = {i: tok for tok, i in self.aa_to_id.items()}
        self.pad_id = self.aa_to_id["<pad>"]
        self.sep_id = self.aa_to_id["<sep>"]
        self.cls_id = self.aa_to_id["<cls>"]
        self.unk_id = self.aa_to_id["<unk>"]

    def encode(self, sequence: str) -> list[int]:
        out = []
        lookup = self.aa_to_id
        unk = self.unk_id
        for ch in sequence.upper():
            out.append(lookup.get(ch, unk))
        return out


_DEFAULT_TOKENIZER: PhyloTokenizer | None = None


def get_tokenizer() -> PhyloTokenizer:
    """Return the shared 38-token PhyloTokenizer (pad=0, sep=1, cls=2, unk=3)."""
    global _DEFAULT_TOKENIZER
    if _DEFAULT_TOKENIZER is None:
        _DEFAULT_TOKENIZER = PhyloTokenizer()
    return _DEFAULT_TOKENIZER


@lru_cache(maxsize=64)
def _patristic_matrix(newick: str) -> np.ndarray:
    """n x n patristic distance matrix (float32) for a Newick tree."""
    tree = dendropy.Tree.get(data=newick, schema="newick")
    leaves = tree.leaf_nodes()
    n = len(leaves)
    ndm = tree.node_distance_matrix()
    m = np.empty((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            m[i, j] = ndm(leaves[i], leaves[j])
    return m


def _tokenize_with_spans(
    seqs: Sequence[str], tokenizer: PhyloTokenizer, max_seq_len: int
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Concatenate [<cls>, seq] blocks with <sep> between; per-seq truncation.

    If the stream exceeds max_seq_len, every sequence is truncated
    proportionally (keeping ALL sequences — a taxon is never dropped). Spans
    cover the stream exactly: [start, end) includes the sequence's <cls>.
    """
    n = len(seqs)
    if n == 0:
        return np.zeros(0, dtype=np.int64), []
    cls_id = tokenizer.cls_id
    sep_id = tokenizer.sep_id
    # per-seq char budget: n cls tokens + (n-1) sep tokens + n*len <= max_seq_len
    budget = max(1, (max_seq_len - n - (n - 1)) // n)
    if budget < 1:
        raise ValueError(
            f"max_seq_len={max_seq_len} too small for {n} sequences "
            f"(need at least {2 * n - 1} tokens)"
        )
    tokens: list[int] = []
    spans: list[tuple[int, int]] = []
    for i, seq in enumerate(seqs):
        start = len(tokens)
        tokens.append(cls_id)
        tokens.extend(tokenizer.encode(seq[:budget]))
        end = len(tokens)
        spans.append((start, end))
        if i < n - 1:
            tokens.append(sep_id)
    return np.asarray(tokens, dtype=np.int64), spans


def _fit_tokens(
    tokens: np.ndarray, spans: Sequence[tuple[int, int]], max_seq_len: int
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Defensive per-sequence truncation of an already-tokenized stream.

    Keeps every sequence (no dropping); each span keeps its <cls> and is
    truncated to a per-seq budget; separators are kept.
    """
    total = len(tokens)
    n = len(spans)
    if total <= max_seq_len:
        return tokens, list(spans)
    budget = max(1, (max_seq_len - n - (n - 1)) // n)
    out: list[int] = []
    new_spans: list[tuple[int, int]] = []
    for i, (s, e) in enumerate(spans):
        cut = min(budget + 1, e - s)  # +1 keeps the <cls> prefix
        start = len(out)
        out.extend(tokens[s : s + cut])
        new_spans.append((start, start + cut))
        if i < n - 1:
            out.append(int(tokens[e]))  # separator right after span i
    return np.asarray(out, dtype=np.int64), new_spans


class ParquetPhyloDataset(Dataset):
    """Torch dataset over one consolidated parquet split (memory-mapped)."""

    def __init__(
        self,
        parquet_path: str,
        max_seq_len: int,
        num_samples: int | None = None,
        seed: int | None = None,
    ) -> None:
        self.parquet_path = parquet_path
        self.max_seq_len = max_seq_len
        self.tokenizer = get_tokenizer()

        self.table = pq.read_table(
            parquet_path,
            columns=["seqs", "tree_newick", "n_tips", "scale"],
            memory_map=True,
            pre_buffer=False,  # FUSE-friendly: no aggressive readahead
        )
        n_total = len(self.table)
        valid = list(range(n_total))
        if num_samples is not None and num_samples < n_total:
            rng = np.random.default_rng(seed if seed is not None else 42)
            valid = sorted(rng.choice(valid, size=num_samples, replace=False).tolist())
        self.index = valid
        print(
            f"[ssm_phylo.data] ParquetPhyloDataset({parquet_path}): "
            f"{len(self.index)}/{n_total} rows"
        )

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        row = self.index[idx]
        seqs = self.table["seqs"][row].as_py()
        newick = self.table["tree_newick"][row].as_py()
        scale = float(self.table["scale"][row].as_py())
        if not scale > 0.0:
            scale = 1.0

        tokens, spans = _tokenize_with_spans(seqs, self.tokenizer, self.max_seq_len)
        dist_matrix = _patristic_matrix(newick) / scale
        return tokens, spans, dist_matrix, scale, newick


def collate_with_bucketing(
    batch: Sequence[tuple],
    max_seq_len: int,
    pad_id: int,
    bucket_step: int = 512,
):
    """Collate samples into bucketed padded tensors.

    Returns (tokens, seq_spans, spans_mask, dm, scales):
    - tokens: (B, L) long, L <= max_seq_len, padded to a bucket multiple of
      bucket_step.
    - seq_spans: (B, N, 2) long, N = max seqs in the batch; padded spans are
      start=end=0.
    - spans_mask: (B, N) bool — PER-SPAN validity (True = real sequence).
    - dm: (B, N, N) float32, padded with -1.0 for missing taxa.
    - scales: (B,) float32.
    """
    if not batch:
        raise ValueError("collate_with_bucketing: empty batch")
    padded: list[tuple] = []
    for sample in batch:
        tokens, spans, dm, scale, _newick = sample
        tokens, spans = _fit_tokens(tokens, spans, max_seq_len)
        key = -(-len(tokens) // bucket_step) * bucket_step
        padded.append((tokens, spans, dm, scale, key))

    B = len(padded)
    target = max(k for *_ignored, k in padded)
    n_max = max(len(spans) for _, spans, *_ in padded)
    tok_t = torch.full((B, target), pad_id, dtype=torch.long)
    spans_t = torch.zeros(B, n_max, 2, dtype=torch.long)  # padded spans: (0, 0)
    spans_mask = torch.zeros(B, n_max, dtype=torch.bool)
    dm_t = torch.full((B, n_max, n_max), -1.0, dtype=torch.float32)
    scales_t = torch.zeros(B, dtype=torch.float32)

    for b, (tokens, spans, dm, scale, _key) in enumerate(padded):
        n_tok = len(tokens)
        tok_t[b, :n_tok] = torch.from_numpy(tokens)
        for r, (s, e) in enumerate(spans):
            spans_t[b, r, 0] = s
            spans_t[b, r, 1] = e
            spans_mask[b, r] = True
        n_taxa = dm.shape[0]
        dm_t[b, :n_taxa, :n_taxa] = torch.from_numpy(np.asarray(dm, dtype=np.float32))
        scales_t[b] = scale
    return tok_t, spans_t, spans_mask, dm_t, scales_t


def load_dataset(
    split: str,
    data_dir: str | None = None,
    max_seq_len: int = 32768,
    num_samples: int | None = None,
    seed: int = 42,
    tokenizer: PhyloTokenizer | None = None,
) -> ParquetPhyloDataset:
    """Load $data_dir/{split}.parquet (split is typically train|val|test|ood).

    Raises FileNotFoundError with a clear message when the file is missing.
    """
    if data_dir is None:
        data_dir = os.environ.get("DATA_DIR")
    if not data_dir:
        raise ValueError("load_dataset: DATA_DIR unset and no data_dir given")
    path = os.path.join(data_dir, f"{split}.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} does not exist — run the simulation pipeline first "
            f"(python -m ssm_phylo.data.simulation --smoke)"
        )
    ds = ParquetPhyloDataset(path, max_seq_len=max_seq_len, num_samples=num_samples, seed=seed)
    if tokenizer is not None:
        ds.tokenizer = tokenizer
    print(f"[ssm_phylo.data] loaded {split} split: {len(ds)} rows from {path}")
    return ds

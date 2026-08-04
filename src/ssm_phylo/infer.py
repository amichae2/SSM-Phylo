"""Inference: checkpoint -> predicted distance matrix (raw substitution units).

predict_distances tokenizes n unaligned sequences (cls-prefixed, sep-separated
— the same convention as ParquetPhyloDataset), runs PhyloModel, and returns
the predicted n x n distance matrix in RAW substitution units (the model
outputs NORMALIZED distances; the checkpoint's global 'scale' factor, stored
by the Phase 3 trainer, is multiplied back).

Pair scoring is BLOCKWISE: never more than (chunk, chunk) embedding pairs are
materialized at once, so large n stays memory-friendly (the (n, n) output
buffer itself is unavoidable).

CLI:
  python -m ssm_phylo.infer --checkpoint ckpt.pt --input seqs.fasta --output dm.npy
  (--input may also be a .parquet path; the first row's seqs column is used)
"""
from __future__ import annotations

import argparse
from collections.abc import Sequence

import numpy as np
import torch

from ssm_phylo.data.datasets import _tokenize_with_spans, get_tokenizer
from ssm_phylo.data.simulation import _read_fasta
from ssm_phylo.models.encoder import build_encoder
from ssm_phylo.models.head import PhyloModel


def _load_phylo_model(checkpoint_path: str, device: torch.device) -> tuple[PhyloModel, dict]:
    """Build PhyloModel from a training checkpoint (config + model_state)."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    encoder = build_encoder(cfg, device=str(device))
    model = PhyloModel(
        encoder,
        d_emb=int(cfg.get("d_emb", 256)),
        max_dist=float(cfg.get("max_dist", 3.0)),
        head_type=str(cfg.get("head", "bilinear_mlp")),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def predict_distances(
    model: PhyloModel,
    seqs: list[str],
    tokenizer,
    max_seq_len: int = 32768,
    batch_size: int = 1,
    device: torch.device | None = None,
    scale: float = 1.0,
    chunk: int = 64,
) -> np.ndarray:
    """Predict the (n, n) distance matrix for n unaligned sequences.

    Raw substitution units = normalized prediction * scale (the checkpoint's
    global scale factor; default 1.0). Pair scoring runs in (chunk, chunk)
    blocks. batch_size is accepted for API symmetry (v1 processes one sample).
    """
    del batch_size  # v1: single sample; kept for API symmetry
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokens, spans = _tokenize_with_spans(seqs, tokenizer, max_seq_len)
    n = len(spans)
    if n == 0:
        raise ValueError("no sequences to predict")
    tok_t = torch.from_numpy(tokens).unsqueeze(0).to(device)
    spans_t = torch.tensor(spans, dtype=torch.long, device=device).unsqueeze(0)
    mask = torch.ones(1, n, dtype=torch.bool, device=device)

    model.eval()
    with torch.no_grad():
        hidden = model.encoder(tok_t, None, None)
        embs = model.head.pool_all(hidden, spans_t, mask)[0]  # (n, d_emb)
        raw = torch.zeros(n, n, dtype=embs.dtype, device=device)
        for i0 in range(0, n, chunk):
            ei = embs[i0 : i0 + chunk].unsqueeze(0)
            ci = ei.shape[1]
            for j0 in range(i0, n, chunk):
                ej = embs[j0 : j0 + chunk].unsqueeze(0)
                cj = ej.shape[1]
                # ci >= cj always (i0 <= j0 => n-i0 >= n-j0), so the upper
                # triangle write is shape-safe; the explicit clamps + assert
                # make this invariant obvious and robust to future edits.
                i1 = min(i0 + ci, n)
                j1 = min(j0 + cj, n)
                upper = model.head._score_blocks(ei, ej)[0]   # (ci, cj)
                assert upper.shape == (ci, cj), upper.shape
                raw[i0:i1, j0:j1] = upper
                if i0 != j0:
                    # Head is NOT symmetric pre-postprocess (untied bilinear W,
                    # asymmetric MLP inputs): lower triangle needs the
                    # role-swapped score s_{j,i} from (ej, ei), NOT a transpose
                    # of s_{i,j}.
                    lower = model.head._score_blocks(ej, ei)[0]  # (cj, ci)
                    assert lower.shape == (cj, ci), lower.shape
                    raw[j0:j1, i0:i1] = lower
        dist = model.head.postprocess(raw)
        return (dist * scale).cpu().numpy().astype(np.float32)


def _read_input(path: str) -> list[str]:
    if path.endswith(".parquet"):
        import pyarrow.parquet as pq

        table = pq.read_table(path, columns=["seqs"])
        if len(table) == 0:
            raise ValueError(f"no rows in {path}")
        seqs = table["seqs"][0].as_py()
        print(f"[infer] using first row of {path} ({len(seqs)} sequences)")
        return seqs
    _headers, seqs = _read_fasta(path)
    return seqs


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ssm_phylo.infer",
        description="Predict a distance matrix from a checkpoint + unaligned sequences.",
    )
    parser.add_argument("--checkpoint", required=True, help="training checkpoint (.pt)")
    parser.add_argument("--input", required=True, help="seqs.fasta or seqs.parquet")
    parser.add_argument("--output", required=True, help="output .npy distance matrix")
    parser.add_argument("--device", default=None, help="cpu or cuda (default auto)")
    parser.add_argument("--max-seq-len", type=int, default=32768)
    args = parser.parse_args(argv)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, ckpt = _load_phylo_model(args.checkpoint, device)
    scale = float(ckpt.get("scale", 1.0))
    seqs = _read_input(args.input)
    tokenizer = get_tokenizer()
    print(f"[infer] {len(seqs)} sequences, scale={scale:.4f}, device={device}")
    dm = predict_distances(model, seqs, tokenizer, max_seq_len=args.max_seq_len,
                           device=device, scale=scale)
    np.save(args.output, dm)
    print(f"[infer] wrote {args.output} ({dm.shape}, symmetric={np.allclose(dm, dm.T, atol=1e-4)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

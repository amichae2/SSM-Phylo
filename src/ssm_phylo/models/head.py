"""The task head (the novel contribution): attention pooling + distance head.

Design (rationale at the bottom of this docstring):
- AttentionPooling: a learned query vector scores every token of a sequence's
  hidden states; softmax weights over valid tokens; the weighted mean is then
  projected to d_emb. Cheap (one learned vector + one linear), gradient flows
  to every token, no length bias (unlike mean pooling over padded spans).
- DistanceHead: bilinear core s_ij = e_i^T W e_j + b captures pairwise
  interactions at rank d_emb; the bilinear_mlp variant (default) refines each
  pair with an MLP over [e_i; e_j; e_i*e_j; |e_i-e_j|] (element-wise products
  = interaction terms, absolute difference = dissimilarity signal). Output is
  symmetrized by transpose-averaging, pushed through softplus (strictly
  positive), its diagonal zeroed, and clipped at max_dist (config default 3.0)
  — guaranteeing a symmetric, non-negative, bounded distance matrix. The pair
  MLP is computed in j-chunks to bound memory at large n (O(B*n*chunk*4d)).
- PhyloModel composes an ENCODER-AGNOSTIC encoder (any object with a
  forward(tokens, ...) returning (B, L, d_model) and a d_model property, e.g.
  MambaEncoder) with the DistanceHead. Distances are predicted in
  NORMALIZED space; the per-sample scale is applied by the loss (Phase 3).

SMOKE: `python -m ssm_phylo.models.head --smoke` builds a tiny from_scratch
encoder (CI-safe, no weights), predicts (2, 60, 60) distance matrices and
prints shape + symmetry check + parameter counts.
"""
from __future__ import annotations

import argparse
import math
import time
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


# --------------------------------------------------------------------------- #
# pooling
# --------------------------------------------------------------------------- #
class AttentionPooling(nn.Module):
    """Learned-query attention pooling over ONE sequence's tokens.

    Args:
        d_model: hidden size of the encoder.
        d_emb: projection dim for the pooled embedding (default 256).
    """

    def __init__(self, d_model: int, d_emb: int = 256) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_emb = d_emb
        self.query = nn.Parameter(torch.empty(d_model))
        nn.init.normal_(self.query, mean=0.0, std=d_model ** -0.5)
        self.proj = nn.Linear(d_model, d_emb)

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """hidden: (B, L, d_model); mask: (B, L) bool (valid tokens) -> (B, d_emb)."""
        scores = hidden @ self.query  # (B, L)
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = F.softmax(scores, dim=-1)  # (B, L), sum 1 over valid tokens
        # All-masked rows produce softmax(-inf) = NaN (0/0). Degenerate rows
        # must pool to ZERO — otherwise a NaN pooled value multiplies the
        # zero gradient of masked spans (NaN * 0 = NaN poisons the whole
        # backward pass in mixed-n batches). nan_to_num passes NaN positions
        # a zero gradient.
        weights = torch.nan_to_num(weights, nan=0.0)
        pooled = (weights.unsqueeze(-1) * hidden).sum(dim=1)  # (B, d_model)
        return self.proj(pooled)


# --------------------------------------------------------------------------- #
# distance head
# --------------------------------------------------------------------------- #
class DistanceHead(nn.Module):
    """Embed sequences -> n x n distance matrix with hard guarantees.

    head_type:
        "bilinear":      s_ij = e_i^T W e_j + b
        "mlp":           s_ij = MLP([e_i; e_j; e_i*e_j; |e_i-e_j|])
        "bilinear_mlp":  s_ij = bilinear + MLP refinement (DEFAULT)
    Output: symmetrized -> softplus -> diagonal zeroed -> clip at max_dist.
    """

    def __init__(
        self,
        d_model: int,
        d_emb: int = 256,
        max_dist: float = 3.0,
        head_type: str = "bilinear_mlp",
        mlp_hidden: int = 256,
        pair_chunk: int = 32,
    ) -> None:
        super().__init__()
        if head_type not in ("bilinear", "mlp", "bilinear_mlp"):
            raise ValueError(f"unknown head_type '{head_type}'")
        self.d_model = d_model
        self.d_emb = d_emb
        self.max_dist = float(max_dist)
        self.head_type = head_type
        self.pair_chunk = pair_chunk

        self.pooling = AttentionPooling(d_model, d_emb)
        self.bilinear_w = nn.Parameter(torch.empty(d_emb, d_emb))
        nn.init.normal_(self.bilinear_w, std=math.sqrt(2.0 / d_emb))
        self.bilinear_b = nn.Parameter(torch.zeros(()))
        self.mlp = nn.Sequential(
            nn.Linear(4 * d_emb, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, 1),
        )
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    # -- pooling ------------------------------------------------------------ #
    def pool_all(
        self,
        hidden_states: torch.Tensor,
        seq_spans: torch.Tensor,
        spans_mask: torch.Tensor,
    ) -> torch.Tensor:
        """hidden_states: (B, L, d_model); seq_spans: (B, N, 2) long;
        spans_mask: (B, N) bool -> (B, N, d_emb).

        Pools each sequence with AttentionPooling. One gather pass per
        sequence index k (padded across the batch to that k's max length), so
        memory stays O(B*L*d_model) overall. Invalid spans (mask=False or
        zero-length) keep a zero embedding, excluded by the caller's mask.
        """
        B, L, d = hidden_states.shape
        N = seq_spans.shape[1]
        embs = torch.zeros(B, N, self.d_emb, dtype=hidden_states.dtype, device=hidden_states.device)
        if N == 0 or L == 0:
            return embs
        for k in range(N):
            valid = spans_mask[:, k]
            if not valid.any():
                continue
            starts = seq_spans[:, k, 0]
            ends = seq_spans[:, k, 1]
            lengths = ends - starts
            l_max = int(lengths.max())
            if l_max <= 0:
                continue
            idx = torch.arange(l_max, device=hidden_states.device).unsqueeze(0) + starts.unsqueeze(1)
            idx = idx.clamp(max=L - 1).unsqueeze(-1).expand(-1, -1, d)
            hk = hidden_states.gather(1, idx)  # (B, l_max, d)
            tok_mask = torch.arange(l_max, device=hidden_states.device).unsqueeze(0) < lengths.unsqueeze(1)
            hk = hk.masked_fill(~tok_mask.unsqueeze(-1), 0.0)
            embs[:, k] = self.pooling(hk, tok_mask)
            embs[~valid, k] = 0.0  # masked rows must not leak pooled content
        return embs

    # -- pairwise scores ---------------------------------------------------- #
    def _score_blocks(self, ei: torch.Tensor, ej: torch.Tensor) -> torch.Tensor:
        """RAW pair scores for two embedding blocks: (B, ci, cj).

        ei: (B, ci, d_emb), ej: (B, cj, d_emb). Bilinear core (+ optional MLP
        refinement) WITHOUT symmetrization/softplus — callers (forward,
        blockwise inference) post-process via self.postprocess. This is the
        single source of truth for pair scoring.
        """
        B, ci, _ = ei.shape
        cj = ej.shape[1]
        s = torch.zeros(B, ci, cj, dtype=ei.dtype, device=ei.device)
        if self.head_type in ("bilinear", "bilinear_mlp"):
            s = s + (ei @ self.bilinear_w) @ ej.transpose(1, 2) + self.bilinear_b
        if self.head_type in ("mlp", "bilinear_mlp"):
            ei_all = ei.unsqueeze(2).expand(B, ci, cj, self.d_emb)
            for j0 in range(0, cj, self.pair_chunk):
                j1 = min(j0 + self.pair_chunk, cj)
                ejb = ej[:, None, j0:j1, :].expand(B, ci, j1 - j0, self.d_emb)
                feats = torch.cat([ei_all[:, :, j0:j1, :], ejb, ei_all[:, :, j0:j1, :] * ejb,
                                   (ei_all[:, :, j0:j1, :] - ejb).abs()], dim=-1)
                s[:, :, j0:j1] = s[:, :, j0:j1] + self.mlp(feats).squeeze(-1)
        return s

    def postprocess(self, raw: torch.Tensor) -> torch.Tensor:
        """raw (n, n) or (B, n, n) -> symmetric, >=0, zero diag, <= max_dist."""
        squeeze = raw.ndim == 2
        if squeeze:
            raw = raw.unsqueeze(0)
        s = 0.5 * (raw + raw.transpose(1, 2))
        dist = F.softplus(s)
        n = s.shape[1]
        eye = torch.eye(n, device=s.device, dtype=s.dtype).unsqueeze(0)
        dist = dist * (1.0 - eye)
        out = dist.clamp(max=self.max_dist)
        return out[0] if squeeze else out

    # -- output ------------------------------------------------------------- #
    def forward(self, embs: torch.Tensor) -> torch.Tensor:
        """embs: (B, n, d_emb) -> (B, n, n) symmetric, >=0, <= max_dist."""
        return self.postprocess(self._score_blocks(embs, embs))


# --------------------------------------------------------------------------- #
# full model
# --------------------------------------------------------------------------- #
class PhyloModel(nn.Module):
    """Encoder-agnostic composition: any encoder + DistanceHead.

    forward(tokens, seq_spans, spans_mask) -> (dist_matrix, embs). Distances
    are predicted in NORMALIZED space (the per-sample scale is applied by the
    training loop / loss, Phase 3).
    """

    def __init__(
        self,
        encoder: nn.Module,
        d_emb: int = 256,
        max_dist: float = 3.0,
        head_type: str = "bilinear_mlp",
        mlp_hidden: int = 256,
    ) -> None:
        super().__init__()
        d_model = int(encoder.d_model)
        self.encoder = encoder
        self.head = DistanceHead(
            d_model=d_model,
            d_emb=d_emb,
            max_dist=max_dist,
            head_type=head_type,
            mlp_hidden=mlp_hidden,
        )

    def forward(
        self,
        tokens: torch.Tensor,
        seq_spans: torch.Tensor,
        spans_mask: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        seq_position_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if position_ids is None or seq_position_ids is None:
            from ssm_phylo.models.encoder import make_position_ids

            position_ids, seq_position_ids = make_position_ids(seq_spans)
        hidden = self.encoder(tokens, position_ids, seq_position_ids)
        embs = self.head.pool_all(hidden, seq_spans, spans_mask)
        dist_matrix = self.head(embs)
        return dist_matrix, embs

    def n_params(self, module: nn.Module | None = None) -> int:
        target = module if module is not None else self
        return sum(p.numel() for p in target.parameters())


# --------------------------------------------------------------------------- #
# smoke CLI
# --------------------------------------------------------------------------- #
def _smoke_config() -> Any:
    from types import SimpleNamespace

    return SimpleNamespace(
        d_model=64,
        n_layer=2,
        vocab_size=38,
        encoder=SimpleNamespace(
            kind="from_scratch",
            checkpoint_dir=None,
            mamba={"state_size": 4, "time_step_rank": 8, "conv_kernel": 3, "expand": 2},
            ptm_model_id="ChatterjeeLab/PTM-Mamba",
        ),
    )


def _smoke_data(batch: int = 2, n_seqs: int = 60, seq_len: int = 60, device: str = "cpu"):
    tokens: list[torch.Tensor] = []
    spans = torch.zeros(batch, n_seqs, 2, dtype=torch.long)
    mask = torch.ones(batch, n_seqs, dtype=torch.bool)
    for b in range(batch):
        parts: list[torch.Tensor] = []
        for k in range(n_seqs):
            start = len(parts)
            parts.append(torch.randint(0, 38, (seq_len,)))
            if k < n_seqs - 1:
                parts.append(torch.full((1,), 20))  # <sep>
            spans[b, k, 0] = start
            spans[b, k, 1] = start + seq_len
        tokens.append(torch.cat(parts))
    return torch.stack(tokens).to(device), spans.to(device), mask.to(device)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ssm_phylo.models.head")
    parser.add_argument("--smoke", action="store_true", help="run the smoke check")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--n-seqs", type=int, default=60)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    if not args.smoke:
        parser.error("nothing to do; pass --smoke")

    t0 = time.time()
    from ssm_phylo.models.encoder import build_encoder

    encoder = build_encoder(_smoke_config(), device=args.device)
    model = PhyloModel(encoder, d_emb=32, max_dist=3.0, head_type="bilinear_mlp").to(args.device)
    tokens, spans, mask = _smoke_data(args.batch, args.n_seqs, device=args.device)
    with torch.no_grad():
        dm, embs = model(tokens, spans, mask)
    sym = torch.allclose(dm, dm.transpose(1, 2), atol=1e-4)
    bounded = bool((dm >= 0.0).all() and (dm <= model.head.max_dist + 1e-4).all())
    head_params = model.n_params(model.head)
    encoder_params = model.n_params(model.encoder)
    print(f"smoke: tokens {tuple(tokens.shape)} -> dist_matrix {tuple(dm.shape)}")
    print(f"smoke: embs {tuple(embs.shape)}; symmetric={sym}; 0<=dm<={model.head.max_dist}={bounded}")
    print(
        f"smoke: params head={head_params:,} encoder={encoder_params:,} "
        f"({head_params / max(encoder_params, 1):.4f}x), wall={time.time() - t0:.1f}s"
    )
    if not sym or not bounded:
        print("smoke: FAILED")
        return 1
    print("smoke: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

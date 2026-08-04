"""Losses for distance-matrix prediction.

Design notes:
- mae_loss: scale-weighted MAE in RAW substitution units, mean |scale*(pred -
  target)| over i<j valid pairs. Per sample, scale is a constant, so this is
  mathematically identical to MAE on raw distances — the real effect is batch
  weighting: deep trees (large scale) contribute proportionally more, which is
  desirable because FastME consumes raw substitution distances. MRE
  fine-tuning (later phase) recovers short-branch relative accuracy. Only the
  strictly-lower-triangle pairs (i<j) count, and padded entries (target < 0)
  are masked out.
- mre_loss: mean relative error |pred - target| / (target + eps). Scale-
  invariant (up to the eps floor) — good for short-branch accuracy.
- four_point_penalty: for every quartet (a,b,c,d), the three pair-sums
  s1=d_ab+d_cd, s2=d_ac+d_bd, s3=d_ad+d_bc satisfy the four-point condition
  (the two largest are equal) for any additive tree metric; the penalty is the
  mean of (max - second_max)/(max + eps) over quartets. Exactly 0 for a
  perfect tree metric, > 0 for a random matrix. All quartets up to n=40
  (C(40,4)=91k), seeded random sample up to 4096 beyond.
- combined_loss returns a dict for logging, including BOTH the normalized MAE
  and the raw-unit MAE so training can record both stories to metrics.csv.
"""
from __future__ import annotations

import itertools

import numpy as np
import torch

_Q_RNG = np.random.default_rng(42)


def _valid_pair_mask(target: torch.Tensor) -> torch.Tensor:
    """(B, n, n) bool: strictly i<j pairs that are not padded (target >= 0)."""
    n = target.shape[-1]
    triu = torch.triu(torch.ones(n, n, dtype=torch.bool, device=target.device), diagonal=1)
    return triu.unsqueeze(0) & (target >= 0)


def mae_loss(
    pred: torch.Tensor, target: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """Scale-weighted (raw-unit) mean absolute error over i<j valid pairs."""
    mask = _valid_pair_mask(target)
    if not mask.any():
        return torch.zeros((), device=pred.device)
    diff = (pred - target) * scale[:, None, None]
    return diff.abs()[mask].mean()


def mre_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """Mean relative error over i<j valid pairs (scale-invariant)."""
    mask = _valid_pair_mask(target)
    if not mask.any():
        return torch.zeros((), device=pred.device)
    rel = (pred - target).abs() / (target + eps)
    return rel[mask].mean()


def four_point_penalty(
    dm: torch.Tensor, sample_quartets: int | None = None
) -> torch.Tensor:
    """Four-point-condition violation penalty on a distance matrix.

    dm: (B, n, n) or (n, n); padded entries (< 0) excluded. Returns the mean
    penalty over valid quartets and batch. Sampling: all quartets when
    C(n,4) <= 4096 (or n <= 40), else a seeded random sample of up to 4096.
    """
    batched = dm.ndim == 3
    if not batched:
        dm = dm.unsqueeze(0)
    n = dm.shape[-1]
    if n < 4:
        return torch.zeros((), device=dm.device)

    total_quartets = n * (n - 1) * (n - 2) * (n - 3) // 24
    if sample_quartets is not None:
        n_q = min(sample_quartets, total_quartets)
    elif total_quartets > 4096:
        n_q = 4096
    else:
        n_q = total_quartets

    if n_q == total_quartets:
        quartets = np.asarray(list(itertools.combinations(range(n), 4)), dtype=np.int64)
    else:
        quartets = _Q_RNG.integers(0, n, size=(n_q, 4))
    q = torch.from_numpy(quartets).to(dm.device)

    d = dm  # (B, n, n)
    dab = d[:, q[:, 0], q[:, 1]]
    dcd = d[:, q[:, 2], q[:, 3]]
    dac = d[:, q[:, 0], q[:, 2]]
    dbd = d[:, q[:, 1], q[:, 3]]
    dad = d[:, q[:, 0], q[:, 3]]
    dbc = d[:, q[:, 1], q[:, 2]]
    s = torch.stack([dab + dcd, dac + dbd, dad + dbc], dim=-1)  # (B, Q, 3)

    valid = (dab >= 0) & (dcd >= 0) & (dac >= 0) & (dbd >= 0) & (dad >= 0) & (dbc >= 0)
    s = s.masked_fill(~valid.unsqueeze(-1), float("-inf"))
    sorted_s = s.sort(dim=-1).values  # ascending; keeps multiplicities
    mx = sorted_s[:, :, 2]
    second = sorted_s[:, :, 1]  # second-largest WITH multiplicity (== mx for a tree metric)
    penalty = (mx - second) / (mx + 1e-8)
    penalty = penalty.masked_fill(mx < 0, 0.0)  # fully-invalid quartets
    return penalty[valid.any(-1)].mean() if valid.any() else torch.zeros((), device=dm.device)


def combined_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    scale: torch.Tensor,
    loss_type: str = "mae",
    lambda_fp: float = 0.01,
    mre_eps: float = 1e-3,
) -> dict[str, torch.Tensor]:
    """Primary loss + four-point regularizer; dict for logging.

    Keys: loss (total, backprop this), primary, four_point, mae_norm
    (normalized-space MAE), mae_raw (raw-unit MAE). Both MAE variants are
    logged so the paper can show both stories.
    """
    if loss_type == "mae":
        primary = mae_loss(pred, target, scale)
    elif loss_type == "mre":
        primary = mre_loss(pred, target, eps=mre_eps)
    else:
        raise ValueError(f"unknown loss_type '{loss_type}'")
    ones = torch.ones_like(scale)
    mae_norm = mae_loss(pred, target, ones)
    mae_raw = mae_loss(pred, target, scale)
    if lambda_fp > 0:
        fp = four_point_penalty(pred)
    else:
        fp = torch.zeros((), device=pred.device)
    total = primary + lambda_fp * fp
    return {
        "loss": total,
        "primary": primary,
        "four_point": fp,
        "mae_norm": mae_norm,
        "mae_raw": mae_raw,
    }

"""Losses: MAE (base, Phyloformer-style), MRE (fine-tune), four-point regularizer.

Implementation lives in ssm_phylo.models.losses; this module re-exports it
and keeps the Phase-0 CLI stub.
"""
from ssm_phylo.models.losses import (  # noqa: F401
    combined_loss,
    four_point_penalty,
    mae_loss,
    mre_loss,
)


def main() -> int:
    """Placeholder: loss functions live here (MAE, MRE, four-point condition)."""
    print("ssm_phylo.losses: MAE / MRE / four-point-condition losses (see models.losses).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Model components: config-driven encoder + task head + losses.

- encoder.py: build_encoder(cfg) with three kinds (from_scratch default,
  degraded_protmamba, dormant ptm_mamba); ProtMambaEncoder, a backend-
  agnostic wrapper over any Mamba-style backbone (hook-based hidden states);
  make_position_ids ("1d" scheme).
- checkpoint_compat.py: shape-inference loader for the ProtMamba v1.0
  release weights (which are internally inconsistent — see its module
  docstring for the full forensics).
- head.py: the NOVEL task head (attention pooling + bilinear/MLP distance
  predictor + hard symmetry/non-negativity/boundedness guarantees);
  PhyloModel composes any encoder with the head.
- losses.py: MAE (raw-unit, scale-weighted), MRE, four-point-condition
  penalty, combined loss with both MAE variants for logging.

Fused mamba kernels are OPTIONAL: `mamba_ssm` imports are never required —
the from_scratch and degraded_protmamba paths use transformers' eager Mamba.
"""

"""Model components: from-scratch encoder + task head + losses.

- encoder.py: build_encoder(cfg) — always a from-scratch transformers
  MambaForCausalLM (random weights, no external checkpoints); MambaEncoder, a
  backend-agnostic wrapper over any Mamba-style backbone (native
  output_hidden_states path with a hook fallback); make_position_ids ("1d").
- head.py: the NOVEL task head (attention pooling + bilinear/MLP distance
  predictor + hard symmetry/non-negativity/boundedness guarantees);
  PhyloModel composes any encoder with the head.
- losses.py: MAE (raw-unit, scale-weighted), MRE, four-point-condition
  penalty, combined loss with both MAE variants for logging.

`kernels` (HuggingFace fused kernels) is an OPTIONAL speedup: the pipeline
runs fully on transformers' eager Mamba without it, and `kernels` imports are
never required.
"""

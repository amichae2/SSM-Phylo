"""ssm-phylo: SSM distance estimator for phylogenetic trees.

A neural network that takes UNALIGNED biological sequences and outputs an
n x n pairwise evolutionary-distance matrix; trees are built from that matrix
with FastME. The encoder is CONFIG-DRIVEN (build_encoder(cfg)): default
`from_scratch` (HF eager MambaForCausalLM, random weights), gated
`degraded_protmamba` (broken ProtMamba v1.0 release via checkpoint_compat),
dormant `ptm_mamba`. The novel code is the task head (per-sequence attention
pooling + bilinear/MLP distance predictor + loss) plus fine-tuning.

Import-time contract: importing this package MUST NOT require mamba-ssm to be
installed. Any optional fused-kernel dependency (mamba_ssm, causal_conv1d)
must be imported lazily inside try/except blocks at point of use.
"""

__version__ = "0.1.0"

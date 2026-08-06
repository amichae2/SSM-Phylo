"""ssm-phylo: SSM distance estimator for phylogenetic trees.

A neural network that takes UNALIGNED biological sequences and outputs an
n x n pairwise evolutionary-distance matrix; trees are built from that matrix
with FastME. The encoder is CONFIG-DRIVEN (build_encoder(cfg)): it always
builds a from-scratch HF transformers MambaForCausalLM (random weights,
license-clean, CI-safe, no weights needed). The novel code is the task head
(per-sequence attention pooling + bilinear/MLP distance predictor + loss)
plus fine-tuning.

Import-time contract: importing this package MUST NOT require `kernels` (the
optional HuggingFace fused-kernel library). Any optional fused-kernel
dependency must be imported lazily at point of use; without it the pipeline
runs on transformers' eager Mamba.
"""

__version__ = "0.1.0"

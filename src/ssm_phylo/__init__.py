"""ssm-phylo: SSM distance estimator for phylogenetic trees.

A neural network that takes UNALIGNED biological sequences and outputs an
n x n pairwise evolutionary-distance matrix; trees are built from that matrix
with FastME. The encoder is a pretrained ProtMamba (Mamba SSM, ~100M params)
reused as-is; the novel code is the task head (per-sequence attention pooling
+ bilinear/MLP distance predictor + loss) plus fine-tuning.

Import-time contract: importing this package MUST NOT require mamba-ssm to be
installed. Any optional fused-kernel dependency (mamba_ssm, causal_conv1d)
must be imported lazily inside try/except blocks at point of use.
"""

__version__ = "0.1.0"

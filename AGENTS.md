# ssm-phylo — SSM Distance Estimator for Phylogenetic Trees

## Project goal
A neural network that takes UNALIGNED biological sequences and outputs an n×n
pairwise evolutionary-distance matrix; trees are built from that matrix with
FastME. The encoder is CONFIG-DRIVEN (build_encoder(cfg), see
src/ssm_phylo/models/encoder.py): it always builds a from-scratch HF
transformers MambaForCausalLM (random weights — license-clean, CI-safe, no
weights needed). The NOVEL code is the task head (per-sequence attention
pooling + bilinear/MLP distance predictor + loss) plus fine-tuning.

## Storage architecture (Google Drive primary — READ THIS FIRST)
Paths use these env vars (set in .env, sourced by every script):
- COLAB_DRIVE=/content/drive/MyDrive/ssm-phylo   (persistent home)
- DATA_DIR=$COLAB_DRIVE/data                      (datasets)
- CKPT_DIR=$COLAB_DRIVE/checkpoints               (checkpoints, mirrored)
- RESULTS_DIR=$COLAB_DRIVE/results                (eval outputs)
- LOCAL_CKPT_DIR=/content/ckpts                   (scratch checkpoints)
- LOCAL_DATA_DIR=/content/data                    (scratch data)

Golden rules:
1. Drive = persistent. /content = scratch (wipes every session). Never rely
   on /content for anything irreplaceable.
2. NEVER write many small files to Drive. Simulated data must be consolidated
   into ONE parquet file per split (train.parquet, val.parquet, test.parquet,
   ood.parquet) before any Drive write. Columns: seqs: list[str],
   tree_newick: str, n_tips: int, (optional) scale: float.
   Raw FASTA/tree archives may live on Drive as a single .tar.gz backup,
   but dataloaders read ONLY parquet.
3. Checkpoint protocol (pull-train-push):
   - At train start: pull CKPT_DIR -> LOCAL_CKPT_DIR (rsync or cp -a).
   - During training: write checkpoints ONLY to LOCAL_CKPT_DIR (atomic
     os.replace, fast local disk).
   - After each save: push LOCAL_CKPT_DIR -> CKPT_DIR (rsync, or cp -a
     fallback if rsync missing). Keep at most 5 checkpoints on Drive
     (latest.pt, best.pt, ckpt-interrupted.pt, plus 2 most recent numbered).
   - Never write a checkpoint file directly to Drive mid-training.
4. Drive FUSE is slower than local disk: batch reads (memory-map parquet),
   batch writes (single-file copies), avoid per-file churn and API-rate hits.
5. Google One quota is ample for this project (10s of GB). Do not build
   storage-saving hacks; build latency-avoiding hacks.
6. HuggingFace Hub is OPTIONAL (publication mirror, Phase 7). It is NOT
   required anywhere in the pipeline.

## Hard constraints (do not violate)
1. INPUT IS UNALIGNED SEQUENCES. Never require a multiple sequence alignment
   at inference. Gap-strip simulated sequences before writing training data.
2. Reuse, don't rebuild: Phyloformer scripts/binaries
   (github.com/lucanest/Phyloformer) for simulation and evaluation; AliSim
   (IQ-TREE >= 2.0) for sequence simulation; FastME for tree building.
3. Target hardware: Google Colab Pro (L4 24GB typical, A100-40GB possible).
   Design for: session death at any moment, ephemeral VM disk, Drive FUSE
   latency. Everything must checkpoint/resume.
4. Fallback-first Mamba: the pipeline runs fully on transformers' EAGER Mamba
   (MambaForCausalLM, transformers>=5). `kernels` (HuggingFace fused kernels,
   fast wheels) is an OPTIONAL speedup installed non-fatally by
   colab_setup.sh; transformers 5.x auto-detects it. mamba-ssm is obsolete
   (does not build on modern torch) and must not be used. Never let an
   install block the pipeline. bf16 only on sm_80+ GPUs (A100/L4/H100); fp16
   fallback on T4 (dev/smoke only).
5. Pins: modern versions are fine. Floor pins only (torch>=2.1,
   transformers>=5.0, numpy>=1.24) in pyproject.toml and
   requirements-colab.txt — strict pins would downgrade Colab's preinstalled
   torch/numpy on every fresh VM. Rest unpinned: pandas, pyarrow, scipy,
   scikit-learn, biopython, dendropy, tqdm, PyYAML, wandb, tensorboard, plus
   pytest, ruff, mypy for dev.
6. Reproducibility: fixed seeds everywhere (42), configs in configs/*.yaml,
   every training run logs step/epoch/loss and writes a resume-able checkpoint.
7. Output distance matrix must be SYMMETRIC and NON-NEGATIVE. Predict only i<j
   pairs and symmetrize, or tie weights. Apply softplus before output.
8. All new Python lives under src/ssm_phylo/ with tests in tests/.
   Code must run headless (no notebook-only logic); notebooks are thin wrappers.
9. Never default to full 20-epoch schedules for validation. Use
   configs/validate.yaml (or --max-epochs overrides) for end-to-end checks —
   the notebook's Section 3 defaults to run_mode="validate".
10. Fused kernels (`kernels`, HuggingFace) are the ONLY big speed lever
    (3-8x+); they install as fast wheels, safe anytime (non-fatally via
    colab_setup.sh, no torch downgrade). Eager Mamba is the default and is
    CPU-bound (GPU idle, minutes/step at 100M): document this, never silently
    let a user start a 20-epoch eager run as if it were quick.

## Key reference facts (from the source repos)
- Phyloformer sim: `simulate_trees.py --ntips N --ntrees M --type birth-death`;
  `alisim.py --substitution LG --gamma GC --length L [--indel]`; tree/alignment
  pairs share filenames (.nwk vs .fasta). Eval: `fastme -i dm -o tree.nwk
  --nni --spr`, then `phylocompare -t -n -o outdir trees predicted`.
  Reference average KF distance on their test set: 0.333.
- Distance target: MAE loss first (Phyloformer's base), MRE fine-tune second.
- Four-point condition: for any 4 taxa, the two largest of the three pair-sums
  must be equal. Use as a soft regularizer (sampled quartets when n > 40).
- Baselines to report: BIONJ/FastME on TRUE distances (upper bound),
  Phyloformer pretrained, Mash/Skmer, IQ-TREE on MAFFT MSA.

## Definition of done for any phase
- Tests pass: `python -m pytest tests/ -x -q`
- Smoke run works on Colab (T4-free-tier acceptable for dev)
- Every script has `--help` and sensible defaults
- No absolute paths; all paths via config/env (COLAB_DRIVE, DATA_DIR, etc.)
- No simulated-data file count explosion: everything consolidated to parquet
# ssm-phylo — SSM Distance Estimator for Phylogenetic Trees

## Project goal
A neural network that takes UNALIGNED biological sequences and outputs an n×n
pairwise evolutionary-distance matrix; trees are built from that matrix with
FastME. The encoder is CONFIG-DRIVEN (build_encoder(cfg), see
src/ssm_phylo/models/encoder.py): the default kind is `from_scratch` (HF
eager MambaForCausalLM, random weights — license-clean, CI-safe, no weights
needed); `degraded_protmamba` loads the ProtMamba v1.0 backbone via
checkpoint_compat.py as a gated experiment (the release weights are broken —
see "Known issue" below); `ptm_mamba` is dormant (no public weights, and the
real PTM-Mamba is cc-by-nc-nd-4.0). The NOVEL code is the task head
(per-sequence attention pooling + bilinear/MLP distance predictor + loss)
plus fine-tuning.

## Storage architecture (Google Drive primary — READ THIS FIRST)
Paths use these env vars (set in .env, sourced by every script):
- COLAB_DRIVE=/content/drive/MyDrive/ssm-phylo   (persistent home)
- DATA_DIR=$COLAB_DRIVE/data                      (datasets)
- CKPT_DIR=$COLAB_DRIVE/checkpoints               (checkpoints, mirrored)
- RESULTS_DIR=$COLAB_DRIVE/results                (eval outputs)
- PROT_MAMBA_CKPT=$COLAB_DRIVE/weights/protmamba  (encoder weights)
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
2. Reuse, don't rebuild: ProtMamba weights (Bitbol-Lab/ProtMamba-ssm,
   Apache-2.0, weights in GitHub releases) for the encoder; Phyloformer
   scripts/binaries (github.com/lucanest/Phyloformer) for simulation and
   evaluation; AliSim (IQ-TREE >= 2.0) for sequence simulation; FastME for
   tree building.
3. Target hardware: Google Colab Pro (L4 24GB typical, A100-40GB possible).
   Design for: session death at any moment, ephemeral VM disk, Drive FUSE
   latency. Everything must checkpoint/resume.
4. Fallback-first Mamba: the pipeline runs fully on transformers' EAGER Mamba
   (MambaForCausalLM). mamba-ssm is OPTIONAL — never installed by default, only
   attempted if it builds (it is gated behind SSM_PHYLO_TRY_MAMBA=1 in
   colab_setup.sh; mamba-ssm==2.1.0 is the last source-buildable version and
   needs an exact old-torch env). Never let the install block the pipeline.
   bf16 only on sm_80+ GPUs (A100/L4/H100); fp16 fallback on T4 (dev/smoke
   only).
5. Pins: modern versions are fine. Floor pins only (torch>=2.1,
   transformers>=4.44, numpy>=1.24) in pyproject.toml and
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

## Key reference facts (from the source repos)
- ProtMamba config: d_model=1024, n_layer=16, vocab_size=38, bf16,
  max_msa_len=131072, lr=6e-4, weight_decay=0.1, betas=(0.9,0.95),
  warmup_steps=500, constant scheduler, save_steps=250, position ids "1d".
  Tokenizer reads a3m files (sequences need NOT be aligned), concatenate=True.
  Hidden states: `get_hidden_states(model, tokens, which_layers, position_ids,
  seq_position_ids)` returns per-layer tensors.
- ProtMamba special tokens: AA_TO_ID includes <cls>, <mask-1..5>, <pad>, <unk>.
  Use <cls> as the per-sequence terminator/separator.
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

Known issue: ProtMamba v1.0 release weights are internally inconsistent (embedding/lm_head 512-dim, vocab 40 vs backbone 1024-dim, vocab 38; config.json lies). Never trust config.json for these weights; always use checkpoint_compat.py's shape-inference loader. Do not spend time re-diagnosing.
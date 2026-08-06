# ssm-phylo

SSM distance estimator for phylogenetic trees. A neural network that takes
**unaligned** biological sequences and outputs an n×n pairwise
evolutionary-distance matrix; trees are built from that matrix with FastME.

The encoder is **config-driven** (`build_encoder(cfg)`): it always builds a
from-scratch HF transformers `MambaForCausalLM` (random weights —
license-clean, CI-safe, no weights needed) on the modern transformers 5.x
stack. The novel code is the task head (per-sequence attention pooling +
bilinear/MLP distance predictor + loss) and fine-tuning.

## Licensing

- **From-scratch Mamba encoder** — your own random-init `MambaForCausalLM`
  (transformers, no external weights): fully clean, no third-party weights.
- All other model weights and dependencies are optional speedups only
  (`kernels`, HuggingFace's fused kernels).

## Storage map

| What | Where | Purpose |
|---|---|---|
| Data (parquet per split) | `$DATA_DIR` (Google Drive) | persistent datasets — **source of truth** |
| Raw simulated trees/FASTA | `$LOCAL_DATA_DIR/raw` (`/content`) | scratch ONLY, ephemeral, deletable |
| Checkpoints | `$CKPT_DIR` (Drive) / `$LOCAL_CKPT_DIR` (`/content`) | mirror, pull→train→push |
| Eval outputs | `$RESULTS_DIR` (Drive) | persistent eval results |
| Code, configs, tests | this GitHub repo | everything reproducible |

Drive = persistent; `/content` = scratch (wiped every session). Nothing
irreplaceable ever lives only on `/content`. Simulated data is generated
into local scratch ONLY, then consolidated into **one parquet file per
split** (`train.parquet`, `val.parquet`, `test.parquet`, `ood.parquet` with
columns `seqs: list[str]`, `tree_newick: str`, `n_tips: int`,
`scale: float`) and copied to Drive atomically (tmp + `os.replace`). If a
session dies mid-simulation, the raw scratch is lost but nothing on Drive is
corrupt — re-run to resume.

## Repo layout

```
pyproject.toml            package metadata, floor pins (kernels optional extra)
requirements-colab.txt    Colab install order + fallback strategy
configs/                  default.yaml + per-hardware overrides (merge chain)
src/ssm_phylo/            data/ (simulation.py, datasets.py), models/
                          (encoder.py, head.py, losses.py),
                          losses.py, train.py, infer.py, build_tree.py, evaluate.py
scripts/                  colab_setup.sh, sync_drive.sh, simulate_big.sh
tests/                    pytest suite
```

## Models (Phase 2)

`build_encoder(cfg)` is config-driven (`configs/default.yaml` → `encoder.mamba`
hyperparams) and always builds a **from-scratch** HF `MambaForCausalLM`
(random weights, d_model 1024 / n_layer 16 / vocab 38 — CI-safe, no weights,
license-clean) on the modern transformers 5.x stack. `kernels` (HuggingFace
fused kernels) is an optional speedup auto-detected by transformers 5.x; the
eager Mamba fallback works everywhere.

`PhyloModel` = any encoder + the novel task head (`head.py`): learned-query
attention pooling per sequence → d_emb 256 → bilinear core + pair MLP
(`[e_i; e_j; e_i*e_j; |e_i-e_j|]`), symmetrized, softplus, clipped at
`max_dist` 3.0 → guaranteed symmetric, non-negative, bounded distance
matrix in normalized space. Losses (`models/losses.py`): scale-weighted
raw-unit MAE, MRE, four-point-condition penalty, combined loss logging both
normalized and raw MAE.

## Data pipeline (Phase 1)

- `simulation.py`: `simulate_trees` (vendored Phyloformer `simulate_trees.py`
  when `PHYLOFORMER_DIR` is set, else internal dendropy birth-death/yule/
  kingman-coalescent), `simulate_alignments` (AliSim via `iqtree --alisim`
  when a binary is found; pure-python uniform-substitution fallback with
  `engine="python"` for dev/smoke boxes without iqtree — gap characters are
  STRIPPED from the unaligned FASTA, an `aligned/` copy is kept for
  baselines), `consolidate_to_parquet` (one snappy parquet per split, sorted
  by n_tips, chunked), `make_splits` (seeded per-tree hash split → atomic
  copies to Drive), `make_ood_set`, `clean_scratch`.
- `datasets.py`: `ParquetPhyloDataset` (memory-mapped parquet; yields
  `tokens, seq_spans, true_distances/scale, scale, tree_newick` with a
  dedicated `<sep>` token), `collate_with_bucketing` (per-seq truncation to
  `max_seq_len`, bucketed padding), `load_dataset`.

### How to regenerate data

```bash
# full-scale resumable job (>=100k pairs) — run on Colab:
bash scripts/simulate_big.sh                     # background with --background
bash scripts/simulate_big.sh --smoke             # 200 alignments, full path
python -m ssm_phylo.data.simulation --smoke      # same, direct
# OOD set:
python -m ssm_phylo.data.simulation ood --raw $LOCAL_DATA_DIR/raw_ood --data-dir $DATA_DIR
# delete raw scratch once parquet is safe on Drive:
python -m ssm_phylo.data.simulation clean --raw $LOCAL_DATA_DIR/raw
```

Every step is resumable: existing `.nwk`/`.fasta` files and existing parquet
are skipped. If the parquet schema ever changes, re-consolidate with:

```bash
python -m ssm_phylo.data.simulation splits --raw $LOCAL_DATA_DIR/raw --data-dir $DATA_DIR
```

(after removing the stale `$DATA_DIR/*.parquet`; raw scratch must still
exist — never rely on Drive parquet as the only copy of raw data before the
dataset is final).

## How to run on Colab

**Single entry point: [`notebooks/V1_Colab.ipynb`](notebooks/V1_Colab.ipynb)** — one
notebook that runs the whole V1 flow in a single VM (setup → simulate small
data → train → evaluate). Colab gives every notebook a fresh VM, so the
environment does NOT persist between notebooks; run everything in one.

1. Open `notebooks/V1_Colab.ipynb` in Colab and run Section 1 (setup).
2. Run Section 2 (simulate small data), Section 3 (train; `--resume latest`
   makes re-runs resume after session death), Section 4 (evaluate).
3. Every cell is idempotent — re-running never duplicates work.

The old per-phase notebooks (`00_setup` … `03_evaluate`) are kept for
reference/CI only. For full-scale data, `scripts/simulate_big.sh` generates
≥100k pairs (resumable); `scripts/colab_train.sh` trains with
`configs/train_l4.yaml`.

## Why is training slow? / Making it fast

**The short version:** the default runtime uses transformers' **eager Mamba**,
whose sequential scan runs on the **CPU** — the GPU sits idle while a 100M-param
step can take ~minutes. 20 epochs on eager Mamba ≈ **1.7 days**. HuggingFace's
**`kernels`** fused kernels move the scan onto the GPU (**3-8x+ faster**, days →
hours) and are the *only* big speed lever — installed as fast wheels (no torch
downgrade, safe anytime) by `scripts/colab_setup.sh`; transformers 5.x
auto-detects the library.

**The recipe:**

1. **For any check** (does it run? does the loss go down?): use **VALIDATION
   MODE** — in the notebook, Section 3's `run_mode` toggle defaults to
   `"validate"`, which trains the real 100M model for just 2 epochs via
   `configs/validate.yaml` (minutes even on eager Mamba). Never run a full
   20-epoch schedule just to validate.
2. **Before real training**: ensure `kernels` is installed — `pip install
   kernels` (or the Section 1 notebook cell; `colab_setup.sh` already tries
   it non-fatally). No fresh-runtime dance, no downgrades. Then run
   `train_l4`/`train_a100` as usual.

If you stay on eager Mamba: early stopping (`early_stop_patience`) and fewer
epochs (`--max-epochs`) are your friends, and the notebook prints the
expected-runtime notice before every full-schedule run.

## How to run locally

```bash
pip install -e .[dev]
python -m pytest tests/ -x -q
python -m ssm_phylo.data.simulation --smoke --data-dir <local-dir> --local-data-dir <scratch-dir>
```

Local runs need no GPU, no Drive, and no `kernels`: importing the package and
running the test suite must never require the fused kernels (eager Mamba
fallback). Without an iqtree binary, simulation falls back to the
pure-python engine (dev/smoke only).

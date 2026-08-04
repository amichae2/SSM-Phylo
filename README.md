# ssm-phylo

SSM distance estimator for phylogenetic trees. A neural network that takes
**unaligned** biological sequences and outputs an n×n pairwise
evolutionary-distance matrix; trees are built from that matrix with FastME.

The encoder is **config-driven** (`build_encoder(cfg)`): the default kind is
`from_scratch` (HF eager MambaForCausalLM, random weights — license-clean,
CI-safe, no weights needed); `degraded_protmamba` loads the ProtMamba v1.0
backbone via `checkpoint_compat.py` as a gated experiment; `ptm_mamba` is
dormant. The novel code is the task head (per-sequence attention pooling +
bilinear/MLP distance predictor + loss) and fine-tuning.

## Known issues / checkpoint provenance

The ProtMamba **v1.0** GitHub release weights (`$PROT_MAMBA_CKPT`) are
internally inconsistent (see `src/ssm_phylo/models/checkpoint_compat.py` for
the full forensics):

| Tensor | Actual shape | Config claims |
|---|---|---|
| `backbone.embedding.weight` / `lm_head.weight` | (40, **512**) | vocab 38, d_model 1024 |
| `backbone.layers.{0-15}.*.mixer.ckpt_layer.*` | d_model **1024** | d_model 1024 |
| `backbone.position_embedding.weight` | (2048, 512) | — |

No coherent architecture can load it fully. Workaround (already implemented,
`encoder.kind: degraded_protmamba`): infer backbone shapes from the state
dict only, remap `mixer.ckpt_layer.*` → `mixer.*`, load the 1024-dim
backbone tensors, **drop** the mismatched embedding/lm_head and re-initialize
at (38, 1024), gated on a finite-loss forward. The embedding is thus random;
Phase 3's fine-tuning includes the encoder.

To re-check a future release: inspect `pytorch_model.bin` shapes — a healthy
checkpoint has `embedding.weight` and `lm_head.weight` with second dimension
equal to `backbone.norm_f.weight`'s first dimension, and `vocab` rows equal
to the config's `vocab_size`.

## Licensing

- `encoder.kind: from_scratch` — your own random-init eager Mamba: fully
  clean, no third-party weights.
- `encoder.kind: degraded_protmamba` — ProtMamba backbone weights, Apache-2.0:
  clean for any release.
- `encoder.kind: ptm_mamba` — **dormant by design**: no public weights exist
  for the requested id, and the real PTM-Mamba is `cc-by-nc-nd-4.0` (the "no
  derivatives" clause could make fine-tuning or representation extraction
  derivative work, contaminating a future release). Private research only,
  via a local checkout pointed to by `SSM_PHYLO_PTM_MAMBA_DIR`.

## Storage map

| What | Where | Purpose |
|---|---|---|
| Data (parquet per split) | `$DATA_DIR` (Google Drive) | persistent datasets — **source of truth** |
| Raw simulated trees/FASTA | `$LOCAL_DATA_DIR/raw` (`/content`) | scratch ONLY, ephemeral, deletable |
| Checkpoints | `$CKPT_DIR` (Drive) / `$LOCAL_CKPT_DIR` (`/content`) | mirror, pull→train→push |
| Eval outputs | `$RESULTS_DIR` (Drive) | persistent eval results |
| ProtMamba weights | `$PROT_MAMBA_CKPT` (Drive) | encoder weights |
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
pyproject.toml            package metadata, pinned deps (mamba-ssm optional!)
requirements-colab.txt    Colab install order + fallback strategy
configs/                  default.yaml + per-hardware overrides (merge chain)
src/ssm_phylo/            data/ (simulation.py, datasets.py), models/
                          (encoder.py, checkpoint_compat.py, head.py, losses.py),
                          losses.py, train.py, infer.py, build_tree.py, evaluate.py
scripts/                  colab_setup.sh, download_weights.sh, sync_drive.sh,
                          simulate_big.sh
tests/                    pytest suite
```

## Models (Phase 2)

`build_encoder(cfg)` is config-driven (`configs/default.yaml` → `encoder.kind`):

- **from_scratch** (default): HF eager `MambaForCausalLM`, random weights,
  d_model 1024 / n_layer 16 / vocab 38 — CI-safe, no weights, license-clean.
- **degraded_protmamba**: ProtMamba v1.0 backbone via `checkpoint_compat.py`
  (shape-inferred, ckpt_layer remap, embedding re-initialized, finite-loss
  gate) from `$PROT_MAMBA_CKPT`.
- **ptm_mamba**: dormant (see Licensing).

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

1. **Setup** (runs from a terminal cell):
   `bash scripts/colab_setup.sh` — mounts Drive, sources `$COLAB_DRIVE/.env`,
   creates dirs, installs pinned deps, attempts fused mamba on sm_80+.
2. `bash scripts/download_weights.sh` — ProtMamba weights into
   `$PROT_MAMBA_CKPT` (idempotent, locked against concurrent downloads).
3. `bash scripts/simulate_big.sh` — simulate >=100k unaligned pairs into
   scratch, consolidate + split, copy parquet to Drive (resumable).
4. Notebooks `01_simulate_data`, `02_train`, `03_evaluate` (added in later
   phases) — thin wrappers around the `ssm_phylo` package.

## How to run locally

```bash
pip install -e .[dev]
python -m pytest tests/ -x -q
python -m ssm_phylo.data.simulation --smoke --data-dir <local-dir> --local-data-dir <scratch-dir>
```

Local runs need no GPU, no Drive, and no mamba-ssm: importing the package and
running the test suite must never require the fused kernels (eager Mamba
fallback). Without an iqtree binary, simulation falls back to the
pure-python engine (dev/smoke only).

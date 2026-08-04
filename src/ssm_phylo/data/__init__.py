"""Data loading and preprocessing (parquet-based, unaligned sequences).

Storage rule: simulated raw data lives in LOCAL scratch ($LOCAL_DATA_DIR/raw,
/content on Colab) ONLY; it is consolidated into ONE parquet file per split
(columns: seqs list[str], tree_newick str, n_tips int, scale float) before any
Drive write. Dataloaders read ONLY parquet from $DATA_DIR (memory-mapped).

Modules:
- simulation.py: tree + alignment simulation (AliSim or python fallback),
  consolidation, train/val/test/ood splits, atomic Drive copies.
- datasets.py: ParquetPhyloDataset (memory-mapped), tokenizer, collate with
  bucketing, load_dataset.
"""

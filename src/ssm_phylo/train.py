"""Phase 3 training entrypoint — survives Colab session death and Drive latency.

Checkpoint protocol (pull-train-push, AGENTS.md):
  pull   --resume != none: sync_drive.sh pull (Drive -> LOCAL_CKPT_DIR) FIRST
  train  checkpoints written ONLY to LOCAL_CKPT_DIR (atomic tmp + os.replace),
         latest.pt refreshed by copy; numbered ckpts pruned to save_total_limit
  push   after every save (and on SIGTERM): sync_drive.sh push in a BACKGROUND
         subprocess; never queued (skip if one is in flight); on completion
         remote numbered checkpoints are pruned to the 2 most recent, keeping
         latest.pt / best.pt / ckpt-interrupted.pt (max 5 mirrored on Drive)
  resume --resume latest/best/path restores model+optimizer+scheduler+RNG
         state; global_step continues monotonically (never double-counted).

Precision: bf16 on sm_80+, fp16+GradScaler otherwise on CUDA, fp32 on CPU —
requested precision auto-downgrades with a warning instead of crashing.

Distributed: single-GPU only (Colab). No DDP.

Usage:
  python -m ssm_phylo.train --config configs/train_l4.yaml --resume latest
  python -m ssm_phylo.train --smoke            # tiny end-to-end check
  python -m ssm_phylo.train --toy              # 500-sample / 50-step loss check
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import json
import logging
import math
import os
import random
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import yaml

from ssm_phylo.data.datasets import (
    collate_with_bucketing,
    get_tokenizer,
    load_dataset,
)
from ssm_phylo.models.encoder import build_encoder
from ssm_phylo.models.head import PhyloModel
from ssm_phylo.models.losses import combined_loss

log = logging.getLogger("ssm_phylo.train")

REPO_ROOT = Path(__file__).resolve().parents[2]
SYNC_SCRIPT = REPO_ROOT / "scripts" / "sync_drive.sh"

CSV_FIELDS = [
    "step", "epoch", "loss", "primary", "mae", "mae_norm", "mre", "fp_penalty",
    "lr", "val_loss", "val_mae",
]
# --------------------------------------------------------------------------- #
# config loading / merging
# --------------------------------------------------------------------------- #
def _env(name: str, default: str | None = None) -> str | None:
    v = os.environ.get(name)
    return v if v not in (None, "") else default


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base (override wins)."""
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_config(path: str) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def _flatten_overrides(args: argparse.Namespace) -> dict:
    """Map CLI override flags onto config keys (dashes -> underscores)."""
    mapping = {
        "max_seq_len": "max_seq_len", "batch_size": "batch_size", "lr": "lr",
        "max_epochs": "max_epochs", "max_steps": "max_steps",
        "precision": "precision", "save_every": "save_every",
        "val_every": "val_every", "warmup_steps": "warmup_steps",
        "scheduler": "scheduler", "loss": "loss", "seed": "seed",
        "grad_accum": "grad_accum", "max_grad_norm": "max_grad_norm",
        "save_total_limit": "save_total_limit", "bucket_step": "bucket_step",
        "early_stop_patience": "early_stop_patience",
        "hard_loss_ceiling": "hard_loss_ceiling", "lambda_fp": "four_point_lambda",
        "log_every": "log_every", "num_train_alignments": "num_train_alignments",
    }
    overrides: dict[str, Any] = {}
    for flag, key in mapping.items():
        value = getattr(args, flag, None)
        if value is not None:
            overrides[key] = value
    if args.grad_checkpointing is not None:
        overrides["grad_checkpointing"] = args.grad_checkpointing
    if args.name is not None:
        overrides["name"] = args.name
    if args.log is not None:
        overrides["log"] = args.log
    return overrides


def _as_namespace(d: dict) -> SimpleNamespace:
    ns = SimpleNamespace()
    for k, v in d.items():
        setattr(ns, k, _as_namespace(v) if isinstance(v, dict) else v)
    return ns


def _resolve_paths(cfg: dict, args: argparse.Namespace) -> dict:
    """data/ckpt/results dirs: CLI > config > env > scratch defaults."""
    data_dir = args.data_dir or cfg.get("data_dir") or _env("DATA_DIR")
    ckpt_dir = args.ckpt_dir or cfg.get("ckpt_dir") or _env("CKPT_DIR")
    results_dir = args.results_dir or cfg.get("results_dir") or _env("RESULTS_DIR")
    local_ckpt = cfg.get("local_ckpt_dir") or _env("LOCAL_CKPT_DIR")
    local_data = cfg.get("local_data_dir") or _env("LOCAL_DATA_DIR")
    if not data_dir:
        raise SystemExit("no data dir: set DATA_DIR or pass --data-dir")
    if not ckpt_dir:
        raise SystemExit("no checkpoint dir: set CKPT_DIR or pass --ckpt-dir")
    if not local_ckpt:
        local_ckpt = os.path.join(tempfile.gettempdir(), "ssm_phylo_ckpts")
    if not local_data:
        local_data = os.path.join(tempfile.gettempdir(), "ssm_phylo_data")
    return {
        "data_dir": os.path.abspath(data_dir),
        "ckpt_dir": os.path.abspath(ckpt_dir),
        "results_dir": os.path.abspath(results_dir) if results_dir else None,
        "local_ckpt_dir": os.path.abspath(local_ckpt),
        "local_data_dir": os.path.abspath(local_data),
    }


# --------------------------------------------------------------------------- #
# precision
# --------------------------------------------------------------------------- #
def _resolve_precision(requested: str, device: torch.device):
    """(autocast_dtype | None, use_grad_scaler). Auto-downgrades loudly."""
    if requested not in ("bf16", "fp16", "fp32"):
        log.warning("unknown precision %r; using fp32", requested)
        requested = "fp32"
    if device.type == "cpu":
        if requested != "fp32":
            log.warning("no GPU: auto-downgrading %s -> fp32", requested)
        return None, False
    if requested == "bf16":
        cc = torch.cuda.get_device_capability(device)
        if cc[0] >= 8:
            return torch.bfloat16, False
        log.warning("GPU sm_%d.%d lacks bf16: downgrading bf16 -> fp16+GradScaler", *cc)
        return torch.float16, True
    if requested == "fp16":
        return torch.float16, True
    return None, False


# --------------------------------------------------------------------------- #
# CSV logger (append-only, crash-safe)
# --------------------------------------------------------------------------- #
class CsvLogger:
    """Append-only metrics.csv; one flushed write per row, survives crashes."""

    def __init__(self, path: str) -> None:
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._new = not os.path.exists(path)
        self._write_header()

    def _write_header(self) -> None:
        if self._new:
            with open(self.path, "w") as fh:
                fh.write(",".join(CSV_FIELDS) + "\n")

    def log(self, **row: Any) -> None:
        line = ",".join(str(row.get(f, "")) for f in CSV_FIELDS)
        with open(self.path, "a") as fh:
            fh.write(line + "\n")
            fh.flush()


# --------------------------------------------------------------------------- #
# drive sync (background, never blocks, never queues)
# --------------------------------------------------------------------------- #
class DriveSync:
    """Background push manager implementing the 'never queue' rule."""

    def __init__(
        self,
        ckpt_dir: str,
        local_ckpt_dir: str,
        results_dir: str | None,
        local_metrics: str,
        enabled: bool,
        push_cmd: str | None = None,
    ) -> None:
        self.ckpt_dir = ckpt_dir
        self.local_ckpt_dir = local_ckpt_dir
        self.results_dir = results_dir
        self.local_metrics = local_metrics
        self.enabled = enabled
        self.push_cmd = push_cmd or os.environ.get("SSM_PHYLO_PUSH_CMD") or str(SYNC_SCRIPT)
        self._proc: subprocess.Popen | None = None
        self._pushes_started = 0

    def _mirror_metrics(self) -> None:
        if not self.results_dir or not os.path.exists(self.local_metrics):
            return
        dst = os.path.join(self.results_dir, "logs", "metrics.csv")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        tmp = dst + ".tmp"
        shutil.copy2(self.local_metrics, tmp)
        os.replace(tmp, dst)

    def push(self) -> bool:
        """Start one background push; return True if started, False if skipped."""
        if not self.enabled:
            return False
        if self._proc is not None and self._proc.poll() is None:
            log.info("push already running; skipping (never queue)")
            return False
        self._mirror_metrics()
        env = os.environ.copy()
        env["CKPT_DIR"] = self.ckpt_dir
        env["LOCAL_CKPT_DIR"] = self.local_ckpt_dir
        log.info("background push started (cmd=%s)", self.push_cmd)
        self._proc = subprocess.Popen(
            [self.push_cmd, "push"], env=env,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        self._pushes_started += 1
        threading.Thread(target=self._watch, daemon=True).start()
        return True

    def _watch(self) -> None:
        proc = self._proc
        if proc is None:
            return
        rc = proc.wait()
        log.info("background push finished (rc=%d)", rc)
        if rc == 0:
            self._prune_remote()

    def _prune_remote(self) -> None:
        """Keep at most 2 numbered checkpoints on Drive (5 total incl. specials)."""
        try:
            numbered = []
            for name in os.listdir(self.ckpt_dir):
                m = re.match(r"^ckpt-(\d+)\.pt$", name)
                if m:
                    numbered.append((int(m.group(1)), name))
            numbered.sort()
            for _, name in numbered[:-2]:
                os.remove(os.path.join(self.ckpt_dir, name))
                meta = name.replace(".pt", ".meta.json")
                if os.path.exists(os.path.join(self.ckpt_dir, meta)):
                    os.remove(os.path.join(self.ckpt_dir, meta))
            if numbered[:-2]:
                log.info("pruned %d old remote numbered checkpoint(s)", len(numbered) - 2)
        except OSError as exc:
            log.warning("remote prune failed (Drive transient?): %s", exc)

    def wait(self, timeout: float = 600.0) -> None:
        """Block until the in-flight push (if any) completes."""
        proc = self._proc
        if proc is not None and proc.poll() is None:
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                log.warning("push still in flight after %.0fs; leaving it to finish", timeout)

    def pull(self) -> None:
        env = os.environ.copy()
        env["CKPT_DIR"] = self.ckpt_dir
        env["LOCAL_CKPT_DIR"] = self.local_ckpt_dir
        subprocess.run([self.push_cmd, "pull"], env=env, check=False)


# --------------------------------------------------------------------------- #
# checkpointing
# --------------------------------------------------------------------------- #
def _rng_state() -> dict:
    state = {"python": random.getstate(), "numpy": np.random.get_state()}
    state["torch"] = torch.get_rng_state()
    try:
        if torch.cuda.is_available():
            state["torch_cuda"] = torch.cuda.get_rng_state_all()
    except Exception:
        log.warning("could not capture CUDA RNG state", exc_info=True)
    return state


def _restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    cuda = state.get("torch_cuda")
    if cuda and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda)


def _meta_file(ckpt_name: str) -> str:
    return ckpt_name.replace(".pt", ".meta.json")


def _write_meta(local_dir: str, ckpt_name: str, meta: dict) -> None:
    tmp = os.path.join(local_dir, _meta_file(ckpt_name) + ".tmp")
    with open(tmp, "w") as fh:
        json.dump(meta, fh)
    os.replace(tmp, os.path.join(local_dir, _meta_file(ckpt_name)))


def _read_metas(local_dir: str) -> dict[str, dict]:
    metas: dict[str, dict] = {}
    for name in os.listdir(local_dir):
        if name.endswith(".meta.json"):
            try:
                with open(os.path.join(local_dir, name)) as fh:
                    metas[name] = json.load(fh)
            except (OSError, json.JSONDecodeError):
                continue
    return metas


def _scan_resume_source(resume: str, local_dir: str) -> tuple[str, int] | None:
    """Find the checkpoint with the highest global_step (numbered + interrupted)."""
    if resume == "best":
        path = os.path.join(local_dir, "best.pt")
        if not os.path.exists(path):
            return None
        meta = _read_metas(local_dir).get("best.meta.json", {})
        return path, int(meta.get("global_step", -1))
    metas = _read_metas(local_dir)
    numbered = []
    for name, meta in metas.items():
        m = re.match(r"^ckpt-(\d+)\.meta\.json$", name)
        if m and os.path.exists(os.path.join(local_dir, name.replace(".meta.json", ".pt"))):
            numbered.append((int(meta.get("global_step", int(m.group(1)))), name))
    interrupted = None
    if os.path.exists(os.path.join(local_dir, "ckpt-interrupted.pt")):
        meta = metas.get("ckpt-interrupted.meta.json", {})
        interrupted = (int(meta.get("global_step", -1)), "ckpt-interrupted.pt")
    candidates = numbered + ([interrupted] if interrupted else [])
    if not candidates:
        return None
    step, meta_name = max(candidates, key=lambda c: c[0])
    return os.path.join(local_dir, meta_name.replace(".meta.json", ".pt")), step


class Checkpointer:
    """Atomic local saves + latest/best refresh + local pruning."""

    def __init__(self, local_dir: str, save_total_limit: int, max_dist: float) -> None:
        self.local_dir = local_dir
        self.save_total_limit = save_total_limit
        self.max_dist = max_dist
        os.makedirs(local_dir, exist_ok=True)

    def _ckpt_dict(self, trainer: Trainer, val_metric: float | None) -> dict:
        return {
            "epoch": trainer.epoch,
            "step": trainer.step_in_epoch,
            "global_step": trainer.global_step,
            "model_state": trainer.model.state_dict(),
            "optimizer_state": trainer.optimizer.state_dict(),
            "scheduler_state": trainer.scheduler.state_dict() if trainer.scheduler else None,
            "rng_state": _rng_state(),
            "config": trainer.cfg,
            "scale": trainer.scale_factor,  # global scale for raw-unit inference
            "val_metric": val_metric,
        }

    def save(self, trainer: Trainer, val_metric: float | None = None) -> str:
        name = f"ckpt-{trainer.global_step}.pt"
        tmp = os.path.join(self.local_dir, f".tmp-{trainer.global_step}.pt")
        torch.save(self._ckpt_dict(trainer, val_metric), tmp)
        os.replace(tmp, os.path.join(self.local_dir, name))
        _write_meta(
            self.local_dir, name,
            {"global_step": trainer.global_step, "epoch": trainer.epoch,
             "step": trainer.step_in_epoch, "val_metric": val_metric},
        )
        shutil.copy2(os.path.join(self.local_dir, name), os.path.join(self.local_dir, "latest.pt"))
        shutil.copy2(os.path.join(self.local_dir, _meta_file(name)),
                     os.path.join(self.local_dir, "latest.meta.json"))
        self._prune_local()
        return os.path.join(self.local_dir, name)

    def save_as(self, trainer: Trainer, name: str, val_metric: float | None = None) -> str:
        tmp = os.path.join(self.local_dir, f".tmp-{name}.pt")
        torch.save(self._ckpt_dict(trainer, val_metric), tmp)
        path = os.path.join(self.local_dir, name)
        os.replace(tmp, path)
        _write_meta(self.local_dir, name, {"global_step": trainer.global_step,
                                           "epoch": trainer.epoch, "step": trainer.step_in_epoch,
                                           "val_metric": val_metric})
        return path

    def _prune_local(self) -> None:
        numbered = []
        for name in os.listdir(self.local_dir):
            m = re.match(r"^ckpt-(\d+)\.pt$", name)
            if m:
                numbered.append((int(m.group(1)), name))
        numbered.sort()
        for _, name in numbered[:-self.save_total_limit]:
            os.remove(os.path.join(self.local_dir, name))
            meta = os.path.join(self.local_dir, _meta_file(name))
            if os.path.exists(meta):
                os.remove(meta)

    @staticmethod
    def load(path: str) -> dict:
        return torch.load(path, map_location="cpu", weights_only=False)


# --------------------------------------------------------------------------- #
# scheduler
# --------------------------------------------------------------------------- #
class WarmupScheduler:
    """LambdaLR-style: linear warmup, then constant or cosine decay."""

    def __init__(self, optimizer: torch.optim.Optimizer, warmup_steps: int,
                 total_steps: int, mode: str, step0: int = 0) -> None:
        self.optimizer = optimizer
        self.warmup_steps = max(1, warmup_steps)
        self.total_steps = total_steps
        self.mode = mode
        self.last_epoch = step0
        for g in optimizer.param_groups:
            g["lr"] = self._lr_at(self.last_epoch)

    def _lr_at(self, t: int) -> float:
        base = self.optimizer.defaults["lr"]
        if t < self.warmup_steps:
            return base * (t + 1) / self.warmup_steps
        if self.mode == "constant":
            return base
        if self.total_steps and self.warmup_steps < self.total_steps:
            progress = (t - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            return base * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return base

    def step(self) -> None:
        self.last_epoch += 1
        for g in self.optimizer.param_groups:
            g["lr"] = self._lr_at(self.last_epoch)

    def state_dict(self) -> dict:
        return {"last_epoch": self.last_epoch}

    def load_state_dict(self, state: dict) -> None:
        self.last_epoch = int(state.get("last_epoch", 0))
        for g in self.optimizer.param_groups:
            g["lr"] = self._lr_at(self.last_epoch)

    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]


# --------------------------------------------------------------------------- #
# trainer
# --------------------------------------------------------------------------- #
class Trainer:
    def __init__(self, args: argparse.Namespace, cfg: dict, paths: dict) -> None:
        self.args = args
        self.cfg = cfg
        self.cfg_ns = _as_namespace(cfg)
        self.paths = paths
        self.seed = int(cfg.get("seed", 42))
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype, self.use_scaler = _resolve_precision(cfg.get("precision", "fp32"), self.device)
        log.info("device=%s precision=%s (scaler=%s)", self.device,
                 cfg.get("precision"), self.use_scaler)

        # ---- model ----
        self.model = PhyloModel(
            build_encoder(self.cfg_ns, device=str(self.device)),
            d_emb=int(cfg.get("d_emb", 256)),
            max_dist=float(cfg.get("max_dist", 3.0)),
            head_type=str(cfg.get("head", "bilinear_mlp")),
        ).to(self.device)
        if cfg.get("grad_checkpointing", True):
            if hasattr(self.model.encoder.backbone, "gradient_checkpointing_enable"):
                self.model.encoder.backbone.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False})
                log.info("gradient checkpointing ENABLED (non-reentrant)")
            else:
                log.warning("gradient checkpointing requested but backbone lacks support")

        # ---- optimizer / scheduler ----
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(cfg.get("lr", 6e-4)),
            weight_decay=float(cfg.get("weight_decay", 0.1)),
            betas=tuple(cfg.get("betas", [0.9, 0.95])),
        )
        total_steps = int(cfg["max_steps"]) if cfg.get("max_steps") else 0
        self.scheduler = WarmupScheduler(
            self.optimizer,
            warmup_steps=int(cfg.get("warmup_steps", 0)),
            total_steps=total_steps,
            mode=str(cfg.get("scheduler", "constant")),
        )
        self.scaler = torch.amp.GradScaler(enabled=self.use_scaler)

        # ---- logging ----
        self.name = args.name or cfg.get("name")
        logs_dir = self.paths["local_data_dir"]
        if self.name:
            logs_dir = os.path.join(logs_dir, self.name)
        self.metrics_path = os.path.join(logs_dir, "metrics.csv")
        self.csv = CsvLogger(self.metrics_path)
        self.tb_writer: Any = None
        self.wandb_run: Any = None
        self._init_extra_logger()

        # ---- drive sync ----
        self.sync = DriveSync(
            ckpt_dir=self.paths["ckpt_dir"],
            local_ckpt_dir=self.paths["local_ckpt_dir"],
            results_dir=self.paths["results_dir"],
            local_metrics=self.metrics_path,
            enabled=not args.no_drive_sync,
        )
        self.checkpointer = Checkpointer(
            self.paths["local_ckpt_dir"],
            int(cfg.get("save_total_limit", 10)),
            float(cfg.get("max_dist", 3.0)),
        )

        # ---- data ----
        self.tokenizer = get_tokenizer()
        self.max_seq_len = int(cfg.get("max_seq_len", 32768))
        self.bucket_step = int(cfg.get("bucket_step", 512))
        self.batch_size = int(cfg.get("batch_size", 2))
        self.train_ds = load_dataset(
            "train", data_dir=self.paths["data_dir"], max_seq_len=self.max_seq_len,
            num_samples=cfg.get("num_train_alignments"), seed=self.seed,
            tokenizer=self.tokenizer,
        )
        self.val_ds = load_dataset(
            "val", data_dir=self.paths["data_dir"], max_seq_len=self.max_seq_len,
            num_samples=min(cfg.get("num_train_alignments") or 2**31, 64), seed=self.seed,
            tokenizer=self.tokenizer,
        )
        self.fixed_val_batch = self._fixed_val_batch()
        self.scale_factor = self._train_scale_factor()

        # ---- state ----
        self.epoch = 0
        self.step_in_epoch = 0
        self.global_step = 0
        self.best_val_mae = float("inf")
        self.best_path: str | None = None
        self._early_stop_counter = 0
        self._stop_requested = False
        self._resumed_from: tuple[str, int] | None = None

    def _train_scale_factor(self) -> float:
        """Median 'scale' column of the train parquet (global raw-unit factor).

        The model predicts NORMALIZED distances; inference multiplies by this
        factor to return raw substitution units. Defaults to 1.0.
        """
        try:
            import pyarrow.parquet as pq

            path = os.path.join(self.paths["data_dir"], "train.parquet")
            table = pq.read_table(path, columns=["scale"])
            scales = table.column(0).to_numpy()
            return float(np.median(scales)) if len(scales) else 1.0
        except Exception:
            log.warning("could not read train scale column; scale_factor=1.0", exc_info=True)
            return 1.0

    # -- logging ------------------------------------------------------------ #
    def _init_extra_logger(self) -> None:
        mode = str(self.cfg.get("log", "file"))
        if mode == "wandb":
            try:
                import wandb

                self.wandb_run = wandb.init(
                    project=str(self.cfg.get("wandb_project", "ssm-phylo")),
                    name=self.name, config=self.cfg, reinit=True,
                )
            except Exception as exc:  # noqa: BLE001 - logging must never kill training
                log.warning("wandb init failed: %s", exc)
        elif mode == "tensorboard":
            from torch.utils.tensorboard import SummaryWriter

            tb_dir = os.path.join(self.paths["local_data_dir"], "tb", self.name or "run")
            self.tb_writer = SummaryWriter(tb_dir)

    def _log_scalars(self, row: dict) -> None:
        self.csv.log(**row)
        if self.tb_writer is not None:
            for key in ("loss", "mae", "mae_norm", "mre", "fp_penalty", "lr", "val_loss", "val_mae"):
                if row.get(key) is not None:
                    self.tb_writer.add_scalar(f"train/{key}", row[key], self.global_step)
        if self.wandb_run is not None:
            self.wandb_run.log(row, step=self.global_step)

    # -- data ---------------------------------------------------------------- #
    def _collate(self, batch: list) -> tuple:
        return collate_with_bucketing(batch, self.max_seq_len, self.tokenizer.pad_id,
                                      bucket_step=self.bucket_step)

    def _fixed_val_batch(self) -> tuple | None:
        if len(self.val_ds) == 0:
            return None
        indices = list(range(min(self.batch_size, len(self.val_ds))))
        samples = [self.val_ds[i] for i in indices]
        return self._collate(samples)

    def _iter_batches(self):
        while True:
            idx = torch.randperm(len(self.train_ds), generator=torch.Generator().manual_seed(
                self.seed + self.global_step))
            for i in range(0, len(idx), self.batch_size):
                batch_idx = idx[i:i + self.batch_size].tolist()
                samples = [self.train_ds[j] for j in batch_idx]
                yield self._collate(samples)

    # -- core steps ---------------------------------------------------------- #
    def _forward(self, batch: tuple):
        tokens, spans, mask, dm, scales = [t.to(self.device) for t in batch]
        autocast = torch.autocast("cuda", dtype=self.dtype) if self.dtype else contextlib.nullcontext()
        with autocast:
            pred, _ = self.model(tokens, spans, spans_mask=mask.bool())
            losses = combined_loss(
                pred, dm, scales, loss_type=str(self.cfg.get("loss", "mae")),
                lambda_fp=float(self.cfg.get("four_point_lambda", 0.01)),
            )
        return losses, pred, dm, scales

    def train_step(self, batch: tuple) -> dict:
        from ssm_phylo.models.losses import mre_loss

        _t0 = time.time()
        losses, pred, dm, _scales = self._forward(batch)
        _t_fwd = time.time() - _t0
        loss = losses["loss"]
        if self.scaler.is_enabled():
            self.scaler.scale(loss).backward()
            if (self.global_step + 1) % int(self.cfg.get("grad_accum", 1)) == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               float(self.cfg.get("max_grad_norm", 1.0)))
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
        else:
            loss.backward()
            if (self.global_step + 1) % int(self.cfg.get("grad_accum", 1)) == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               float(self.cfg.get("max_grad_norm", 1.0)))
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()
        return {
            "loss": float(loss.detach()),
            "primary": float(losses["primary"].detach()),
            "mae": float(losses["mae_raw"].detach()),
            "mae_norm": float(losses["mae_norm"].detach()),
            "mre": float(mre_loss(pred.float(), dm).detach()),
            "fp_penalty": float(losses["four_point"].detach()),
            "t_fwd": _t_fwd,
            "t_bwd": time.time() - _t0,
        }

    def validate(self) -> dict:
        if self.fixed_val_batch is None:
            return {"val_loss": float("nan"), "val_mae": float("nan")}
        self.model.eval()
        with torch.no_grad():
            losses, _pred, _dm, _scales = self._forward(self.fixed_val_batch)
        self.model.train()
        return {"val_loss": float(losses["loss"]), "val_mae": float(losses["mae_raw"])}

    # -- checkpointing ------------------------------------------------------- #
    def save_checkpoint(self, val_metric: float | None = None) -> None:
        path = self.checkpointer.save(self, val_metric)
        log.info("saved checkpoint %s (step %d)", path, self.global_step)
        self.sync.push()

    def save_interrupted(self) -> None:
        log.info("saving emergency checkpoint ckpt-interrupted.pt (step %d)", self.global_step)
        self.checkpointer.save_as(self, "ckpt-interrupted.pt")

    def save_best(self, val_mae: float) -> None:
        path = self.checkpointer.save_as(self, "best.pt", val_metric=val_mae)
        self.best_path = path
        self.best_val_mae = val_mae
        log.info("new best val_mae=%.5f -> best.pt", val_mae)
        self.sync.push()

    # -- resume -------------------------------------------------------------- #
    def maybe_resume(self) -> None:
        resume = self.args.resume
        if resume == "none":
            return
        if not self.args.no_drive_sync:
            log.info("pulling checkpoints from Drive mirror (CKPT_DIR -> LOCAL_CKPT_DIR)")
            self.sync.pull()
        source = None
        if resume in ("latest", "best"):
            source = _scan_resume_source(resume, self.paths["local_ckpt_dir"])
        elif resume != "none":
            path = resume if os.path.isabs(resume) else os.path.join(self.paths["local_ckpt_dir"], resume)
            if os.path.exists(path):
                source = (path, -1)
        if source is None:
            log.info("no checkpoint to resume from (%s); starting fresh", resume)
            return
        path, _ = source
        ckpt = Checkpointer.load(path)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        if ckpt.get("scheduler_state") and self.scheduler:
            self.scheduler.load_state_dict(ckpt["scheduler_state"])
        _restore_rng(ckpt["rng_state"])
        self.epoch = int(ckpt.get("epoch", 0))
        self.step_in_epoch = int(ckpt.get("step", 0))
        self.global_step = int(ckpt.get("global_step", 0))
        self.best_val_mae = float(ckpt.get("val_metric") or self.best_val_mae)
        pulled = "" if self.args.no_drive_sync else " (pulled from Drive)"
        self._resumed_from = (path, self.global_step)
        print(f"[train] RESUMED from step {self.global_step}{pulled} ({path})", flush=True)

    # -- main loop ----------------------------------------------------------- #
    def run(self) -> int:
        self.maybe_resume()
        max_steps = int(self.cfg["max_steps"]) if self.cfg.get("max_steps") else 1 << 62
        max_epochs = int(self.cfg.get("max_epochs", 20))
        patience = int(self.cfg.get("early_stop_patience", 5))
        ceiling = float(self.cfg.get("hard_loss_ceiling", 100.0))
        if self._resumed_from:
            self._stop_requested = False
        batches = self._iter_batches()
        steps_per_epoch = max(1, math.ceil(len(self.train_ds) / self.batch_size))
        finite_steps = max_steps < (1 << 62)
        total_steps = max_steps if finite_steps else max_epochs * steps_per_epoch
        log.info("training start: %d samples, batch %d, max_epochs %d, up to %d steps%s",
                 len(self.train_ds), self.batch_size, max_epochs, total_steps,
                 "" if finite_steps else " (max_steps unset)")
        wall0 = time.time()
        t_step = wall0
        ema_ms: float | None = None
        epoch_loss_sum = 0.0
        epoch_loss_n = 0
        log_every = max(1, int(self.cfg.get("log_every", 10)))
        _warned_slow = False
        while self.epoch < max_epochs and self.global_step < max_steps:
            _t_coll0 = time.time()
            try:
                batch = next(batches)
            except StopIteration:
                break
            t_coll = time.time() - _t_coll0
            self.model.train()
            if self.args.smoke_step_delay:
                time.sleep(self.args.smoke_step_delay)
            if self.global_step == 0:
                log.info(
                    "step 1: collated in %.2fs (tokens %s); running forward+backward. "
                    "NOTE: the eager-Mamba fallback is sequential — at large sizes the "
                    "first step can take minutes; you WILL see step 1 when it completes.",
                    t_coll, tuple(batch[0].shape))
            row = self.train_step(batch)
            self.global_step += 1
            self.epoch = self.global_step // steps_per_epoch
            self.step_in_epoch = self.global_step % steps_per_epoch
            row["step"] = self.global_step
            row["epoch"] = self.epoch
            row["lr"] = self.scheduler.get_lr()
            row["t_coll"] = t_coll
            ms = (time.time() - t_step) * 1000.0
            t_step = time.time()
            ema_ms = ms if ema_ms is None else 0.95 * ema_ms + 0.05 * ms
            if not _warned_slow and row["t_fwd"] > 60.0:
                _warned_slow = True
                log.warning(
                    "forward pass took %.1fs (d_model=%s, stream len %d) — sequential "
                    "eager-Mamba is very slow at this size; training will be impractically "
                    "slow without fused kernels. Consider train_small or reducing max_seq_len.",
                    row["t_fwd"], self.cfg.get("d_model", "?"), batch[0].shape[1])
            epoch_loss_sum += row["loss"]
            epoch_loss_n += 1
            if self.global_step % steps_per_epoch == 0:
                log.info(
                    "epoch %d/%d complete: %d steps, avg loss=%.4f, best val_mae=%.5f",
                    self.epoch, max_epochs, steps_per_epoch,
                    epoch_loss_sum / max(epoch_loss_n, 1), self.best_val_mae)
                epoch_loss_sum = 0.0
                epoch_loss_n = 0
            if self.global_step == 1 or self.global_step % log_every == 0:
                frac = 100.0 * self.global_step / max(total_steps, 1)
                remaining = max(total_steps - self.global_step, 0)
                eta_s = (ema_ms / 1000.0) * remaining
                log.info(
                    "step %d/%d (epoch %d/%d, %.1f%%) loss=%.4f mae=%.4f mae_norm=%.4f "
                    "fp=%.4f lr=%.2e | %.0fms/step (fwd %.1fs bwd %.1fs coll %.2fs) ETA %dm%02ds",
                    self.global_step, total_steps, self.epoch, max_epochs, frac,
                    row["loss"], row["mae"], row["mae_norm"], row["fp_penalty"],
                    row["lr"], ema_ms, row["t_fwd"], row["t_bwd"], row["t_coll"],
                    int(eta_s // 60), int(eta_s % 60))
            if self.global_step % int(self.cfg.get("save_every", 250)) == 0:
                self.save_checkpoint()
            if self.global_step % int(self.cfg.get("val_every", 500)) == 0:
                log.info("step %d: validating (val_every=%d)...", self.global_step,
                         int(self.cfg.get("val_every", 500)))
                _tval = time.time()
                val = self.validate()
                row["val_loss"] = val["val_loss"]
                row["val_mae"] = val["val_mae"]
                log.info("step %d val_loss=%.4f val_mae=%.4f (%.1fs)", self.global_step,
                         val["val_loss"], val["val_mae"], time.time() - _tval)
                if val["val_mae"] < self.best_val_mae - 1e-9:
                    self.save_best(val["val_mae"])
                    self._early_stop_counter = 0
                else:
                    self._early_stop_counter += 1
                    log.info("val_mae not improved (%d/%d patience)", self._early_stop_counter, patience)
                    if self._early_stop_counter >= patience:
                        log.info("early stopping: no improvement for %d val checks", patience)
                        self._stop_requested = True
            self._log_scalars(row)
            if self._stop_requested:
                break
            if not math.isfinite(row["loss"]) or row["loss"] > ceiling:
                log.error("hard loss ceiling: loss=%.4f > %.4f; aborting", row["loss"], ceiling)
                self.save_interrupted()
                self.sync.push()
                self.sync.wait()
                return 1
            if self.global_step >= max_steps:
                log.info("reached max_steps=%d", max_steps)
        # final
        self.save_checkpoint()
        if self._stop_requested:
            log.info("early stopping exit")
        wall = time.time() - wall0
        log.info("training finished: %d steps in %.1fs (%.3fs/step)", self.global_step, wall,
                 wall / max(self.global_step, 1))
        self.sync.wait()
        return 0


_ACTIVE_TRAINER: Trainer | None = None


def _signal_handler(signum: int, _frame: Any) -> None:
    trainer = _ACTIVE_TRAINER
    print(f"[train] received signal {signum}; saving emergency checkpoint and pushing...", flush=True)
    if trainer is not None:
        try:
            trainer.save_interrupted()
            trainer.sync.push()
            trainer.sync.wait(timeout=300)
        except Exception as exc:  # noqa: BLE001 - never die inside a handler
            print(f"[train] emergency save failed: {exc}", flush=True)
    print("[train] clean exit after signal", flush=True)
    os._exit(0)


# --------------------------------------------------------------------------- #
# smoke / toy datasets
# --------------------------------------------------------------------------- #
def _make_tiny_dataset(local_data_dir: str, n_samples: int, seed: int = 42) -> str:
    """Simulate n tiny trees, consolidate + split into a local 'Drive' dir."""
    from ssm_phylo.data.simulation import simulate_alignments, simulate_trees

    raw = os.path.join(local_data_dir, "raw")
    os.makedirs(raw, exist_ok=True)
    stems = [f"t005_{i:06d}" for i in range(n_samples)]
    simulate_trees(5, n_samples, raw, seed=seed, stems=stems)
    simulate_alignments(raw, raw, length=30, seed=seed, engine="python", stems=stems)
    data_dir = os.path.join(local_data_dir, "drive")
    from ssm_phylo.data.simulation import make_splits

    make_splits(raw, data_dir, seed=seed, local_tmp_dir=os.path.join(local_data_dir, "ptmp"))
    return data_dir


def _tiny_cfg() -> dict:
    return {
        "d_model": 32, "n_layer": 2, "vocab_size": 38,
        "encoder": {"kind": "from_scratch", "checkpoint_dir": None,
                    "mamba": {"state_size": 4, "time_step_rank": 8, "conv_kernel": 3, "expand": 2},
                    "ptm_model_id": "x"},
        "d_emb": 16, "head": "bilinear_mlp", "max_dist": 3.0,
        "loss": "mae", "four_point_lambda": 0.01,
        "lr": 3e-3, "weight_decay": 0.1, "betas": [0.9, 0.95],
        "warmup_steps": 2, "scheduler": "constant",
        "max_seq_len": 512, "batch_size": 2, "grad_checkpointing": False,
        "precision": "fp32", "max_grad_norm": 1.0, "grad_accum": 1,
        "save_every": 1, "val_every": 2, "max_epochs": 100,
        "max_steps": 8, "early_stop_patience": 1000, "hard_loss_ceiling": 1000.0,
        "save_total_limit": 10, "bucket_step": 64, "seed": 42,
        "log_every": 1, "log": "file", "name": None,
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ssm_phylo.train",
        description="Train the SSM distance estimator (pull-train-push checkpointing).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", default=None, help="base YAML config (defaults merge over it)")
    p.add_argument("--smoke", action="store_true", help="tiny end-to-end run (>=3 steps, 64 samples)")
    p.add_argument("--toy", action="store_true", help="500-sample / 50-step loss-decrease check")
    p.add_argument("--smoke-step-delay", type=float, default=0.0, metavar="SEC",
                   help="sleep between steps (testing only)")
    p.add_argument("--max-seq-len", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--max-epochs", type=int, default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--precision", default=None, choices=["bf16", "fp16", "fp32"])
    p.add_argument("--save-every", type=int, default=None)
    p.add_argument("--val-every", type=int, default=None)
    p.add_argument("--warmup-steps", type=int, default=None)
    p.add_argument("--scheduler", default=None, choices=["constant", "cosine"])
    p.add_argument("--loss", default=None, choices=["mae", "mre"])
    p.add_argument("--lambda-fp", type=float, default=None)
    p.add_argument("--grad-checkpointing", dest="grad_checkpointing",
                   action=argparse.BooleanOptionalAction, default=None,
                   help="use gradient checkpointing (default from config: on)")
    p.add_argument("--resume", default="none", metavar="{none,latest,best,path}",
                   help="resume mode; latest scans numbered + ckpt-interrupted")
    p.add_argument("--data-dir", default=None, help="parquet dir (default $DATA_DIR)")
    p.add_argument("--ckpt-dir", default=None, help="Drive mirror dir (default $CKPT_DIR)")
    p.add_argument("--results-dir", default=None, help="Drive results dir (default $RESULTS_DIR)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--log", default=None, choices=["wandb", "tensorboard", "file", "none"])
    p.add_argument("--name", default=None, help="run name (wandb + logs subdir)")
    p.add_argument("--no-drive-sync", action="store_true",
                   help="run entirely locally (no pull/push; for dev machines)")
    p.add_argument("--log-every", type=int, default=None)
    p.add_argument("--max-grad-norm", type=float, default=None)
    p.add_argument("--grad-accum", type=int, default=None)
    p.add_argument("--save-total-limit", type=int, default=None)
    p.add_argument("--bucket-step", type=int, default=None)
    p.add_argument("--early-stop-patience", type=int, default=None)
    p.add_argument("--hard-loss-ceiling", type=float, default=None)
    p.add_argument("--num-train-alignments", type=int, default=None)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    args = build_parser().parse_args(argv)

    if args.smoke or args.toy:
        n_samples = 64 if args.smoke else 500
        cfg = _tiny_cfg()
        if args.toy:
            cfg["max_steps"] = 50
            cfg["batch_size"] = 8
            cfg["save_every"] = 10
            cfg["val_every"] = 25
        cfg = _deep_merge(cfg, _flatten_overrides(args))
        local_data = os.path.abspath(args.data_dir or _env("LOCAL_DATA_DIR")
                                     or os.path.join(tempfile.gettempdir(), "ssm_phylo_smoke"))
        os.makedirs(local_data, exist_ok=True)
        data_dir = _make_tiny_dataset(local_data, n_samples, seed=args.seed or 42)
        cfg["data_dir"] = data_dir
        paths = _resolve_paths(cfg, args)
        paths["data_dir"] = data_dir
        paths["local_data_dir"] = local_data
    else:
        if not args.config:
            raise SystemExit("pass --config (or use --smoke/--toy)")
        base = _load_config(os.path.join(REPO_ROOT, "configs", "default.yaml"))
        user = _load_config(args.config)
        cfg = _deep_merge(base, user)
        overrides = _flatten_overrides(args)
        cfg = _deep_merge(cfg, overrides)
        paths = _resolve_paths(cfg, args)

    for d in ("local_ckpt_dir", "local_data_dir", "ckpt_dir"):
        os.makedirs(paths[d], exist_ok=True)
    if paths["results_dir"]:
        os.makedirs(paths["results_dir"], exist_ok=True)

    trainer = Trainer(args, cfg, paths)
    global _ACTIVE_TRAINER
    _ACTIVE_TRAINER = trainer
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    if args.smoke or args.toy:
        _memory_probe(trainer, args)
    rc = trainer.run()
    _ACTIVE_TRAINER = None
    if trainer.wandb_run is not None:
        trainer.wandb_run.finish()
    if trainer.tb_writer is not None:
        trainer.tb_writer.close()
    return rc


def _memory_probe(trainer: Trainer, args: argparse.Namespace) -> None:
    """Compare peak memory with/without grad checkpointing (smoke, CUDA only)."""
    if trainer.device.type != "cuda":
        print("[smoke] memory comparison skipped (no CUDA in this torch build)")
        return
    batch = next(iter(trainer._iter_batches()))
    for label, enable in (("grad-checkpointing OFF", False), ("grad-checkpointing ON", True)):
        torch.cuda.reset_peak_memory_stats()
        backbone = trainer.model.encoder.backbone
        if hasattr(backbone, "gradient_checkpointing_enable"):
            if enable:
                backbone.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False})
            elif hasattr(backbone, "gradient_checkpointing_disable"):
                backbone.gradient_checkpointing_disable()
        trainer.model.zero_grad()
        losses, *_ = trainer._forward(batch)
        losses["loss"].backward()
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"[smoke] {label}: peak CUDA memory {peak:.2f} GB")
    if hasattr(backbone, "gradient_checkpointing_enable"):
        backbone.gradient_checkpointing_enable()


if __name__ == "__main__":
    raise SystemExit(main())

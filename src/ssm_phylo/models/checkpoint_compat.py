"""checkpoint_compat — shape-inference loader for the ProtMamba v1.0 release.

FORENSIC ANALYSIS of `Bitbol-Lab/ProtMamba-ssm` release v1.0
(`ProtMamba_model-weights.zip`, downloaded by scripts/download_weights.sh):

config.json (from the release) claims:  d_model=1024, n_layer=16, vocab_size=38,
max_position_embeddings=2048.

pytorch_model.bin ACTUAL shapes:

    backbone.embedding.weight             (40, 512)      # vocab 40, dim 512
    backbone.position_embedding.weight   (2048, 512)     # custom pos-embed module
    lm_head.weight                        (40, 512)      # vocab 40, dim 512
    backbone.layers.{0-15}.norm.weight   (1024,)         # d_model 1024
    ...mixer.ckpt_layer.in_proj.weight   (4096, 1024)    # d_inner 2048, d_model 1024
    ...mixer.ckpt_layer.out_proj.weight  (1024, 2048)
    ...mixer.ckpt_layer.conv1d.weight    (2048, 1, 4)    # conv_kernel 4
    ...mixer.ckpt_layer.x_proj.weight     (96, 2048)     # dt_rank 64 + 2*state_size 16
    ...mixer.ckpt_layer.dt_proj.weight   (2048, 64)      # dt_rank 64
    ...mixer.ckpt_layer.A_log            (2048, 16)      # state_size 16
    ...mixer.ckpt_layer.D                 (2048,)
    backbone.norm_f.weight               (1024,)

CONCLUSION: the token embedding and LM head are 512-dimensional while the
entire 16-layer backbone is 1024-dimensional. NO coherent architecture — not
even the authors' own MambaLMHeadModelwithPosids at the documented config —
can load this checkpoint. It also disagrees with config.json on vocab
(40 vs 38). The release additionally bundles training artifacts
(optimizer.pt, scheduler.pt, rng_state.pth, trainer_state.json at
epoch 2.09 / global_step 8750), i.e. it is an intermediate training snapshot
with a resized/re-scoped embedding. See AGENTS.md "Known issue".

WHAT THIS MODULE DOES (degraded_protmamba mode):
1. infers backbone hyperparameters from state-dict SHAPES ONLY — never from
   config.json, which lies;
2. remaps mamba-ssm checkpoint_mixer naming `mixer.ckpt_layer.*` -> `mixer.*`;
3. validates every backbone tensor shape against the caller's config and
   fails LOUDLY (ProtMambaWeightsError) on any backbone mismatch;
4. drops the mismatched embedding / lm_head / position_embedding (they cannot
   load into any coherent model) and re-initializes a fresh embedding at
   (vocab_size, d_model);
5. gates the result on a finite-loss forward pass.

The resulting model is a fine-tuning base: real 1024-dim ProtMamba backbone
weights + randomly initialized embedding (Phase 3 fine-tunes the encoder).
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

import torch

log = logging.getLogger(__name__)

# ProtMamba release v1.0 facts (kept for the loader contract and re-check tooling)
RELEASE_EMBEDDING_SHAPE = (40, 512)  # mismatched; dropped on load
RELEASE_BACKBONE_D_MODEL = 1024


class ProtMambaWeightsError(RuntimeError):
    """Raised when ProtMamba weights are missing or structurally unusable."""


@dataclass
class LoadReport:
    """What a degraded load did with the checkpoint."""

    total_keys: int
    loaded_keys: int
    skipped_shape_mismatch: int
    dropped_special: int
    checkpoint_shapes: dict[str, int]
    embedding_reinitialized: bool = True

    def summary(self) -> str:
        return (
            f"loaded {self.loaded_keys}/{self.total_keys} keys "
            f"({self.skipped_shape_mismatch} shape-mismatched skipped, "
            f"{self.dropped_special} special tensors dropped); "
            f"checkpoint shapes: {self.checkpoint_shapes}"
        )


# --------------------------------------------------------------------------- #
# key handling
# --------------------------------------------------------------------------- #
def remap_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """mamba-ssm checkpoint_mixer naming -> transformers MambaForCausalLM naming.

    `backbone.layers.{i}.mixer.ckpt_layer.{param}` -> `backbone.layers.{i}.mixer.{param}`
    (other keys pass through unchanged).
    """
    out: dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        out[k.replace("mixer.ckpt_layer.", "mixer.")] = v
    return out


def infer_checkpoint_shapes(state_dict: dict[str, torch.Tensor]) -> dict[str, int]:
    """Infer backbone hyperparameters from tensor shapes — config.json is ignored.

    Requires remapped keys (see remap_keys). Raises ProtMambaWeightsError if
    the checkpoint has no recognizable backbone structure.
    """
    mixers = {
        int(k.split(".")[2]): k
        for k in state_dict
        if k.startswith("backbone.layers.") and ".mixer." in k
    }
    if not mixers:
        raise ProtMambaWeightsError(
            "no backbone.layers.*.mixer.* tensors found in checkpoint"
        )
    n_layer = max(mixers) + 1
    any_layer = f"backbone.layers.{min(mixers)}"

    def shape(key: str) -> tuple[int, ...]:
        if key not in state_dict:
            raise ProtMambaWeightsError(f"missing expected tensor {key}")
        return tuple(state_dict[key].shape)

    d_model = shape("backbone.norm_f.weight")[0]
    d_inner = shape(f"{any_layer}.mixer.out_proj.weight")[0]
    conv_kernel = shape(f"{any_layer}.mixer.conv1d.weight")[2]
    state_size = shape(f"{any_layer}.mixer.A_log")[1]
    dt_rank = shape(f"{any_layer}.mixer.dt_proj.weight")[1]
    expand = d_inner // d_model
    vocab = shape("backbone.embedding.weight")[0]
    return {
        "d_model": d_model,
        "n_layer": n_layer,
        "d_inner": d_inner,
        "state_size": state_size,
        "dt_rank": dt_rank,
        "conv_kernel": conv_kernel,
        "expand": expand,
        "vocab": vocab,
    }


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def validate_backbone(
    state_dict: dict[str, torch.Tensor], shapes: dict[str, int], cfg: Any
) -> None:
    """Check every backbone tensor shape against the target config.

    Raises ProtMambaWeightsError with a clear message on ANY mismatch —
    the caller should never load a structurally different backbone silently.
    """
    encoder_cfg = _get(cfg, "encoder", {}) or {}
    mamba_cfg = _get(encoder_cfg, "mamba", {}) or {}
    d_model = int(cfg.d_model)
    n_layer = int(cfg.n_layer)
    d_inner = d_model * int(_get(mamba_cfg, "expand", 2))
    state_size = int(_get(mamba_cfg, "state_size", 16))
    dt_rank = int(_get(mamba_cfg, "time_step_rank", 64))
    conv_kernel = int(_get(mamba_cfg, "conv_kernel", 4))

    if shapes["d_model"] != d_model:
        raise ProtMambaWeightsError(
            f"backbone d_model mismatch: checkpoint has {shapes['d_model']}, "
            f"config expects {d_model}. The ProtMamba v1.0 release weights are "
            "known to be internally inconsistent (AGENTS.md); if this is a new "
            "release, re-inspect it before trusting it."
        )
    if shapes["n_layer"] != n_layer:
        raise ProtMambaWeightsError(
            f"backbone n_layer mismatch: checkpoint {shapes['n_layer']} vs config {n_layer}"
        )
    expected: dict[str, tuple[int, ...]] = {
        "backbone.norm_f.weight": (d_model,),
        # NOTE: backbone.embedding.weight is intentionally NOT validated: the
        # ProtMamba v1.0 release embedding (40, 512) mismatches the backbone
        # (1024) by design of the broken release; it is dropped and
        # re-initialized by the caller.
    }
    for i in range(n_layer):
        p = f"backbone.layers.{i}."
        expected.update(
            {
                f"{p}norm.weight": (d_model,),
                f"{p}mixer.in_proj.weight": (2 * d_inner, d_model),
                f"{p}mixer.conv1d.weight": (d_inner, 1, conv_kernel),
                f"{p}mixer.conv1d.bias": (d_inner,),
                f"{p}mixer.x_proj.weight": (dt_rank + 2 * state_size, d_inner),
                f"{p}mixer.dt_proj.weight": (d_inner, dt_rank),
                f"{p}mixer.dt_proj.bias": (d_inner,),
                f"{p}mixer.out_proj.weight": (d_model, d_inner),
                f"{p}mixer.A_log": (d_inner, state_size),
                f"{p}mixer.D": (d_inner,),
            }
        )
    mismatches = [
        (k, actual, want)
        for k, want in expected.items()
        if (actual := tuple(state_dict.get(k, torch.empty(0)).shape)) != want
    ]
    if mismatches:
        detail = "; ".join(
            f"{k}: got {a}, want {w}" for k, a, w in mismatches[:5]
        )
        raise ProtMambaWeightsError(
            f"backbone shape validation failed for {len(mismatches)} tensor(s): {detail} "
            "— refusing to load a structurally mismatched backbone. If the "
            "ProtMamba release changed, re-inspect it; do not silently load."
        )


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def _compatible_state_dict(
    model: torch.nn.Module, remapped: dict[str, torch.Tensor]
) -> tuple[dict[str, torch.Tensor], int, int]:
    """Keep only keys present in the model with matching shapes."""
    model_sd = model.state_dict()
    compatible: dict[str, torch.Tensor] = {}
    n_skip = 0
    for k, v in remapped.items():
        if k not in model_sd:
            continue
        if v.shape != model_sd[k].shape:
            n_skip += 1
            continue
        compatible[k] = v
    return compatible, n_skip, len(remapped)


def load_degraded_backbone(
    model: torch.nn.Module, checkpoint_dir: str, cfg: Any
) -> LoadReport:
    """Load the compatible ProtMamba backbone into an HF MambaForCausalLM.

    Drops the mismatched embedding/lm_head/position_embedding and leaves the
    (already randomly initialized) model embedding untouched; caller re-inits
    it explicitly via reinit_embedding. See module docstring for the forensic
    context.
    """
    ckpt_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
    if not os.path.exists(ckpt_path):
        raise ProtMambaWeightsError(
            f"no weights at {ckpt_path}. Download them first:\n"
            "    bash scripts/download_weights.sh"
        )
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    remapped = remap_keys(state_dict)
    shapes = infer_checkpoint_shapes(remapped)
    validate_backbone(remapped, shapes, cfg)
    if shapes["vocab"] != int(cfg.vocab_size):
        log.warning(
            "checkpoint vocab=%d differs from config vocab=%d; the embedding is "
            "dropped and re-initialized anyway (expected for ProtMamba v1.0)",
            shapes["vocab"],
            int(cfg.vocab_size),
        )
    compatible, n_skip, total = _compatible_state_dict(model, remapped)
    special = [k for k in remapped if not k.startswith("backbone.layers.")]
    model.load_state_dict(compatible, strict=False)
    report = LoadReport(
        total_keys=total,
        loaded_keys=len(compatible),
        skipped_shape_mismatch=n_skip,
        dropped_special=len(special),
        checkpoint_shapes=shapes,
    )
    log.info("ProtMamba degraded load: %s", report.summary())
    return report


def reinit_embedding(model: torch.nn.Module, d_model: int, vocab_size: int) -> None:
    """Re-initialize the token embedding at (vocab_size, d_model)."""
    emb: torch.nn.Embedding = model.backbone.embeddings
    new_emb = torch.nn.Embedding(vocab_size, d_model)
    with torch.no_grad():
        emb.weight.data = new_emb.weight.data.clone()
        if hasattr(emb, "padding_idx") and emb.padding_idx is not None:
            emb.weight.data[emb.padding_idx].zero_()
    log.info("embedding re-initialized at (%d, %d)", vocab_size, d_model)


def gate_finite_forward(model: torch.nn.Module, vocab_size: int) -> None:
    """Fail loudly if a smoke forward produces non-finite outputs (backbone gate)."""
    model.eval()
    tokens = torch.randint(0, max(vocab_size, 2), (1, 64), dtype=torch.long)
    with torch.no_grad():
        try:
            out = model(input_ids=tokens, use_cache=False)
            logits = out.logits if hasattr(out, "logits") else out[0]
            loss = torch.nn.functional.cross_entropy(
                logits[0].reshape(-1, logits.shape[-1]), tokens[0]
            )
        except Exception as exc:
            raise ProtMambaWeightsError(
                f"degraded ProtMamba backbone failed its smoke forward: {exc}"
            ) from exc
    if not torch.isfinite(logits).all() or not torch.isfinite(loss):
        raise ProtMambaWeightsError(
            "degraded ProtMamba backbone produced non-finite outputs in the "
            "smoke forward — weights are structurally incompatible."
        )
    log.info("finite-loss gate passed (loss=%.4f)", float(loss))

"""Config-driven encoder construction + a backend-agnostic encoder wrapper.

build_encoder(cfg) supports three kinds (see configs/default.yaml `encoder:`):
- from_scratch:     HF transformers MambaForCausalLM (eager, random weights)
                    at cfg d_model/n_layer/vocab_size. License-clean, CI-safe,
                    needs no weights. DEFAULT.
- degraded_protmamba: ProtMamba v1.0 backbone loaded via checkpoint_compat.py
                    (shape-inferred, ckpt_layer remap, embedding re-initialized,
                    finite-forward gate). Requires $PROT_MAMBA_CKPT.
- ptm_mamba:        DORMANT. No public weights exist for ChatterjeeLab/PTM-Mamba
                    (code-only repo, cc-by-nc-nd-4.0, whose ND clause forbids
                    derivatives); raises a clear "not available" error.
                    Time-boxed wrapper: raises a clear "not available" error
                    unless SSM_PHYLO_PTM_MAMBA_DIR points at a local checkout
                    that loads cleanly.

ProtMambaEncoder wraps ANY backbone (the design is encoder-agnostic): hidden
states are extracted with forward hooks on the backbone's layer stack, so no
dependency on mamba-ssm APIs or backend-specific kwargs. Position ids
(ProtMamba "1d" scheme) are forwarded only to backbones that support them.
"""
from __future__ import annotations

import logging
import os
import threading
from typing import Any

import torch
from torch import nn

log = logging.getLogger(__name__)

DOWNLOAD_HINT = (
    "bash scripts/download_weights.sh   # -> $PROT_MAMBA_CKPT (then set PROT_MAMBA_CKPT)"
)

DEFAULT_MAMBA = {"state_size": 16, "time_step_rank": 64, "conv_kernel": 4, "expand": 2}


# --------------------------------------------------------------------------- #
# helpers: config access (dict or SimpleNamespace-style objects)
# --------------------------------------------------------------------------- #
def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _encoder_section(cfg: Any) -> dict:
    section = _get(cfg, "encoder", {}) or {}
    if isinstance(section, dict):
        return section
    return {k: getattr(section, k) for k in dir(section) if not k.startswith("_")}


def _mamba_kwargs(cfg: Any) -> dict:
    section = _encoder_section(cfg)
    mamba = _get(section, "mamba", {}) or {}
    if not isinstance(mamba, dict):
        mamba = {k: getattr(mamba, k) for k in dir(mamba) if not k.startswith("_")}
    kw = dict(DEFAULT_MAMBA)
    kw.update({k: v for k, v in mamba.items() if v is not None})
    return kw


# --------------------------------------------------------------------------- #
# backbone builders
# --------------------------------------------------------------------------- #
def _mamba_config(cfg: Any) -> Any:
    from transformers import MambaConfig

    kw = _mamba_kwargs(cfg)
    return MambaConfig(
        vocab_size=int(_get(cfg, "vocab_size")),
        hidden_size=int(_get(cfg, "d_model")),
        num_hidden_layers=int(_get(cfg, "n_layer")),
        state_size=int(kw["state_size"]),
        time_step_rank=int(kw["time_step_rank"]),
        conv_kernel=int(kw["conv_kernel"]),
        expand=int(kw["expand"]),
        tie_word_embeddings=False,
    )


def _build_from_scratch(cfg: Any, device: str | None) -> nn.Module:
    """HF eager MambaForCausalLM with random weights (CI-safe, license-clean)."""
    from transformers import MambaForCausalLM

    model = MambaForCausalLM(_mamba_config(cfg))
    if device:
        model = model.to(device)
    log.info("encoder kind=from_scratch: MambaForCausalLM %s", _mamba_config(cfg))
    return model


def _build_degraded_protmamba(
    cfg: Any, checkpoint_dir: str | None, device: str | None
) -> nn.Module:
    """ProtMamba v1.0 backbone: compatible weights + re-init embedding + gate."""
    from transformers import MambaForCausalLM

    from ssm_phylo.models import checkpoint_compat as cc

    ckpt_dir = (
        checkpoint_dir
        or _get(_encoder_section(cfg), "checkpoint_dir")
        or os.environ.get("PROT_MAMBA_CKPT")
    )
    if not ckpt_dir or not os.path.isdir(ckpt_dir):
        raise cc.ProtMambaWeightsError(
            f"degraded_protmamba needs the ProtMamba checkpoint directory "
            f"(got {ckpt_dir!r}). Download it first:\n    {DOWNLOAD_HINT}"
        )
    model = MambaForCausalLM(_mamba_config(cfg))
    cc.load_degraded_backbone(model, ckpt_dir, cfg)
    cc.reinit_embedding(model, int(_get(cfg, "d_model")), int(_get(cfg, "vocab_size")))
    cc.gate_finite_forward(model, int(_get(cfg, "vocab_size")))
    if device:
        model = model.to(device)
    return model


def _timebox(seconds: float, fn, *args, **kwargs):
    """Run fn with a wall-clock deadline; raise TimeoutError if it overruns."""
    result: dict = {}
    err: dict = {}

    def runner() -> None:
        try:
            result["value"] = fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - re-raised by the caller
            err["value"] = exc

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join(timeout=seconds)
    if thread.is_alive():
        raise TimeoutError(f"ptm_mamba load exceeded {seconds}s wall-clock budget")
    if "value" in result:
        return result["value"]
    raise err["value"]


def _build_ptm_mamba(cfg: Any, device: str | None) -> nn.Module:
    """DORMANT: bidirectional gated Mamba + ESM-2 (PTM-Mamba), time-boxed.

    No public weights exist for the requested id and the real PTM-Mamba
    weights are cc-by-nc-nd-4.0 (no derivatives — fine-tuning or representation
    extraction could count as derivative work, which would contaminate a
    commercial release). This mode only works with a local checkout via
    SSM_PHYLO_PTM_MAMBA_DIR and raises a clear "not available" error otherwise.
    """
    model_id = _get(_encoder_section(cfg), "ptm_model_id", "ChatterjeeLab/PTM-Mamba")
    local = os.environ.get("SSM_PHYLO_PTM_MAMBA_DIR")
    budget = float(os.environ.get("SSM_PHYLO_PTM_MAMBA_TIMEOUT", "120"))
    try:
        if local and os.path.isdir(local):
            model = _timebox(
                budget,
                _load_ptm_mamba_local,
                local,
                int(_get(cfg, "d_model")),
                int(_get(cfg, "n_layer")),
                int(_get(cfg, "vocab_size")),
            )
        else:
            model = _timebox(
                budget, _load_ptm_mamba_hf, model_id,
                int(_get(cfg, "d_model")), int(_get(cfg, "n_layer")), int(_get(cfg, "vocab_size")),
            )
    except Exception as exc:
        raise RuntimeError(
            f"ptm_mamba not available: {exc} "
            f"(no public weights for '{model_id}'; real PTM-Mamba is "
            "cc-by-nc-nd-4.0; set SSM_PHYLO_PTM_MAMBA_DIR to a local checkout)"
        ) from exc
    if device:
        model = model.to(device)
    return model


def _load_ptm_mamba_hf(model_id: str, d_model: int, n_layer: int, vocab_size: int) -> nn.Module:
    from transformers import AutoConfig, AutoModel

    config = AutoConfig.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id, config=config)
    _validate_ptm_shape(model, d_model, n_layer, vocab_size)
    return model


def _load_ptm_mamba_local(path: str, d_model: int, n_layer: int, vocab_size: int) -> nn.Module:
    """Load from a local checkout; the exact loader depends on the checkout layout."""
    import importlib.util

    for candidate in (
        os.path.join(path, "protein_lm", "modeling", "models", "ptm_mamba.py"),
        os.path.join(path, "ptm_mamba.py"),
        os.path.join(path, "modeling.py"),
    ):
        if os.path.exists(candidate):
            spec = importlib.util.spec_from_file_location("ptm_mamba_local", candidate)
            if spec is None or spec.loader is None:
                raise FileNotFoundError(f"cannot load {candidate}")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            for name in ("load_pretrained", "PTMMamba", "create_model", "load_model"):
                fn = getattr(mod, name, None)
                if callable(fn):
                    return _validate_ptm_shape(fn(), d_model, n_layer, vocab_size)
    raise FileNotFoundError("no recognizable model loader in the local checkout")


def _validate_ptm_shape(model: nn.Module, d_model: int, n_layer: int, vocab_size: int) -> nn.Module:
    cfg = getattr(model, "config", None)
    if cfg is not None:
        got = {
            "d_model": getattr(cfg, "hidden_size", getattr(cfg, "d_model", None)),
            "n_layer": getattr(cfg, "num_hidden_layers", getattr(cfg, "n_layer", None)),
            "vocab": getattr(cfg, "vocab_size", None),
        }
        want = {"d_model": d_model, "n_layer": n_layer, "vocab": vocab_size}
        if any(got[k] is not None and got[k] != want[k] for k in want):
            raise ValueError(
                f"ptm_mamba shape mismatch: got {got}, want {want}"
            )
    return model


# --------------------------------------------------------------------------- #
# position ids ("1d" scheme, pure function)
# --------------------------------------------------------------------------- #
def make_position_ids(
    seq_spans: torch.Tensor, max_pos: int = 2048, max_seq_pos: int = 512
) -> tuple[torch.Tensor, torch.Tensor]:
    """ProtMamba "1d" position ids for a padded span tensor.

    Args:
        seq_spans: (B, N, 2) long tensor of (start, end) token spans, or (N, 2)
            for a single sample.
        max_pos: clip cap for per-token stream positions (1-indexed).
        max_seq_pos: clip cap for per-sequence indices (0-indexed).

    Returns:
        (position_ids, seq_position_ids), each (B, L) long, where L is the
        largest span end in the batch; padding positions (beyond a sample's
        stream) are 0.
    """
    if seq_spans.ndim == 2:
        seq_spans = seq_spans.unsqueeze(0)
    B, N, _ = seq_spans.shape
    device = seq_spans.device
    L = int(seq_spans[:, :, 1].max())
    position_ids = torch.arange(1, L + 1, device=device).repeat(B, 1)
    seq_position_ids = torch.zeros(B, L, dtype=torch.long, device=device)
    for b in range(B):
        for k in range(N):
            start = int(seq_spans[b, k, 0])
            end = min(int(seq_spans[b, k, 1]), L)
            if end > start:
                seq_position_ids[b, start:end] = k
    position_ids = position_ids.clamp(max=max_pos)
    seq_position_ids = seq_position_ids.clamp(max=max_seq_pos)
    return position_ids, seq_position_ids


# --------------------------------------------------------------------------- #
# backend-agnostic wrapper
# --------------------------------------------------------------------------- #
class ProtMambaEncoder(nn.Module):
    """Wraps any Mamba-style backbone; extracts last-layer hidden states.

    Hidden states come from forward hooks on the backbone's layer stack, so
    the wrapper works identically for transformers MambaForCausalLM, the
    degraded ProtMamba build, and a hypothetical PTM-Mamba wrapper — no
    mamba-ssm dependency.
    """

    def __init__(self, backbone: nn.Module, supports_positions: bool | None = None) -> None:
        super().__init__()
        self.backbone = backbone
        if supports_positions is None:
            name = type(backbone).__name__
            self.supports_positions = bool(
                hasattr(backbone, "position_embedding")
                or "Posids" in name
                or "PTM" in name
            )
        else:
            self.supports_positions = supports_positions
        self.backbone.eval()  # eval mode by default; .train() re-enables training
        self._hidden_path: str | None = None  # "output_hidden_states" | "hooks"

    @property
    def d_model(self) -> int:
        cfg = getattr(self.backbone, "config", None)
        if cfg is not None:
            for attr in ("hidden_size", "d_model"):
                v = getattr(cfg, attr, None)
                if v:
                    return int(v)
        state = self.backbone.state_dict()
        for key in ("backbone.norm_f.weight", "norm_f.weight"):
            if key in state:
                return int(state[key].shape[0])
        raise RuntimeError("cannot determine d_model for the wrapped backbone")

    def freeze(self, frozen: bool) -> None:
        """Toggle requires_grad on all backbone parameters."""
        for p in self.backbone.parameters():
            p.requires_grad = not frozen

    def _layer_stack(self) -> nn.ModuleList | None:
        backbone = getattr(self.backbone, "backbone", None)
        layers = getattr(backbone, "layers", None) if backbone is not None else None
        if isinstance(layers, nn.ModuleList) and len(layers):
            return layers
        return None

    def forward(
        self,
        tokens: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        seq_position_ids: torch.Tensor | None = None,
        layer: int = -1,
    ) -> torch.Tensor:
        """Return the chosen layer's per-token hidden states (B, L, d_model).

        Two extraction paths:
        - "output_hidden_states": transformers-style backbones (e.g.
          MambaForCausalLM) return per-layer hidden states natively. NO forward
          hooks — hooks defeat torch.utils.checkpoint's reentrant mode and
          destroy the gradient-checkpointing memory win.
        - "hooks": fallback for custom backbones (ProtMamba / PTM wrappers)
          that lack output_hidden_states; last-layer hook capture.
        """
        kwargs: dict[str, Any] = {"input_ids": tokens, "use_cache": False}
        if self.supports_positions and position_ids is not None:
            kwargs["position_ids"] = position_ids
            if seq_position_ids is not None:
                kwargs["seq_position_ids"] = seq_position_ids
        if self._hidden_path is None:
            self._hidden_path = self._probe_hidden_path(**kwargs)
        if self._hidden_path == "output_hidden_states":
            kwargs["output_hidden_states"] = True
            out = self.backbone(**kwargs)
            hs = out.hidden_states
            idx = layer if layer >= 0 else len(hs) + layer
            return hs[idx]
        return self._forward_with_hooks(tokens, layer, kwargs)

    def _probe_hidden_path(self, **kwargs: Any) -> str:
        """Discover the native hidden-states mechanism with one tiny forward."""
        probe = dict(kwargs)
        probe["input_ids"] = kwargs["input_ids"][:, : min(kwargs["input_ids"].shape[1], 8)]
        probe["output_hidden_states"] = True
        try:
            out = self.backbone(**probe)
            if getattr(out, "hidden_states", None) is not None:
                return "output_hidden_states"
        except TypeError:
            pass
        return "hooks"

    def _forward_with_hooks(
        self, tokens: torch.Tensor, layer: int, kwargs: dict[str, Any]
    ) -> torch.Tensor:
        layers = self._layer_stack()
        captured: dict[str, torch.Tensor] = {}
        hooks: list[Any] = []
        if layers is not None:
            idx = layer if layer >= 0 else len(layers) + layer
            hook = layers[idx].register_forward_hook(
                lambda _m, _i, o: captured.update(hidden=o[0] if isinstance(o, tuple) else o)
            )
            hooks.append(hook)
        try:
            out = self.backbone(**kwargs)
        finally:
            for h in hooks:
                h.remove()
        if captured:
            return captured["hidden"]
        hidden = getattr(out, "hidden_states", None)
        if hidden is not None:
            idx = layer + 1 if layer >= 0 else len(hidden) + layer
            return hidden[idx]
        raise RuntimeError(
            f"could not extract hidden states from backbone {type(self.backbone).__name__}"
        )


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def build_encoder(
    cfg: Any, checkpoint_dir: str | None = None, device: str | None = None
) -> ProtMambaEncoder:
    """Build the encoder per cfg.encoder.kind (from_scratch default)."""
    kind = _get(_encoder_section(cfg), "kind", "from_scratch")
    if kind == "from_scratch":
        backbone = _build_from_scratch(cfg, device)
    elif kind == "degraded_protmamba":
        backbone = _build_degraded_protmamba(cfg, checkpoint_dir, device)
    elif kind == "ptm_mamba":
        backbone = _build_ptm_mamba(cfg, device)
    else:
        raise ValueError(f"unknown encoder.kind '{kind}'")
    return ProtMambaEncoder(backbone)

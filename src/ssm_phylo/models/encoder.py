"""From-scratch Mamba encoder construction + a backend-agnostic wrapper.

build_encoder(cfg, device=None) ALWAYS builds the from-scratch transformers
MambaForCausalLM at cfg d_model/n_layer/vocab_size (random weights —
license-clean, CI-safe, no external weights). `kernels` (HuggingFace's fused
kernels, fast wheels) is an OPTIONAL speedup auto-detected by transformers
5.x; the eager Mamba path works everywhere and the old fused-kernel package
is obsolete.

MambaEncoder wraps the backbone: hidden states are extracted via the
backbone's native output_hidden_states path (no forward hooks — hooks defeat
torch.utils.checkpoint's reentrant mode and destroy the gradient-checkpointing
memory win), with a hook-based fallback for backbones lacking the native
mechanism. Position ids (the "1d" scheme) are forwarded only to backbones
that support them.
"""
from __future__ import annotations

import logging
from typing import Any

import torch
from torch import nn

log = logging.getLogger(__name__)

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
        use_mambapy=bool(_get(kw, "use_mambapy", False)),
        tie_word_embeddings=False,
    )


# --------------------------------------------------------------------------- #
# position ids ("1d" scheme, pure function)
# --------------------------------------------------------------------------- #
def make_position_ids(
    seq_spans: torch.Tensor, max_pos: int = 2048, max_seq_pos: int = 512
) -> tuple[torch.Tensor, torch.Tensor]:
    """Position ids for a padded span tensor (per-token stream positions).

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
class MambaEncoder(nn.Module):
    """Wraps any Mamba-style backbone; extracts chosen-layer hidden states.

    Hidden states come from the backbone's native output_hidden_states when
    available (the transformers MambaForCausalLM path), falling back to
    forward hooks on the backbone's layer stack for custom backbones.
    """

    def __init__(self, backbone: nn.Module, supports_positions: bool | None = None) -> None:
        super().__init__()
        self.backbone = backbone
        if supports_positions is None:
            name = type(backbone).__name__
            self.supports_positions = bool(
                hasattr(backbone, "position_embedding") or "Posids" in name
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
        - "hooks": fallback for custom backbones that lack
          output_hidden_states; last-layer hook capture.
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
def build_encoder(cfg: Any, device: str | None = None) -> MambaEncoder:
    """Build the from-scratch transformers MambaForCausalLM encoder.

    No external weights, no dispatcher: the from-scratch path is the only
    path. `kernels` (optional fused kernels) is auto-detected by transformers.
    """
    from transformers import MambaForCausalLM

    model = MambaForCausalLM(_mamba_config(cfg))
    if device:
        model = model.to(torch.device(device))  # type: ignore[arg-type]  # torch 2.5 _Wrapped typing noise
    log.info("encoder: MambaForCausalLM %s", _mamba_config(cfg))
    return MambaEncoder(model)

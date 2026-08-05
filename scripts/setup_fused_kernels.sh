#!/usr/bin/env bash
# setup_fused_kernels.sh — one-time installer for the fused Mamba CUDA kernels.
#
# WHAT / WHY
#   The pipeline normally runs on transformers' EAGER Mamba, whose sequential
#   implementation is CPU-bound: at d_model=1024 / 32k-token streams the GPU
#   sits idle while one step can take ~minutes (20 epochs on eager ~= 1.7
#   days). mamba-ssm's fused kernels move the scan onto the GPU (3-8x+), which
#   turns days into hours — the ONLY big speed lever.
#   mamba-ssm==2.1.0 is the last version that builds from source, and it needs
#   an EXACT old-torch environment (torch 2.3.x era): this script DOWNGRADES
#   torch/transformers. Run it ONCE in a FRESH Colab runtime (before anything
#   else; ~20-30 min one-time). NEVER in the middle of a session — the
#   downgrade breaks whatever else is running in that VM.
#
# Safe to run twice. Exit code is ALWAYS 0: fused kernels are an OPTIONAL
# speedup — without them the pipeline runs fine on eager Mamba (use VALIDATION
# MODE in the notebook for fast checks; never run 20-epoch schedules on
# eager). On any failure this script prints the fallback guidance and exits 0.
set -u

say()  { printf '\n[setup_fused] %s\n' "$*"; }
warn() { printf '[setup_fused][warn] %s\n' "$*"; }

fail() {
  printf '\n[setup_fused] fused kernels NOT available — this is fine; the pipeline runs on eager Mamba.\n'
  printf '[setup_fused] Use VALIDATION MODE in the notebook and never run 20-epoch schedules on eager.\n'
  exit 0
}

# ------------------------------------------------------------- guards
if python -c "import mamba_ssm, causal_conv1d" >/dev/null 2>&1; then
  say "mamba-ssm + causal_conv1d already importable — nothing to do (idempotent)."
  python -c "from mamba_ssm.ops.selective_scan_interface import selective_scan_fn; print('FUSED KERNELS OK')"
  exit 0
fi

if ! python -c "import torch" >/dev/null 2>&1; then
  warn "torch not importable yet — run scripts/colab_setup.sh first (or install torch manually)."
  fail
fi

CC="$(python - <<'PY' 2>/dev/null
import torch
print(".".join(str(x) for x in torch.cuda.get_device_capability()) if torch.cuda.is_available() else "")
PY
)"
if [ -z "$CC" ] || ! awk -v c="$CC" 'BEGIN{exit !(c+0 >= 8.0)}'; then
  say "no sm_80+ GPU detected (capability '${CC:-none}') — fused kernels need sm_80+ (A100/L4/H100). Skipping."
  fail
fi

# Fresh-runtime heuristic: Colab ships torch >= 2.11, which is NOT the pinned
# 2.3.1 this script downgrades to. If the current torch is anything else, the
# user probably ran this in a used runtime by mistake — warn loudly, continue.
TORCH_VER="$(python -c 'import torch; print(torch.__version__)' 2>/dev/null)"
case "$TORCH_VER" in
  2.3.*) say "torch $TORCH_VER already at the pinned version — continuing." ;;
  *)     warn "torch is $TORCH_VER, not the pinned 2.3.1 — this script DOWNGRADES it. Did you delete the runtime? Run this in a FRESH Colab runtime (or your environment will be broken). Continuing anyway..." ;;
esac

# ------------------------------------------------------------- install
# Order matters: torch+transformers FIRST (mamba-ssm pins/builds against the
# already-installed torch), then causal-conv1d (mamba-ssm's build dep), then
# mamba-ssm itself.
say "Step 1/3: pinning torch==2.3.1 + transformers==4.44.2..."
if ! python -m pip install "torch==2.3.1" "transformers==4.44.2" -q; then
  warn "torch/transformers pin failed."
  fail
fi

say "Step 2/3: installing causal-conv1d..."
if ! python -m pip install causal-conv1d; then
  warn "causal-conv1d install failed."
  fail
fi

say "Step 3/3: building mamba-ssm==2.1.0 from source (30-minute timeout)..."
if command -v timeout >/dev/null 2>&1; then
  if ! timeout 1800 python -m pip install mamba-ssm==2.1.0 --no-build-isolation; then
    warn "mamba-ssm build failed/timed out (30 min)."
    fail
  fi
else
  if ! python -m pip install mamba-ssm==2.1.0 --no-build-isolation; then
    warn "mamba-ssm build failed."
    fail
  fi
fi

# ------------------------------------------------------------- verify
if python -c "import mamba_ssm, causal_conv1d; from mamba_ssm.ops.selective_scan_interface import selective_scan_fn; print('FUSED KERNELS OK')" >/dev/null 2>&1; then
  say "FUSED KERNELS OK — the pipeline will now use the GPU fast path (3-8x+ faster steps)."
else
  warn "import check failed after a successful-looking install."
  fail
fi

exit 0

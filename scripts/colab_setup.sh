#!/usr/bin/env bash
# colab_setup.sh — idempotent environment setup for Google Colab (and safe on
# any Linux box: GPU checks degrade gracefully, never fails on install errors).
#
# What it does:
#   1. Reports GPU / compute capability and the precision+kernel path that
#      will be used (bf16+fused on sm_80+, fp16+eager on T4, CPU otherwise).
#   2. Mounts Google Drive when COLAB_DRIVE is set and google.colab is present,
#      then sources $COLAB_DRIVE/.env (mirrored to /content/ssm-phylo/.env).
#   3. Creates the persistent/scratch directory layout.
#   4. Installs deps from requirements-colab.txt (ALWAYS run — pip is
#      idempotent, satisfied pins are fast no-ops), then installs the
#      ssm_phylo package itself with `pip install -e .` (src layout).
#   5. On sm_80+ only: attempts `pip install mamba-ssm==2.1.0 --no-build-isolation`
#      with a 15-minute timeout; on any failure prints "FALLBACK: eager Mamba"
#      and continues.
#
# Exit code is ALWAYS 0. Safe to run twice.
set -u

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

say()  { printf '\n[colab_setup] %s\n' "$*"; }
warn() { printf '[colab_setup][warn] %s\n' "$*"; }

# ---------------------------------------------------------------- 0. local env
# A repo-root .env (local dev) takes precedence as a base; Drive .env (if any)
# is sourced later and overrides.
if [ -f "$REPO_DIR/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "$REPO_DIR/.env"
  set +a
  say "Sourced $REPO_DIR/.env"
fi

# ------------------------------------------------------------ 1. GPU detection
if command -v nvidia-smi >/dev/null 2>&1; then
  say "nvidia-smi:"
  nvidia-smi 2>/dev/null || warn "nvidia-smi failed"
else
  warn "nvidia-smi not found (no NVIDIA driver?)"
fi

CC=""  # compute capability "major.minor", empty when unknown
if python -c "import torch" >/dev/null 2>&1; then
  CC="$(python - <<'PY' 2>/dev/null
import torch
if torch.cuda.is_available():
    print(".".join(str(x) for x in torch.cuda.get_device_capability()))
else:
    print("")
PY
)"
fi
# torch reports nothing (CPU-only build or no torch) but nvidia-smi sees a GPU:
# fall back to a name->capability heuristic so precision paths still print.
if [ -z "$CC" ] && command -v nvidia-smi >/dev/null 2>&1; then
  GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1)"
  case "$GPU_NAME" in
    *H100*|*H200*|*B200*|*A100*|*A6000*|*A40*|*L40*|*L4*|*L20*|*RTX\ 5*|*RTX\ 4*|*RTX\ 3*) CC="8.0" ;;
    *T4*|*Quadro*T4*|*RTX\ 2*|*GTX\ 16*|*V100*) CC="7.5" ;;
    *GTX\ 10*|*P100*) CC="6.1" ;;
    *) CC="" ;;
  esac
fi

if [ -z "$CC" ]; then
  say "No GPU detected (compute capability unknown). Path: CPU / dev — eager Mamba fallback, fp32."
elif awk -v c="$CC" 'BEGIN{exit !(c+0 >= 8.0)}'; then
  say "GPU compute capability $CC (sm_80+). Path: bf16 precision + fused mamba kernels (with eager fallback)."
elif awk -v c="$CC" 'BEGIN{exit !(c+0 >= 7.5)}'; then
  say "GPU compute capability $CC (sm_75, e.g. T4). Path: fp16 precision + eager Mamba fallback (dev/smoke only)."
else
  say "GPU compute capability $CC (< sm_75). Path: eager Mamba fallback."
fi

# ------------------------------------------------------- 2. Drive mount + env
if [ -n "${COLAB_DRIVE:-}" ] && python -c "import google.colab" >/dev/null 2>&1; then
  if [ -d "$COLAB_DRIVE" ]; then
    say "Drive already mounted at $COLAB_DRIVE"
  else
    say "Mounting Google Drive..."
    python - <<'PY' || warn "Drive mount failed (run again after granting access)"
from google.colab import drive
drive.mount("/content/drive")
PY
  fi
  if [ -d "$COLAB_DRIVE" ] && [ -f "$COLAB_DRIVE/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "$COLAB_DRIVE/.env"
    set +a
    say "Sourced $COLAB_DRIVE/.env"
    if mkdir -p /content/ssm-phylo 2>/dev/null; then
      cp -f "$COLAB_DRIVE/.env" /content/ssm-phylo/.env
      say "Session env copy written to /content/ssm-phylo/.env"
    else
      warn "/content not writable; skipping session env copy"
    fi
  else
    warn "$COLAB_DRIVE/.env not found; using current env vars"
  fi
else
  say "Skipping Drive mount (not Colab or COLAB_DRIVE unset). Env vars used as-is."
fi

# ------------------------------------------------------ 3. directory layout
mkdir_guard() {  # mkdir_guard "$path" "$description"
  if [ -n "${1:-}" ]; then
    mkdir -p "$1" 2>/dev/null && say "dir ready: $1" || warn "cannot create $2 ($1)"
  fi
}
mkdir_guard "${DATA_DIR:-}"          "DATA_DIR"
mkdir_guard "${CKPT_DIR:-}"          "CKPT_DIR"
mkdir_guard "${RESULTS_DIR:-}"       "RESULTS_DIR"
mkdir_guard "${LOCAL_CKPT_DIR:-}"    "LOCAL_CKPT_DIR"
mkdir_guard "${LOCAL_DATA_DIR:-}"    "LOCAL_DATA_DIR"
if [ -n "${COLAB_DRIVE:-}" ]; then
  mkdir_guard "$COLAB_DRIVE/weights" "COLAB_DRIVE/weights"
fi

# ------------------------------------------------------------ 4. pip installs
# ALWAYS run the requirements install. Colab ships torch, so a
# "torch importable -> skip" guard would skip EVERYTHING else (transformers,
# dendropy, biopython, datasets, PyYAML, ...) and the repo package — breaking
# `import ssm_phylo` on fresh VMs. pip is idempotent: already-satisfied pins
# are fast no-ops, so running this twice is safe and quick.
say "Installing pinned deps from requirements-colab.txt..."
if python -m pip install -r "$REPO_DIR/requirements-colab.txt"; then
  say "Pinned deps installed."
else
  warn "pip install failed for requirements-colab.txt (retrying common deps is up to you)"
fi

# Install the repo package itself (src layout — sys.path hacks do NOT help).
# --no-deps: pyproject.toml's strict pins (torch==2.3.1, numpy==1.24.4,
# transformers==4.44.2) would downgrade Colab's preinstalled versions
# (multi-GB, slow, often fails); dependencies are handled by
# requirements-colab.txt above.
say "Installing ssm_phylo package (pip install -e . --no-deps)..."
if python -m pip install -e "$REPO_DIR" --no-deps; then
  say "ssm_phylo installed."
else
  warn "pip install -e . --no-deps failed; ssm_phylo imports will not work (check pip errors above)"
fi

# --------------------------------------- 5. mamba-ssm attempt (sm_80+ only)
NEED_MAMBA=0
if [ -n "$CC" ] && awk -v c="$CC" 'BEGIN{exit !(c+0 >= 8.0)}'; then
  NEED_MAMBA=1
fi
if [ "$NEED_MAMBA" = "1" ]; then
  # Gate on a CUDA-capable torch and nvcc: without both, a from-source build
  # is doomed (this also keeps non-Colab boxes fast).
  CUDA_TORCH=0
  if python - <<'PY' >/dev/null 2>&1; then CUDA_TORCH=1; fi
import torch
exit(0 if torch.cuda.is_available() else 1)
PY
  if [ "$CUDA_TORCH" = "0" ]; then
    say "CUDA-capable torch not detected; skipping mamba-ssm build (would fail from source)."
    printf '\n[colab_setup] FALLBACK: eager Mamba\n'
  elif ! command -v nvcc >/dev/null 2>&1; then
    warn "nvcc not found; skipping mamba-ssm source build."
    printf '\n[colab_setup] FALLBACK: eager Mamba\n'
  elif python -c "import mamba_ssm" >/dev/null 2>&1; then
    say "mamba-ssm already installed; skipping build"
  elif command -v timeout >/dev/null 2>&1 && command -v pip >/dev/null 2>&1; then
    say "Building mamba-ssm==2.1.0 (15-minute timeout)..."
    if timeout 900 pip install mamba-ssm==2.1.0 --no-build-isolation; then
      say "mamba-ssm installed."
    else
      warn "mamba-ssm build failed/timed out."
      printf '\n[colab_setup] FALLBACK: eager Mamba\n'
    fi
  else
    warn "timeout/pip not available; skipping mamba-ssm build."
    printf '\n[colab_setup] FALLBACK: eager Mamba\n'
  fi
else
  say "Skipping mamba-ssm build (requires sm_80+; got '${CC:-no GPU}')."
  printf '\n[colab_setup] FALLBACK: eager Mamba\n'
fi

say "Setup complete."
exit 0

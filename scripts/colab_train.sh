#!/usr/bin/env bash
# colab_train.sh — thin Colab training wrapper (resumable by design).
#
#   1. sources .env (repo root, then $COLAB_DRIVE if mounted)
#   2. pulls the Drive checkpoint mirror -> LOCAL_CKPT_DIR (idempotent)
#   3. launches the Phase 3 training entrypoint with env-derived paths
#
# Extra flags can be passed through: e.g.
#   bash scripts/colab_train.sh --max-steps 10000 --log wandb
# or set SSM_PHYLO_TRAIN_EXTRA. Exit code = train's exit code.
set -u

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

say()  { printf '\n[colab_train] %s\n' "$*"; }
warn() { printf '[colab_train][warn] %s\n' "$*"; }

[ -f "$REPO_DIR/.env" ] && { set -a; # shellcheck disable=SC1091
  source "$REPO_DIR/.env"; set +a; }

# Drive-mounted .env overrides the repo one (Colab session)
if [ -n "${COLAB_DRIVE:-}" ] && [ -f "$COLAB_DRIVE/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "$COLAB_DRIVE/.env"
  set +a
fi

for var in DATA_DIR CKPT_DIR LOCAL_CKPT_DIR; do
  if [ -z "${!var:-}" ]; then warn "$var is unset; training will fail without it"; fi
done

CONFIG="${SSM_PHYLO_TRAIN_CONFIG:-configs/train_l4.yaml}"
say "pulling checkpoint mirror (Drive -> scratch)"
bash "$REPO_DIR/scripts/sync_drive.sh" pull

# interpreter resolution (mirrors simulate_big.sh)
PYTHON_BIN="${PYTHON:-python}"
if [ -n "${VIRTUAL_ENV:-}" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then
  PYTHON_BIN="$VIRTUAL_ENV/bin/python"
elif [ -x "$REPO_DIR/.venv/bin/python" ]; then
  PYTHON_BIN="$REPO_DIR/.venv/bin/python"
fi

ARGS=(--config "$REPO_DIR/$CONFIG" --resume latest
      --data-dir "$DATA_DIR" --ckpt-dir "$CKPT_DIR")
[ -n "${RESULTS_DIR:-}" ] && ARGS+=(--results-dir "$RESULTS_DIR")
# shellcheck disable=SC2206
ARGS+=( $SSM_PHYLO_TRAIN_EXTRA "$@" )

say "running: $PYTHON_BIN -m ssm_phylo.train ${ARGS[*]}"
cd "$REPO_DIR" || exit 1
exec "$PYTHON_BIN" -m ssm_phylo.train "${ARGS[@]}"

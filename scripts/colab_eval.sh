#!/usr/bin/env bash
# colab_eval.sh — evaluate the latest checkpoint on the test parquet.
#
#   1. sources .env (repo root, then $COLAB_DRIVE if mounted)
#   2. pulls the Drive checkpoint mirror -> LOCAL_CKPT_DIR (idempotent)
#   3. runs evaluate on $DATA_DIR/test.parquet with LOCAL_CKPT_DIR/latest.pt,
#      writing to $RESULTS_DIR/eval_v1/
#
# Exit code: non-zero ONLY if evaluate itself fails (Drive sync issues never
# fail the caller — sync_drive.sh always exits 0 and errors are warnings).
set -u

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

say()  { printf '\n[colab_eval] %s\n' "$*"; }
warn() { printf '[colab_eval][warn] %s\n' "$*"; }

[ -f "$REPO_DIR/.env" ] && { set -a; # shellcheck disable=SC1091
  source "$REPO_DIR/.env"; set +a; }
if [ -n "${COLAB_DRIVE:-}" ] && [ -f "$COLAB_DRIVE/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "$COLAB_DRIVE/.env"
  set +a
fi

for var in DATA_DIR CKPT_DIR LOCAL_CKPT_DIR; do
  if [ -z "${!var:-}" ]; then warn "$var unset; relying on defaults/env"; fi
done
[ -n "${RESULTS_DIR:-}" ] || RESULTS_DIR="${LOCAL_CKPT_DIR%/}/../results"

say "pulling checkpoint mirror (Drive -> scratch)"
bash "$REPO_DIR/scripts/sync_drive.sh" pull

CKPT="${SSM_PHYLO_EVAL_CKPT:-$LOCAL_CKPT_DIR/latest.pt}"
TEST_PARQUET="${SSM_PHYLO_EVAL_TEST:-$DATA_DIR/test.parquet}"
OUT_DIR="${SSM_PHYLO_EVAL_OUT:-$RESULTS_DIR/eval_v1}"

if [ ! -f "$CKPT" ]; then
  warn "no checkpoint at $CKPT — run training first (scripts/colab_train.sh)"
  exit 1
fi
if [ ! -f "$TEST_PARQUET" ]; then
  warn "no test parquet at $TEST_PARQUET — run simulation first (scripts/simulate_big.sh --smoke)"
  exit 1
fi

PYTHON_BIN="${PYTHON:-python}"
if [ -n "${VIRTUAL_ENV:-}" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then
  PYTHON_BIN="$VIRTUAL_ENV/bin/python"
elif [ -x "$REPO_DIR/.venv/bin/python" ]; then
  PYTHON_BIN="$REPO_DIR/.venv/bin/python"
fi

say "evaluating $CKPT on $TEST_PARQUET -> $OUT_DIR"
cd "$REPO_DIR" || exit 1
exec "$PYTHON_BIN" -m ssm_phylo.evaluate \
  --checkpoint "$CKPT" \
  --test-parquet "$TEST_PARQUET" \
  --out-dir "$OUT_DIR" \
  ${SSM_PHYLO_EVAL_EXTRA:-}

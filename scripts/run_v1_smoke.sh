#!/usr/bin/env bash
# run_v1_smoke.sh — the one-command "does V1 work?" check (CPU-only, ~5-10 min).
#
#   1. simulate tiny data: python -m ssm_phylo.data.simulation --smoke into
#      $LOCAL_DATA_DIR scratch, consolidated + split to a local tmp "Drive"
#   2. train --toy (50 steps, from_scratch encoder) -> checkpoints
#   3. evaluate --checkpoint <latest.pt> --test-parquet <tmp test.parquet>
#      --out-dir <tmp results> --max-alignments 50
#   4. print the markdown table + PASS/FAIL line.
#
# PASS: mean RF < 0.5 (the bar is LOW on toy data; the point is the CHAIN
# works). Exit code: 0 unless a CHAIN step fails (accuracy FAIL still exits 0
# with a note — on a 50-step toy model accuracy is not the goal).
#
# --quick: shrink further for CI (fewer train steps + eval alignments).
set -u

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

say()  { printf '\n[run_v1_smoke] %s\n' "$*"; }
warn() { printf '[run_v1_smoke][warn] %s\n' "$*"; }

QUICK=0
[ "${1:-}" = "--quick" ] && QUICK=1

PYTHON_BIN="${PYTHON:-python}"
if [ -n "${VIRTUAL_ENV:-}" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then
  PYTHON_BIN="$VIRTUAL_ENV/bin/python"
elif [ -x "$REPO_DIR/.venv/bin/python" ]; then
  PYTHON_BIN="$REPO_DIR/.venv/bin/python"
fi

TMP="$(mktemp -d /tmp/ssm_v1_smoke.XXXXXX)"
# Each step gets its OWN scratch: simulation --smoke and train --toy both
# simulate data (raw/ + parquet) — sharing a scratch dir would mix their
# trees (e.g. the smoke's t020 trees landing in the toy's raw dir).
export LOCAL_CKPT_DIR="$TMP/ckpts"
export CKPT_DIR="$TMP/drive_ckpts"
mkdir -p "$LOCAL_CKPT_DIR" "$CKPT_DIR"

step() { say "STEP $1: $2"; }

set -e

step 1/3 "simulate tiny data (simulation --smoke)"
export LOCAL_DATA_DIR="$TMP/smoke_local"
export DATA_DIR="$TMP/drive"
mkdir -p "$LOCAL_DATA_DIR" "$DATA_DIR"
"$PYTHON_BIN" -m ssm_phylo.data.simulation --smoke

step 2/3 "train toy model (50 steps)"
export LOCAL_DATA_DIR="$TMP/toy_local"
unset DATA_DIR
mkdir -p "$LOCAL_DATA_DIR"
if [ "$QUICK" = "1" ]; then
  "$PYTHON_BIN" -m ssm_phylo.train --toy --max-steps 8 --batch-size 4
else
  "$PYTHON_BIN" -m ssm_phylo.train --toy
fi

step 3/3 "evaluate chain (predict -> tree -> RF)"
MAX_ALN=50
[ "$QUICK" = "1" ] && MAX_ALN=10
"$PYTHON_BIN" -m ssm_phylo.evaluate \
  --checkpoint "$LOCAL_CKPT_DIR/latest.pt" \
  --test-parquet "$TMP/drive/test.parquet" \
  --out-dir "$TMP/results" \
  --max-alignments "$MAX_ALN"

set +e

RESULT_CSV="$TMP/results/results.csv"
if [ ! -f "$RESULT_CSV" ]; then
  warn "CHAIN ERROR: no results.csv produced"
  exit 1
fi
MEAN_RF="$("$PYTHON_BIN" - <<PY
import csv
rows = list(csv.DictReader(open("$RESULT_CSV")))
print(f"{sum(float(r['rf_pred']) for r in rows)/len(rows):.4f}")
PY
)"
MEAN_UB="$("$PYTHON_BIN" - <<PY
import csv
rows = list(csv.DictReader(open("$RESULT_CSV")))
print(f"{sum(float(r['rf_true_dist_upper_bound']) for r in rows)/len(rows):.4f}")
PY
)"

say "V1 smoke results (toy):"
printf '| metric | value |\n|---|---|\n| mean RF (predicted tree vs true) | %s |\n| mean RF (true-dist upper bound) | %s |\n| alignments | %s |\n' \
  "$MEAN_RF" "$MEAN_UB" "$(wc -l < "$RESULT_CSV" | awk '{print $1-1}')"
cat "$TMP/results/results.csv" > /dev/null
tail -n +2 "$RESULT_CSV" | cut -d, -f1,2,4 | head -12

if [ "$(echo "$MEAN_RF < 0.5" | bc 2>/dev/null || python3 -c "print(1 if float('$MEAN_RF') < 0.5 else 0)")" = "1" ]; then
  say "PASS: mean RF $MEAN_RF < 0.5"
  exit 0
else
  say "FAIL (accuracy): mean RF $MEAN_RF >= 0.5 on a 50-step toy model — expected; the CHAIN itself worked (see table)."
  exit 0
fi

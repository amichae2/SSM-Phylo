#!/usr/bin/env bash
# simulate_big.sh — orchestrate the full simulation as a resumable job.
#
# Default (production): 8 tip counts x 200 trees/bin x 3 lengths x 21
# replicates = 100,800 unique (tree, alignment) pairs >= 100k, written to
# $LOCAL_DATA_DIR/raw ONLY (scratch). Then consolidated + split into
# train/val/test parquet and copied to $DATA_DIR on Drive atomically.
#
#   --smoke       200 alignments only (tips 20, len 150); still exercises the
#                 full consolidate -> Drive path
#   --resume      skip trees/alignments already present in scratch
#   --background  nohup + pidfile + log under $LOCAL_DATA_DIR
#   --workers N   parallel simulation processes (default 4)
#   --data-dir / --local-data-dir  overrides for env-less runs
#
# Never writes raw data to Drive. Exit code 0 on success.
set -u

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

say()  { printf '\n[simulate_big] %s\n' "$*"; }
warn() { printf '[simulate_big][warn] %s\n' "$*"; }

if [ -f "$REPO_DIR/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "$REPO_DIR/.env"
  set +a
fi

SMOKE=0
RESUME=0
BACKGROUND=0
WORKERS=4
DATA_DIR_FLAG=""
LOCAL_DATA_DIR_FLAG=""

while [ $# -gt 0 ]; do
  case "$1" in
    --smoke)     SMOKE=1 ;;
    --resume)    RESUME=1 ;;
    --background) BACKGROUND=1 ;;
    --workers)   WORKERS="${2:-4}"; shift ;;
    --data-dir)  DATA_DIR_FLAG="${2:-}"; shift ;;
    --local-data-dir) LOCAL_DATA_DIR_FLAG="${2:-}"; shift ;;
    -h|--help)
      say "usage: simulate_big.sh [--smoke] [--resume] [--background] [--workers N] [--data-dir DIR] [--local-data-dir DIR]"
      exit 0 ;;
    *) warn "unknown arg '$1'"; exit 1 ;;
  esac
  shift
done

if [ -z "${LOCAL_DATA_DIR:-}" ] && [ -z "$LOCAL_DATA_DIR_FLAG" ]; then
  warn "LOCAL_DATA_DIR unset; defaulting to /content/data"
fi

CMD="big"
if [ "$SMOKE" = "1" ]; then
  CMD="smoke"
elif [ "$RESUME" = "1" ]; then
  CMD="big --resume"
fi

# interpreter resolution: explicit PYTHON env > active venv > repo venv > python
PYTHON_BIN="${PYTHON:-python}"
if [ -n "${VIRTUAL_ENV:-}" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then
  PYTHON_BIN="$VIRTUAL_ENV/bin/python"
elif [ -x "$REPO_DIR/.venv/bin/python" ]; then
  PYTHON_BIN="$REPO_DIR/.venv/bin/python"
fi

PY_ARGS="--workers $WORKERS"
[ -n "$DATA_DIR_FLAG" ]       && PY_ARGS="$PY_ARGS --data-dir $DATA_DIR_FLAG"
[ -n "$LOCAL_DATA_DIR_FLAG" ] && PY_ARGS="$PY_ARGS --local-data-dir $LOCAL_DATA_DIR_FLAG"

RUN="$PYTHON_BIN -m ssm_phylo.data.simulation $CMD $PY_ARGS"
say "command: $RUN"

if [ "$BACKGROUND" = "1" ]; then
  LOCAL="${LOCAL_DATA_DIR_FLAG:-${LOCAL_DATA_DIR:-/content/data}}"
  mkdir -p "$LOCAL"
  LOG="$LOCAL/simulate.log"
  PIDFILE="$LOCAL/simulate.pid"
  if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    warn "simulation already running (pid $(cat "$PIDFILE")); not starting a second one."
    exit 0
  fi
  nohup bash -c "cd '$REPO_DIR' && $RUN" > "$LOG" 2>&1 &
  echo $! > "$PIDFILE"
  say "started in background (pid $(cat "$PIDFILE")); log: $LOG"
  say "tail -f $LOG   |   kill \$(cat $PIDFILE) to stop"
  exit 0
fi

cd "$REPO_DIR" || exit 1
bash -c "$RUN"
exit 0

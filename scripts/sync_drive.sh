#!/usr/bin/env bash
# sync_drive.sh — Drive <-> scratch checkpoint sync helper for training.
#
#   sync_drive.sh pull   $CKPT_DIR/    -> $LOCAL_CKPT_DIR/   (session start)
#   sync_drive.sh push   $LOCAL_CKPT_DIR/ -> $CKPT_DIR/      (after each save)
#
# Push is delete-safe: remote files that exist only on Drive are pruned, but
# anything matching the protected list is NEVER deleted:
#   - hardcoded: latest.pt, best.pt, ckpt-interrupted.pt
#   - plus any rsync filter rules in $LOCAL_CKPT_DIR/.drivekeep (or the
#     fallback plain-name list in $CKPT_DIR/.drivekeep.names)
#
# Transient Drive errors never fail the caller: retry once, then warn and
# exit 0. Idempotent.
set -u

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

say()  { printf '[sync_drive] %s\n' "$*"; }
warn() { printf '[sync_drive][warn] %s\n' "$*"; }

if [ -f "$REPO_DIR/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "$REPO_DIR/.env"
  set +a
fi

MODE="${1:-}"
SRC_REMOTE="${CKPT_DIR:-}"
LOCAL="${LOCAL_CKPT_DIR:-}"
SRC_LOCAL="${LOCAL:-}"

if [ -z "$MODE" ]; then
  warn "usage: sync_drive.sh {pull|push}"
  exit 0
fi
if [ -z "$SRC_REMOTE" ] || [ -z "$LOCAL" ]; then
  warn "CKPT_DIR and/or LOCAL_CKPT_DIR unset; nothing to sync."
  exit 0
fi

# ------------------------------------------------------------------ helpers
retry() {  # retry <n> <cmd...>  -> 0 on success
  local n="$1"; shift
  local i
  for i in $(seq 1 "$n"); do
    if "$@"; then return 0; fi
    if [ "$i" -lt "$n" ]; then sleep 2; warn "attempt $i failed; retrying"; fi
  done
  return 1
}

prune_remote() {  # delete remote files not on local and not protected
  local remote="$1" local_dir="$2"
  local keep_names="latest.pt best.pt ckpt-interrupted.pt"
  [ -f "$local_dir/.drivekeep.names" ] && keep_names="$keep_names $(tr '\n' ' ' < "$local_dir/.drivekeep.names" 2>/dev/null)"
  find "$remote" -maxdepth 1 -type f ! -name ".drivekeep*" | while IFS= read -r f; do
    local base; base="$(basename "$f")"
    local keep=0
    for k in $keep_names; do [ "$base" = "$k" ] && keep=1; done
    [ "$keep" = "1" ] && continue
    [ -e "$local_dir/$base" ] || { warn "pruning remote-only file: $base"; rm -f "$f"; }
  done
}

# ---------------------------------------------------------------- do the sync
case "$MODE" in
  pull)
    mkdir -p "$LOCAL"
    if command -v rsync >/dev/null 2>&1; then
      if retry 2 rsync -a --partial "$SRC_REMOTE/" "$LOCAL/"; then
        say "pull OK ($SRC_REMOTE -> $LOCAL)"
      else
        warn "pull failed after retry (Drive transient?); continuing without local copy."
      fi
    else
      warn "rsync missing; cp -a fallback"
      retry 2 cp -a "$SRC_REMOTE/." "$LOCAL/" && say "pull OK (cp fallback)"
    fi
    ;;
  push)
    mkdir -p "$SRC_REMOTE"
    if command -v rsync >/dev/null 2>&1; then
      # --delete with protect filters: never delete protected/keep-listed files.
      if [ -f "$LOCAL/.drivekeep" ]; then
        if retry 2 rsync -a --delete --filter="merge $LOCAL/.drivekeep" \
               --filter="P latest.pt" --filter="P best.pt" --filter="P ckpt-interrupted.pt" \
               "$LOCAL/" "$SRC_REMOTE/"; then
          say "push OK (rsync, delete with .drivekeep protection)"
        else
          warn "push failed after retry; keeping Drive state as-is."
        fi
      else
        if retry 2 rsync -a --delete \
               --filter="P latest.pt" --filter="P best.pt" --filter="P ckpt-interrupted.pt" \
               "$LOCAL/" "$SRC_REMOTE/"; then
          say "push OK (rsync, delete with default protection)"
        else
          warn "push failed after retry; keeping Drive state as-is."
        fi
      fi
    else
      warn "rsync missing; cp -a fallback with manual prune"
      if retry 2 cp -a "$LOCAL/." "$SRC_REMOTE/"; then
        prune_remote "$SRC_REMOTE" "$LOCAL"
        say "push OK (cp fallback)"
      else
        warn "push failed after retry; keeping Drive state as-is."
      fi
    fi
    ;;
  *)
    warn "unknown mode '$MODE' (expected pull|push)"
    ;;
esac

exit 0

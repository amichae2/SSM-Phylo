#!/usr/bin/env bash
# download_weights.sh — download ProtMamba encoder weights from the
# Bitbol-Lab/ProtMamba-ssm GitHub release into $PROT_MAMBA_CKPT.
#
# Idempotent: skips if weights are already present. Uses a lock file so
# concurrent sessions don't double-download. Prints size/SHA for verification.
# Always exits 0.
set -u

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

say()  { printf '\n[download_weights] %s\n' "$*"; }
warn() { printf '[download_weights][warn] %s\n' "$*"; }

if [ -f "$REPO_DIR/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "$REPO_DIR/.env"
  set +a
fi

REPO="Bitbol-Lab/ProtMamba-ssm"
DEST="${PROT_MAMBA_CKPT:-}"
LOCK="$DEST/.download.lock"
# releases/latest is v1.1 (no assets); the weights live in v1.0. Candidate
# tags in priority order, then the known-good direct URL.
CANDIDATE_TAGS=("latest" "tags/v1.0")
KNOWN_URL="https://github.com/$REPO/releases/download/v1.0/ProtMamba_model-weights.zip"

if [ -z "$DEST" ]; then
  warn "PROT_MAMBA_CKPT is not set; nothing to do."
  exit 0
fi

# ---------------------------------------------------------------- idempotent
has_weights() { [ -n "$(find "$DEST" -maxdepth 2 -type f ! -name '.download.lock' 2>/dev/null | head -n1)" ]; }

if has_weights; then
  say "Weights already present in $DEST; skipping download."
  du -sh "$DEST" 2>/dev/null || true
  find "$DEST" -maxdepth 2 -type f -exec sha256sum {} \; 2>/dev/null || true
  exit 0
fi

# ------------------------------------------------------- concurrent sessions
mkdir -p "$DEST"
if ! mkdir "$LOCK" 2>/dev/null; then
  say "Another session is downloading (lock $LOCK). Waiting up to 10 minutes..."
  for _ in $(seq 1 120); do
    if [ ! -d "$LOCK" ] && has_weights; then
      say "Weights appeared (other session finished)."
      exit 0
    fi
    sleep 5
  done
  warn "Lock still held after 10 minutes; continuing anyway (may race)."
fi
trap 'rm -rf "$LOCK"' EXIT

# ------------------------------------------------------------------ download
say "Downloading ProtMamba weights from $REPO release into $DEST"
mkdir -p "$DEST"

if command -v curl >/dev/null 2>&1; then
  DL="curl -fL --retry 3 -o"
else
  warn "curl not found; trying wget."
  DL="wget -O"
fi

# 1) gather asset URLs from the GitHub API (latest first, then known tags)
ASSET_URLS=""
for TAG in "${CANDIDATE_TAGS[@]}"; do
  ASSET_URLS="$ASSET_URLS $(curl -fsSL --max-time 60 "https://api.github.com/repos/$REPO/releases/$TAG" 2>/dev/null | python -c "
import json, sys
try:
    data = json.load(sys.stdin)
    for a in data.get('assets', []):
        print(a['browser_download_url'])
except Exception:
    pass
" 2>/dev/null || true)"
done
[ -z "$ASSET_URLS" ] && ASSET_URLS="$KNOWN_URL"

# 2) download + extract
OK=0
DL_DIR="$DEST/.downloads"
mkdir -p "$DL_DIR"
for URL in $ASSET_URLS; do
  NAME="$(basename "$URL")"
  say "Downloading $NAME"
  if $DL "$DL_DIR/$NAME" "$URL"; then
    case "$NAME" in
      *.zip)
        if command -v unzip >/dev/null 2>&1 && unzip -o -q "$DL_DIR/$NAME" -d "$DL_DIR/extracted"; then
          # flatten: move files from any nested folder into $DEST
          (cd "$DL_DIR/extracted" && find . -type f -exec mv -f {} "$DEST"/ \; 2>/dev/null)
          OK=1
        else
          warn "unzip failed for $NAME"
        fi
        ;;
      *tar.gz|*.tgz)
        if tar -xzf "$DL_DIR/$NAME" -C "$DEST" --strip-components=1 2>/dev/null; then OK=1; else warn "extract failed for $NAME"; fi
        ;;
      *)
        cp -f "$DL_DIR/$NAME" "$DEST/$NAME" && OK=1
        ;;
    esac
  else
    warn "download failed: $URL"
  fi
done
rm -rf "$DL_DIR"

# --------------------------------------------------------------- verification
if [ "$OK" = "1" ] && has_weights; then
  say "Download finished. Verification:"
  du -sh "$DEST" 2>/dev/null || true
  find "$DEST" -maxdepth 2 -type f -exec sha256sum {} \; 2>/dev/null || true
  say "Done. Weights at $DEST"
else
  warn "Download failed; nothing usable in $DEST."
fi
exit 0

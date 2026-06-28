#!/usr/bin/env bash
# Deploy tg-agent to the VPS and ALWAYS prune build artifacts afterwards, so the
# 14 GB box never fills up again (a full disk makes the linker fail with a cryptic
# `collect2: ld returned 1`). Run from the repo root:  ./deploy.sh
#
# Env overrides: VPS_HOST (default root@5.129.234.9), SSH_KEY
# (default ~/.ssh/id_ed25519_vps), REMOTE_DIR (default /opt/tg-agent).
set -euo pipefail

VPS_HOST="${VPS_HOST:-root@5.129.234.9}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519_vps}"
REMOTE_DIR="${REMOTE_DIR:-/opt/tg-agent}"
SSH=(ssh -i "$SSH_KEY" -o ConnectTimeout=15)

echo "==> rsync source to $VPS_HOST:$REMOTE_DIR"
rsync -az -e "ssh -i $SSH_KEY" \
  --exclude .env --exclude state.json --exclude sessions/ \
  --exclude target/ --exclude .git/ \
  ./ "$VPS_HOST:$REMOTE_DIR/"

echo "==> build release + restart + PRUNE artifacts"
"${SSH[@]}" "$VPS_HOST" "bash -s" <<REMOTE
set -e
cd "$REMOTE_DIR"
"\$HOME/.cargo/bin/cargo" build --release 2>&1 | tail -3
systemctl restart tg-agent
sleep 3
systemctl is-active tg-agent
grep -m1 '^version' Cargo.toml

# --- cleanup: keep only what the running bot needs (target/release) ---
# Debug + test artifacts are disposable; cargo rebuilds them on demand.
rm -rf target/debug target/tmp
# Drop stale incremental compile caches in release too.
rm -rf target/release/incremental
# Trim systemd journal so logs don't grow without bound.
journalctl --vacuum-size=50M >/dev/null 2>&1 || true

echo "--- disk after cleanup ---"
df -h / | tail -1
REMOTE

echo "==> done"

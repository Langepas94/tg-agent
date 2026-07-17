#!/usr/bin/env bash
# Deploy tg-agent to the VPS and ALWAYS prune build artifacts afterwards, so the
# 14 GB box never fills up again (a full disk makes the linker fail with a cryptic
# `collect2: ld returned 1`). Run from the repo root:  ./deploy.sh
#
# Env overrides: VPS_HOST (default root@5.129.234.9), SSH_KEY
# (default ~/Documents/ai/.ssh/timeweb_tg_agent_ed25519), REMOTE_DIR (default /opt/tg-agent),
# ENABLE_NGINX_PROXY=0 to skip installing/updating the public /admin proxy.
set -euo pipefail

VPS_HOST="${VPS_HOST:-root@5.129.234.9}"
SSH_KEY="${SSH_KEY:-$HOME/Documents/ai/.ssh/timeweb_tg_agent_ed25519}"
REMOTE_DIR="${REMOTE_DIR:-/opt/tg-agent}"
ENABLE_NGINX_PROXY="${ENABLE_NGINX_PROXY:-1}"
SSH=(ssh -i "$SSH_KEY" -o ConnectTimeout=15)

echo "==> rsync source to $VPS_HOST:$REMOTE_DIR"
rsync -az -e "ssh -i $SSH_KEY" \
  --exclude .env --exclude state.json --exclude sessions/ \
  --exclude target/ --exclude .git \
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

if [ "$ENABLE_NGINX_PROXY" = "1" ]; then
  echo "--- nginx public /admin proxy ---"
  if ! command -v nginx >/dev/null 2>&1; then
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq
    apt-get install -y nginx >/dev/null
  fi
  cat >/etc/nginx/sites-available/tg-agent <<'NGINX'
server {
    listen 80 default_server;
    listen [::]:80 default_server;
    server_name 5.129.234.9 _;

    client_max_body_size 2m;

    location = / {
        return 302 /admin;
    }

    location /admin {
        proxy_pass http://127.0.0.1:8080;
        proxy_http_version 1.1;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
}
NGINX
  rm -f /etc/nginx/sites-enabled/default
  ln -sf /etc/nginx/sites-available/tg-agent /etc/nginx/sites-enabled/tg-agent
  nginx -t >/dev/null
  systemctl enable --now nginx >/dev/null
  systemctl reload nginx
  systemctl is-active nginx
fi

echo "--- disk after cleanup ---"
df -h / | tail -1
REMOTE

echo "==> done"

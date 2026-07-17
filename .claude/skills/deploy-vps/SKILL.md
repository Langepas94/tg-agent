---
name: deploy-vps
description: Deploy tg-agent to the production VPS safely and verify the result. Use when the user asks to deploy, redeploy, ship to server, restart the bot on the VPS, or after merging changes that must go live.
---

# Deploy tg-agent to VPS

## Preconditions

- `cargo build && cargo test` green locally. Never deploy a red tree.
- Deploy from `main` after merge (branch policy: feature branch → merge).
- Version bumped in Cargo.toml for user-visible changes.

## Deploy

```bash
cd ~/Documents/tg-agent && ./deploy.sh
```

`deploy.sh` = rsync source (excludes .env/state.json/sessions) → remote
`cargo build --release` → `systemctl restart tg-agent` → prune build
artifacts + vacuum journal (14GB disk!) → nginx /admin proxy. SSH key is passed
through `SSH_KEY`; its canonical local path is
`$HOME/Documents/ai/.ssh/timeweb_tg_agent_ed25519`, and the host defaults to
`root@5.129.234.9`.

## FOOTGUN — never do this

**NEVER append `tg-agent --version` (or any bare binary invocation) to a
deploy command.** The binary ignores the flag and boots a FULL SECOND BOT →
endless `TerminatedByOtherGetUpdates`, the ghost old binary answers Telegram
while the fresh systemd instance is starved. Looks exactly like "deploy had
no effect".

## Verify after every deploy (mandatory)

```bash
ssh -i "$HOME/Documents/ai/.ssh/timeweb_tg_agent_ed25519" root@5.129.234.9 \
  'pgrep -af tg-agent | grep -v pgrep; \
   grep -m1 "^version" /opt/tg-agent/Cargo.toml; \
   journalctl -u tg-agent --since "-3 min" | grep -ci TerminatedByOtherGetUpdates; \
   journalctl -u tg-agent -n 3 --no-pager'
```

Expect: exactly ONE pid, the new version, `0` conflicts, `Dispatcher
starting` in the log tail. Stray process → `kill -9 <pid>`, then the killed
long-poll keeps erroring up to ~50s before it clears — do not panic-restart.

## Server layout

- Bot: `/opt/tg-agent` (env: `/opt/tg-agent/.env`, systemd `tg-agent`).
- Env changes require `systemctl restart tg-agent`; `deploy.sh` does not replace `.env`.

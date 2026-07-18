# Configuration and operations

## Required environment

`TELEGRAM_BOT_TOKEN` is required. For real natural-language answers and trip
planning, configure `LLM_API_KEY` or `DEEPSEEK_API_KEY`.

Recommended production variables:

```text
TELEGRAM_BOT_TOKEN=<telegram token>
LLM_API_KEY=<provider key>
LLM_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-chat
BOT_PASSWORD=<unique telegram access password>
DIGEST_INTERVAL_MINUTES=360
STATE_FILE=state.json
SESSIONS_DIR=sessions
RUST_LOG=tg_agent=info,rmcp=info
```

The source-code fallback for `BOT_PASSWORD` is intended only for local
development. Set a unique value in every real deployment.

## Optional web admin

The admin server starts only when `ADMIN_PASSWORD` is present. Its password
must differ from `BOT_PASSWORD`.

```text
ADMIN_ADDR=127.0.0.1:8082
ADMIN_USERNAME=admin
ADMIN_PASSWORD=<unique admin password>
```

Choose a loopback port that is not already used by nginx or another local
service. Configure nginx to proxy `/admin` to that exact address. If the admin
is intentionally disabled, do not treat a missing admin listener as a bot
failure.

## MCP configuration

MCP servers are not configured through static `MCP_*` environment variables.
The root user connects them at runtime:

```text
/connect https://example.com/mcp name=weather
/connect stdio npx -y package-name name=maps
/mcps
/tools
```

Connection parameters are persisted in `STATE_FILE` and restored after a
restart. Credentials supplied in headers or stdio environment entries must not
be printed in logs or copied into documentation.

## Local verification

```bash
cargo fmt --all -- --check
cargo test
cargo build --release
python3 -m unittest discover -s .github/scripts -p 'test_*.py'
git diff --check
```

Ignored integration tests require real MCP and LLM services:

```bash
cargo test --test live_trip_flow -- --ignored --nocapture --test-threads=1
cargo test --test live_orchestrator -- --ignored --nocapture --test-threads=1
```

## Deployment

The production target used by this project is `/opt/tg-agent` on Timeweb Cloud,
managed by `tg-agent.service`. The required weather service is
`open-meteo-mcp.service`.

```bash
ENABLE_NGINX_PROXY=0 ./deploy.sh
```

The canonical Timeweb private key for this workstation is
`$HOME/Documents/ai/.ssh/timeweb_tg_agent_ed25519`. `deploy.sh` uses this path
by default. Override `SSH_KEY` only when intentionally deploying from another
workstation or with a rotated key.

Use `ENABLE_NGINX_PROXY=0` when nginx is already shared with the support
assistant or another explicitly retained endpoint. Otherwise review the nginx
configuration before deployment so an existing `/support/` route is not
overwritten.

After deployment verify:

```bash
systemctl is-active tg-agent
systemctl is-active open-meteo-mcp
systemctl show tg-agent -p MainPID -p ActiveEnterTimestamp
grep -m1 '^version' /opt/tg-agent/Cargo.toml
journalctl -u tg-agent -n 100 --no-pager
```

Exactly one bot process must be active. Naming, local RAG and Ollama services
are outside the current product and must remain disabled.

# tg-agent

Telegram AI agent that connects to **MCP servers** at runtime, answers in
natural language using their tools, and runs periodic jobs 24/7.

## Features

- **Runtime MCP management** — `/connect`, `/mcps`, `/tools`, `/call`,
  `/disconnect`. Two transports:
  - **HTTP**: `/connect <url> [name= auth= Header:Value]` — remote
    Streamable-HTTP servers, per-server credentials.
  - **stdio**: `/connect stdio <program> [args...] [name=N] [env=KEY=VAL ...]` —
    spawns a local child process (npx/uvx servers, no HTTP bridge needed),
    e.g. `/connect stdio npx -y @cocal/google-calendar-mcp name=gcal`.
- **Natural-language agent** — free-text questions go through an LLM
  (OpenAI-compatible, DeepSeek by default) tool-calling loop over the connected
  MCP tools.
- **Agent self-connect** — the agent can attach MCP servers on its own via the
  `mcp_connect` / `mcp_disconnect` meta-tools: when a request needs a capability
  no connected server provides, it picks the server, asks the user for any
  credentials in chat, connects (HTTP or stdio), and the new tools become
  callable in the same turn. No curated list, no platform assumptions.
- **Periodic summaries** — `/watch <server> <tool> <minutes> [json]` polls a
  tool on a schedule and posts the result. The agent can also subscribe the user
  itself via the `schedule_summary` meta-tool ("collect weather hourly and keep
  me posted").
- **Agent runtime** (ported from the ai-playground project):
  - layered **sticky-facts memory** (short-term / working / long-term)
  - editable **user profile** + interview extraction
  - **extra info** (`/info`) — free-form labelled preferences a **router agent**
    mixes into the prompt only when relevant to the turn (e.g. a file-format
    note is injected when you ask for a document, ignored otherwise)
  - **invariants** checked in code (Pass/Fail/Advisory) and injected into the
    system prompt
  - layered **PromptBuilder**
  - multi-agent **travel-weather flow** (Planning → Execution → Validation → Done)
- **Persistence** — connected servers, subscribers, watches and per-chat
  sessions survive restarts.

## Configure

Copy `.env.example` to `.env`:

```
TELEGRAM_BOT_TOKEN=...
DEEPSEEK_API_KEY=...        # or LLM_API_KEY (OpenAI-compatible)
LLM_MODEL=deepseek-v4-flash
BOT_PASSWORD=202020         # Telegram /start password
ADMIN_ADDR=127.0.0.1:8080   # web admin bind; put nginx in front on VPS
ADMIN_USERNAME=admin
ADMIN_PASSWORD=...          # defaults to BOT_PASSWORD if omitted
DIGEST_INTERVAL_MINUTES=360
STATE_FILE=state.json
SESSIONS_DIR=sessions
```

## Web admin

The bot starts a small root admin UI at `http://ADMIN_ADDR/admin`. By default
the Rust process binds to `127.0.0.1:8080`, and `deploy.sh` exposes it through
nginx as `http://5.129.234.9/admin`.

The UI lets the owner inspect users, profile fields, notes, sticky facts,
compacted summary, recent messages, watches, push subscriptions, raw session
JSON, and manage access/context/profile/notes.

## Run

```bash
cargo run --release
```

## Test

```bash
cargo test                                   # unit tests
cargo test -- --ignored --nocapture          # live tests (need MCP + LLM key)
```

## Commands

`/start` `/help` `/connect` `/mcps` `/tools` `/call` `/watch` `/unwatch`
`/watches` `/disconnect` `/profile` `/info` `/facts` `/trip` `/reset`

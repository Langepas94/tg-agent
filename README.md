# tg-agent

Telegram AI agent that connects to **MCP servers** at runtime, answers in
natural language using their tools, and plans outdoor leisure around weather,
location, dates and user constraints.

The core flow is: collect only missing constraints → inspect weather and places
→ compare suitable activity-and-place options → ask the user to choose → build
and verify a concrete route → create only explicitly requested artifacts.

The production bot is intentionally separate from the developer and support
assistants. RAG over this repository is implemented by the standalone
`project-assistant`; document retrieval is not part of the Telegram runtime.

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
- **User support** — `/support <question>` sends the product question and the
  authenticated Telegram user ID to the isolated support service. Its read-only
  JSON CRM MCP resolves that user's active ticket before RAG answering.
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
  - dynamic multi-agent **trip-planning swarm** — a planner LLM builds the task
    graph from the live MCP tool inventory; each agent (Brief / Options / Planner
    / Worker / Verifier / Artifacts / Final) is a separate entity with its own
    role, permissions and (optionally) its own model via `SWARM_MODEL_<AGENT>`
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
ADMIN_PASSWORD=...          # required for web admin; must differ from BOT_PASSWORD
DIGEST_INTERVAL_MINUTES=360
STATE_FILE=state.json
SESSIONS_DIR=sessions
```

## Web admin

When enabled, the bot starts a small root admin UI at
`http://ADMIN_ADDR/admin`. The source-code default bind is
`127.0.0.1:8080`; choose another loopback port when nginx or a retained service
already owns that port, then point the `/admin` proxy to the same address.

The UI lets the owner inspect users, profile fields, notes, sticky facts,
compacted summary, recent messages, watches, push subscriptions, raw session
JSON, and manage access/context/profile/notes.

`/admin` is disabled unless `ADMIN_PASSWORD` is set, and the admin password must
be different from `BOT_PASSWORD`.

## Run

```bash
cargo run --release
```

Before a production run, set a non-default `BOT_PASSWORD`. Add
`ADMIN_PASSWORD` only when the optional web admin must be enabled, and keep it
different from the Telegram password.

## Test

```bash
cargo test                                   # unit tests
cargo test -- --ignored --nocapture          # live tests (need MCP + LLM key)
```

## Commands

`/start` `/help` `/connect` `/mcps` `/tools` `/call` `/watch` `/unwatch`
`/watches` `/disconnect` `/profile` `/info` `/facts` `/trip` `/compact` `/reset`

## Documentation

- [User guide](docs/user/README.md) — first launch, the canonical trip flow,
  commands, FAQ and safe support instructions. Support assistants treat this
  section as the authoritative product source.
- [Architecture](docs/architecture.md) — runtime boundaries, components and
  the verified outdoor-planning sequence.
- [Configuration and operations](docs/configuration.md) — environment,
  security, persistence and deployment checks.
- [Troubleshooting](docs/troubleshooting.md) — operator checks for Telegram,
  authorization, MCP, LLM and web endpoints after the user guide is exhausted.
- [Demonstration guide](docs/demo.md) — reproducible scenarios for the bot,
  developer assistant, support assistant and AI review.
- [Homework readiness report](docs/homework-readiness.md) — assignment matrix,
  evidence and remaining publication risk.
- [Automatic AI review](docs/ai-review.md) — GitHub Actions design, safety and
  local verification.

## Required verification

```bash
cargo fmt --all -- --check
cargo test
cargo build --release
python3 -m unittest discover -s .github/scripts -p 'test_*.py'
git diff --check
```

# tg-agent

Telegram AI agent that connects to **MCP servers** at runtime, answers in
natural language using their tools, and runs periodic jobs 24/7.

## Features

- **Runtime MCP management** — `/connect <url> [name= auth= Header:Value]`,
  `/mcps`, `/tools`, `/call`, `/disconnect`. Streamable-HTTP transport with
  per-server credentials.
- **Natural-language agent** — free-text questions go through an LLM
  (OpenAI-compatible, DeepSeek by default) tool-calling loop over the connected
  MCP tools.
- **Periodic summaries** — `/watch <server> <tool> <minutes> [json]` polls a
  tool on a schedule and posts the result. The agent can also subscribe the user
  itself via the `schedule_summary` meta-tool ("collect weather hourly and keep
  me posted").
- **Agent runtime** (ported from the ai-playground project):
  - layered **sticky-facts memory** (short-term / working / long-term)
  - editable **user profile** + interview extraction
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
DIGEST_INTERVAL_MINUTES=360
STATE_FILE=state.json
SESSIONS_DIR=sessions
```

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
`/watches` `/disconnect` `/profile` `/facts` `/trip` `/reset`

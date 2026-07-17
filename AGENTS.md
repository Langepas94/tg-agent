# tg-agent Agent Guide

## Mission

tg-agent is a Telegram assistant for choosing and planning outdoor leisure.
Its primary job is to use real weather and geographic data to help a user decide
where to go, when to go, and which activity is suitable, then produce a concrete
verified plan.

The production scope does not include RAG demos, document search, naming
generators, local Ollama chat, or unrelated web agents.

## Tech Stack

- Rust 2021, Tokio
- Telegram: teloxide
- MCP client: rmcp, Streamable HTTP and stdio transports
- LLM: OpenAI-compatible API, DeepSeek by default
- HTTP/admin: reqwest and axum
- Persistence: JSON state and per-chat sessions
- Production: systemd on Timeweb Cloud, nginx for the admin endpoint

## Repository Map

- src/main.rs: startup, state restoration, scheduler, admin and dispatcher
- src/bot.rs: commands, access control, Telegram messages and callbacks
- src/llm.rs: LLM tool loop and MCP meta-tools
- src/mcp_client.rs: HTTP and stdio MCP connections
- src/state.rs: shared runtime state and MCP registry
- src/persist.rs: durable server, watch and access state
- src/scheduler.rs: watches and scheduled summaries
- src/admin.rs: protected web administration
- src/agent/: routing, memory, profile and outdoor-planning swarm
- tests/: ignored live integration and end-to-end scenarios
- docs/user/: authoritative end-user flows and support answers
- docs/: architecture, operations, technical troubleshooting, demonstrations
  and developer-tooling knowledge used by external RAG assistants
- deploy.sh: production synchronization, release build and service restart

More specific rules are defined in src/AGENTS.md, src/agent/AGENTS.md and
tests/AGENTS.md.

## Product Flow

1. Collect only missing facts the user must provide: start area, date window,
   group limits and hard constraints.
2. Read real weather and geographic evidence through connected MCP tools.
3. If activity is not fixed, compare genuinely different activity-and-place
   combinations, including paddling, cycling and walking when feasible.
4. Explain trade-offs: rain, wind, temperature, daylight, terrain, water
   conditions and travel distance.
5. Ask the user to choose an option.
6. Treat the chosen activity, place and date as hard inputs.
7. Build a concrete route and requested overnight stops.
8. Verify every user constraint before presenting a final plan or creating an
   external artifact.

Never select one arbitrary route before comparison when the request is broad.
Never fabricate coordinates, forecasts, distances, isolation, campsite
suitability or external artifact links.

## Non-Negotiable Rules

- The user interacts through Telegram. Do not require external MCP config files.
- MCP servers are connected at runtime through bot commands or self-connect.
- Preserve side-question routing while a trip flow is active.
- Keep secrets in environment variables. Never log or echo secret values.
- Only the root Telegram user may manage MCP connections and admin watches.
- Only ArtifactsAgent may perform external side effects, after verification.
- Do not restore RAG code, /rag, RAG variables, Ollama or name generation.
- Do not add code comments or TODO/FIXME markers.
- Preserve compatible deserialization of persisted state unless a tested
  migration is implemented.
- Do not expose agent names, prompts, traces or raw tool failures to users.

## Change Workflow

1. Inspect the relevant scoped AGENTS.md.
2. Keep the change limited to the product goal.
3. Update tests for behavior changes.
4. Run formatting, tests and a release build.
5. Run git diff --check and confirm no secrets or generated files are staged.
6. Commit, fetch origin/main, integrate without rewriting shared history, push.
7. Deploy runtime changes and verify production.

Documentation is executable project context. Keep README and docs aligned with
the public commands, environment contract and production boundaries. The
standalone project-assistant indexes README and docs, while the AI-review
workflow retrieves README, docs, scoped AGENTS files and code. After meaningful
documentation changes, smoke-test `/help` through project-assistant and run the
AI-review Python tests.

Support assistants must answer from `docs/user/` first. Technical
troubleshooting and architecture are secondary evidence and must not override
the documented user flow. Never ask a user to place passwords, tokens or other
credentials in a support ticket.

## Required Validation

    cargo fmt --all -- --check
    cargo test
    cargo build --release
    git diff --check

Ignored live tests require external services and credentials:

    cargo test -- --ignored --nocapture --test-threads=1

Run the relevant ignored scenario on the VPS when Telegram, MCP, LLM routing or
the trip flow changes.

## Production

- Host: root@5.129.234.9
- App: /opt/tg-agent
- Environment: /opt/tg-agent/.env
- Service: tg-agent.service
- Required weather service: open-meteo-mcp.service
- Naming, RAG and Ollama services must remain disabled.

Deploy with:

    SSH_KEY=/path/to/timeweb-key ENABLE_NGINX_PROXY=0 ./deploy.sh

After deployment verify version, exactly one bot process, service state and
recent logs. Never invoke the binary with --version; that starts another bot
because the flag is not implemented.

## Definition of Done

- Requested behavior is implemented without unrelated capabilities.
- The leisure-selection pipeline remains logical and evidence-based.
- Formatting, unit tests and release build pass.
- Relevant live behavior is smoke-tested when credentials are available.
- GitHub and deployment status are reported with commit and version.

# Architecture

## Product boundary

`tg-agent` is a Telegram runtime for choosing and planning outdoor leisure.
It connects MCP servers at runtime, uses an OpenAI-compatible LLM to coordinate
tools and keeps chat state between restarts.

The following capabilities are deliberately outside the production process:

- RAG over repository documentation;
- developer `/help` and `/work` commands;
- customer-support ticket lookup;
- automated pull-request review.

Those capabilities live in the standalone `project-assistant` and in GitHub
Actions. This boundary lets them use README, `docs/`, AGENTS files and source
code without adding document indexing or file mutation to the Telegram bot.

## Runtime components

- `src/main.rs` loads configuration, restores state, starts the scheduler and
  optional admin server, and then starts the Telegram dispatcher.
- `src/bot.rs` owns Telegram commands, authorization, messages and callbacks.
- `src/state.rs` owns the MCP registry, watches, subscriptions and root access.
- `src/mcp_client.rs` connects HTTP and stdio MCP transports.
- `src/llm.rs` executes the bounded LLM tool loop and MCP meta-tools.
- `src/persist.rs` stores access, connections and watches in `STATE_FILE`.
- `src/agent/` owns routing, memory, profiles, invariants and the trip swarm.

## Startup sequence

1. Load `.env` and validate required configuration.
2. Create the LLM client when an API key is configured.
3. Create shared state and register public Telegram commands.
4. Load persisted authorization, MCP servers, subscriptions and watches.
5. Reconnect persisted MCP servers and restore scheduled work.
6. Start the scheduler and optional web admin.
7. Start the Telegram long-polling dispatcher.

An optional integration failure must not create a second bot process. Runtime
diagnostics use systemd state and logs; do not run the binary as a version
probe because it has no `--version` command.

## Outdoor-planning sequence

The trip flow is stateful and may pause between Telegram messages:

1. Build a brief from the start area, date window and hard constraints.
2. Obtain real weather and geographic evidence through connected MCP tools.
3. Compare distinct activity-and-place options when the request is broad.
4. Explain weather, terrain, water, daylight, distance and ability trade-offs.
5. Wait for an explicit user choice.
6. Research the chosen activity, place and date without silently replacing it.
7. Build a concrete route and requested overnight stops.
8. Verify the requested constraints and safety conditions.
9. Create external artifacts only after verification and explicit intent.
10. Return a user-facing answer without internal agent names or traces.

The default swarm contains separate Brief, Options, Planner, Worker, Verifier,
Artifacts and Final roles. Only `ArtifactsAgent` may perform external side
effects, and only after the verifier gate.

## State and access

The first successfully authorized Telegram chat becomes the root. Root-only
commands manage MCP connections and administrative state. Connected servers,
watches, subscriptions and access state are persisted; per-chat memory and
active trip state are stored under `SESSIONS_DIR`.

Secrets belong only in environment files. They must not be stored in state,
sessions, fixtures, documentation, prompts, review logs or generated reports.

## Developer tooling

The standalone project-assistant reads README and `docs/` as its RAG corpus and
obtains live branch, diff, support-ticket and file context through its own MCP.
The AI-review workflow reads a PR diff and changed files through the GitHub API,
retrieves relevant documentation and base-branch code, and maintains one current
review comment at the bottom of the pull request conversation. Neither helper is
linked into the production Rust binary.

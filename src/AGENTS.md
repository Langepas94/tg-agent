# Runtime Module Guide

## Scope

This directory contains the Telegram runtime, MCP integration, LLM tool loop,
configuration, persistence, scheduling and administration. The root AGENTS.md
applies in full.

## Module Boundaries

- main.rs wires components and restores state. Keep business decisions out.
- config.rs parses env variables and validates unsafe combinations.
- bot.rs owns commands, access checks, user responses and message chunking.
- llm.rs owns the LLM protocol, tool-call loop and MCP meta-tools.
- mcp_client.rs owns transport setup and connection lifecycle.
- state.rs owns shared registries and authorization state.
- persist.rs owns stable on-disk formats and persistence behavior.
- scheduler.rs owns recurring watches and push summaries.
- admin.rs owns the authenticated admin surface.

## Runtime Contracts

- Perform access checks before processing that reveals state or causes effects.
- Keep Telegram messages below platform limits and avoid empty sends.
- Preserve typing and progress behavior on long LLM or MCP calls.
- Namespace MCP tools by server and handle collisions deterministically.
- Do not hold async locks across network, LLM, Telegram or child-process calls.
- Reconnect persisted servers and watches without duplicate processes.
- Treat headers, bearer tokens, stdio env and LLM keys as secrets.
- Reject invalid JSON and connection parameters before MCP calls.
- Optional integration failure must not prevent dispatcher startup.
- User errors are concise; detailed cause chains belong in logs.

## MCP and Side Effects

- Runtime inventory is the source of truth. Do not assume a tool exists.
- Regular users may not connect, disconnect or administer MCP servers.
- External artifacts follow the verifier gate in agent/AGENTS.md.
- Do not add a second bot process for diagnostics.

## Validation

    cargo fmt --all -- --check
    cargo test
    cargo build --release

For MCP lifecycle changes also run the relevant ignored tests in
tests/live_connect.rs, tests/live_autoconnect.rs or tests/live_orchestrator.rs.

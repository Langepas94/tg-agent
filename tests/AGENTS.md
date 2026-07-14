# Integration Test Guide

## Scope

This directory contains live tests against real LLM and MCP services. Tests are
ignored by default so normal unit runs remain deterministic.

## Test Matrix

- live_agent.rs: natural-language weather answer through MCP
- live_connect.rs: MCP connection and registry lifecycle
- live_autoconnect.rs: self-connect followed by tool use
- live_orchestrator.rs: subscriptions, profile, invariants and trip control
- live_trip_flow.rs: kayak, cycling, walking and optional Google artifacts

## Rules

- Keep ordinary cargo test free of network and secret requirements.
- Mark live tests ignored and document required environment variables.
- Never print tokens, passwords, auth headers or full environment dumps.
- Use unique temporary state and session paths.
- Assert user-visible outcomes, not private prompts or incidental prose.
- Outdoor scenarios cover both water and non-water activities.
- A broad activity-choice scenario demonstrates weather comparison before route
  construction.
- Final answers contain concrete dates and places and do not expose agent names,
  markdown tables or unresolved constraints.
- Optional integrations may skip only when credentials are absent; core weather
  and map failures fail the relevant scenario.

## Commands

    cargo test
    cargo test --test live_trip_flow -- --ignored --nocapture --test-threads=1
    cargo test --test live_orchestrator -- --ignored --nocapture --test-threads=1

Run live tests on the VPS when local access to production-equivalent MCP
services is unavailable.

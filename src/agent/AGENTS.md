# Outdoor Planning Agent Guide

## Scope

This directory contains semantic routing, session memory, profile extraction,
prompt construction, invariants and the stateful outdoor-planning swarm.

## Responsibilities

- router.rs classifies new messages and protects an active trip from side chat.
- flow.rs builds the brief, options, choice, plan, verification and final answer.
- session.rs persists chat memory and active trip state compatibly.
- memory.rs, profile.rs and notes.rs retain useful non-sensitive context.
- prompt.rs assembles bounded prompts without leaking internal state.
- invariants.rs enforces response constraints in code.
- context_budget.rs keeps prompts within model context limits.

## Required Trip Sequence

    brief
      -> weather and place evidence
      -> activity-and-place options
      -> explicit user choice
      -> selected-option research
      -> constraint verification
      -> optional artifacts
      -> final answer

The flow may pause across Telegram turns but must not skip the explicit choice
checkpoint for broad requests.

## Brief Rules

- Ask only questions whose answers cannot be discovered through tools.
- Minimum input is a start area and date window.
- Reuse known profile facts instead of asking again.
- Preserve the original request verbatim in flow state.
- If the user asks the bot to choose the activity, record that objective and do
  not force the choice during clarification.
- Stop clarifying after the bounded round limit and state assumptions.

## Option Rules

- Use connected weather and geographic tools before recommending.
- If activity is open, compare at least two distinct feasible activity-and-place
  options when data supports them.
- Evaluate precipitation, wind in m/s, temperature, daylight, terrain, water,
  group ability and travel limits.
- For a fixed activity, preserve it and compare places or route variants.
- For a broad area, do not collapse alternatives into one place.
- Place options use real names and coordinates from tool evidence.
- Exclude unsafe options instead of merely ranking them lower.
- End with a clear request for the user to choose.

## Planning and Verification Rules

- Treat selected activity, place and date as fixed unless the user changes them.
- Give each worker one narrow task and only the context it needs.
- Workers return concrete findings, not narration about future work.
- Missing capability or incomplete evidence must be explicit.
- Verify only constraints the user actually requested.
- Do not mark a plan ready with unresolved requested details or safety.
- Do not create artifacts from draft or unverified records.
- Only ArtifactsAgent can create external artifacts.
- Final responses hide swarm internals and preserve concrete evidence.

## Memory and Privacy

- Do not persist secrets, transient weather or noisy task output as facts.
- Keep working memory isolated between workers.
- Preserve long-term profile facts through compact and reset behavior.
- Maintain serde compatibility for existing sessions.

## Validation

Every behavior change needs a focused unit test.

    cargo test agent::
    cargo test
    cargo build --release

Routing or trip sequence changes also require the relevant ignored scenario in
tests/live_trip_flow.rs or tests/live_orchestrator.rs.

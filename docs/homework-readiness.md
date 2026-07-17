# Homework readiness report

Audit date: 2026-07-17.

## Scope collected from neighboring tasks

The audit found these assignments in the `ai` and `project-assistant` tasks:

1. restore the travel Telegram agent and its logical weather-based activity
   selection pipeline;
2. build a standalone `/help` assistant with README/docs RAG and git context
   through MCP;
3. add automatic AI review for pull requests using diff, changed files and RAG;
4. build a support assistant using product docs and JSON ticket/user context
   through MCP;
5. extend the assistant with repeatable multi-file read, search, write, diff and
   report workflows.

## Readiness matrix

| Assignment | Implementation | Verified evidence | Demo status |
| --- | --- | --- | --- |
| Travel bot | `tg-agent` 0.15.0, runtime MCP, stateful trip swarm | 138 Rust tests, release build, production services active | Ready for Telegram smoke; full live route still depends on Telegram, LLM and MCP availability |
| Project `/help` | Standalone `project-assistant`, README/docs RAG, MCP branch/files/diff | 15 assistant tests, release build, one-shot `/help` smoke | Ready locally; new docs improve source precision |
| AI PR review | GitHub Action, GitHub API diff, RAG, one upserted comment | 8 Python tests and successful GitHub Actions smoke run | Ready; verify automatic trigger on the next documentation PR |
| Support assistant | `/support`, JSON users/tickets, MCP join, RAG, CLI and web UI | `TICKET-1001 → USER-42`, public page and health `200`, unauthenticated API `401` | Ready on VPS; real LLM answer depends on provider availability |
| File assistant | `/work` search and deterministic change report, bounded MCP file tools | Multi-file search, disposable-clone report smoke and 15 tests | Ready locally; demonstrate writes only in a disposable clone |

## Validation performed

- `tg-agent`: formatting, 138 unit tests, release build and `git diff --check`.
- AI review: 8 deterministic Python tests.
- `project-assistant`: formatting, 15 tests and release build.
- `/help`, `/support TICKET-1001` and `/work` search executed through the release
  assistant against `tg-agent`.
- The change-report scenario was run twice in a disposable clone; the second
  run returned `No changes` after excluding `.assistant/` from MCP git status.
- GitHub `main` contains merged PRs #1 and #2 for AI review.
- The latest manual `AI code review` workflow completed successfully after the
  48,000-character context limit fix.
- Production `tg-agent.service`, `open-meteo-mcp.service` and
  `project-assistant-support.service` are active.
- Public support page and health return `200`; protected ticket context returns
  `401` without the operator password.
- `name-gen` and `ollama` services are disabled and inactive.

## Remaining risks

### Project-assistant publication

The support, web and file-operation work exists in the local
`project-assistant` checkout as uncommitted changes on top of initial commit
`eb861fe`. Its GitHub `main` therefore cannot reproduce the demonstrated state
yet. This is the only blocker to calling every homework artifact recoverable
and ready for use on another machine.

Required closure: review the existing local diff, remove generated files such
as `.DS_Store` and local reports, commit the intended sources, push them to the
standalone `Langepas94/project-assistant` repository and rerun its CI.

### Optional admin endpoint

The Telegram bot does not require the web admin for these assignments. On the
audited production host `ADMIN_PASSWORD` is absent, so the Rust admin listener
is intentionally disabled. An old nginx route can return `502`; do not include
`/admin` in the demonstration until a unique admin password, free loopback port
and matching nginx upstream are configured.

### External dependencies

Live Telegram, LLM, weather, GitHub Models and Sites behavior can fail outside
the codebase. Run the pre-demo smoke checklist immediately before presenting
and keep the CLI evidence mode available when an LLM provider is unavailable.

## Conclusion

All five assignments have working local or deployed implementations and
repeatable demonstrations. The `tg-agent` repository and AI-review pipeline are
reproducible from GitHub. The support endpoint is live. The remaining release
gap is publication of the accumulated `project-assistant` changes; until that
commit and push are completed, the local assistant is demonstrable but not yet
fully recoverable for real use on a clean machine.

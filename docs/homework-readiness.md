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
| Project `/help` | Standalone `project-assistant`, README/docs RAG, MCP branch/files/diff | 16 assistant tests, release build, one-shot `/help` smoke | Ready in project-assistant PR #1 |
| AI PR review | GitHub Action, GitHub API diff, RAG, one upserted comment | 8 Python tests and successful GitHub Actions smoke run | Ready; verify automatic trigger on the next documentation PR |
| Support assistant | `/support`, JSON users/tickets, MCP join, user-first RAG, CLI and web UI | `TICKET-1001 → USER-42`, `docs/user/` first, public page and health `200`, unauthenticated API `401` | Ready in PR #1 and on VPS; deployed process must be updated after merge |
| File assistant | `/work` search and deterministic change report, bounded MCP file tools | Multi-file search, disposable-clone report smoke and 16 tests | Ready in project-assistant PR #1; demonstrate writes only in a disposable clone |

## Validation performed

- `tg-agent`: formatting, 138 unit tests, release build and `git diff --check`.
- AI review: 9 deterministic Python tests, including recursive `docs/user/`
  discovery in the knowledge corpus.
- `project-assistant`: formatting, 16 tests and release build.
- `/help`, `/support TICKET-1001` and `/work` search executed through the release
  assistant against `tg-agent`.
- The change-report scenario was run twice in a disposable clone; the second
  run returned `No changes` after excluding `.assistant/` from MCP git status.
- GitHub `main` contains merged PRs #1 and #2 for AI review.
- The latest manual `AI code review` workflow completed successfully; the
  documentation PR exposed a smaller GitHub Models gateway limit and now uses
  a dedicated 20,000-character provider budget.
- Production `tg-agent.service`, `open-meteo-mcp.service` and
  `project-assistant-support.service` are active.
- Public support page and health return `200`; protected ticket context returns
  `401` without the operator password.
- `name-gen` and `ollama` services are disabled and inactive.

## Remaining risks

### Pending merges and support deployment

The complete assistant suite is published in
`Langepas94/project-assistant` pull request #1 and its GitHub CI passes. The
expanded `tg-agent` documentation is published in documentation pull request
#3. Both changes remain reproducible, but a clean checkout of `main` will not
contain them until the pull requests are merged.

After both merges, rebuild and restart the support service so the web assistant
uses the user-first retrieval behavior and the new `docs/user/` knowledge.

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

All five assignments have working implementations and repeatable
demonstrations. Their latest changes are published in open pull requests. The
support endpoint is live; merging both pull requests and redeploying the
support service are the remaining steps before the user-first support flow is
the default on a clean checkout and in production.

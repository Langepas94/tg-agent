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
| Project `/help` | Standalone `project-assistant`, README/docs RAG, MCP branch/files/diff | 16 assistant tests, release build, one-shot `/help` smoke | Merged to project-assistant `main` and ready locally |
| AI PR review | GitHub Action, GitHub API diff, RAG, one upserted comment | 9 Python tests and successful automatic review of documentation PR #3 | Merged to `main` and verified on a real pull request |
| Support assistant | `/support`, JSON users/tickets, MCP join, user-first RAG, CLI and web UI | `TICKET-1001 → USER-42`, `docs/user/` first, public page and health `200`, unauthenticated API `401` | Merged to `main`; production endpoint is available, binary update is pending VPS access |
| File assistant | `/work` search and deterministic change report, bounded MCP file tools | Multi-file search, disposable-clone report smoke and 16 tests | Merged to project-assistant `main`; demonstrate writes only in a disposable clone |

## Validation performed

- `tg-agent`: formatting, 138 unit tests, release build and `git diff --check`.
- AI review: 9 deterministic Python tests, including recursive `docs/user/`
  discovery in the knowledge corpus.
- `project-assistant`: formatting, 16 tests and release build.
- `/help`, `/support TICKET-1001` and `/work` search executed through the release
  assistant against `tg-agent`.
- The change-report scenario was run twice in a disposable clone; the second
  run returned `No changes` after excluding `.assistant/` from MCP git status.
- GitHub `main` contains merged AI-review PRs #1, #2 and #4 plus documentation
  PR #3. Project-assistant PR #1 is also merged to its `main`.
- Automatic `AI code review` completed successfully on documentation PR #3
  after applying the dedicated 20,000-character GitHub Models budget.
- Production `tg-agent.service`, `open-meteo-mcp.service` and
  `project-assistant-support.service` are active.
- Public support page and health return `200`; protected ticket context returns
  `401` without the operator password.
- `name-gen` and `ollama` services are disabled and inactive.

## Remaining risks

### Support deployment access

All required pull requests are merged. The public support page and health check
return `200`, and an unauthenticated ticket request returns `401`. Updating the
running support binary requires the Timeweb SSH private key configured by the
deployment environment. The expected local path is
`~/.ssh/id_ed25519_vps`; it was not available during the final audit.

After restoring VPS access, rebuild and restart
`project-assistant-support.service`, then verify that `TICKET-1001` returns a
response whose first source is `docs/user/getting-started.md` or
`docs/user/support-and-faq.md`.

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

All five assignments have working implementations, repeatable demonstrations
and merged source code. A clean checkout of both repositories reproduces the
verified local state. The only remaining production action is restarting the
support service with the merged binary after VPS SSH access is restored.

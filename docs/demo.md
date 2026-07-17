# Demonstration guide

This guide covers the Telegram product and the four AI homework scenarios found
in the neighboring `ai` and `project-assistant` tasks.

## Before the demo

1. Confirm `tg-agent.service` and `open-meteo-mcp.service` are active.
2. Confirm exactly one Telegram bot process is polling.
3. Open the Telegram bot and the public support console in advance.
4. Build `project-assistant` with `cargo build --release`.
5. Keep the support password and all API keys out of screen sharing, shell
   history, tickets and slides.
6. Prepare a small pull request so automatic AI review can be shown without
   editing production code during the presentation.

## Scenario 1: outdoor trip planner

In Telegram, start a new trip with a broad request:

```text
/trip Подбери отдых на выходные из Москвы. Сравни байдарку, велосипед и пеший маршрут с учетом погоды.
```

Demonstrate that the bot asks only for missing date or group constraints, uses
weather and place evidence, offers multiple activity-and-place options, waits
for a choice, and only then builds and verifies a route.

Success criteria:

- no fabricated weather, coordinates or distances;
- at least two genuinely different options when feasible;
- explicit user choice before the final route;
- concrete date, place and safety trade-offs in the answer.

## Scenario 2: project `/help` with RAG and MCP

From the `project-assistant` repository:

```bash
cargo run --release -- \
  --project ../tg-agent \
  "/help Как устроен выбор поездки и кто может создавать внешние артефакты?"
```

Without an LLM key, the output still proves the MCP branch lookup and shows
retrieved sources. With a key, it returns a synthesized evidence-bound answer.
Expected sources include `docs/architecture.md` and scoped AGENTS guidance.

## Scenario 3: support assistant

CLI version:

```bash
cargo run --release -- \
  --project ../tg-agent \
  --support-data examples/travel-support-data.json \
  "/support TICKET-1001 Почему не работает авторизация?"
```

Web version: open `http://5.129.234.9/support/`, enter the operator password
outside screen sharing, load `TICKET-1001` and ask the same question.

Success criteria:

- ticket `TICKET-1001` is joined with `USER-42` through MCP;
- the first answer source is `docs/user/getting-started.md` or
  `docs/user/support-and-faq.md`, with technical docs used only as fallback;
- the answer never asks for or reveals a password;
- no password returns `401`;
- `/support/health` returns `200`.

## Scenario 4: file-working assistant

Read-only multi-file search:

```bash
cargo run --release -- \
  --project ../tg-agent \
  "/work Найди все места, где используется ArtifactsAgent"
```

Writable report in a disposable clone:

```bash
cargo run --release -- \
  --project /path/to/disposable/tg-agent \
  "/work Подготовь отчет об изменениях"
```

Run the report goal twice. The first run must create
`.assistant/change-report.md` with git status and diff; the second must return
`No changes`. Never demonstrate the write scenario against the only production
checkout.

## Scenario 5: automatic AI review

Open or update a small pull request. The `AI code review` workflow must:

1. run its Python tests;
2. obtain changed files and diff through the GitHub API;
3. retrieve relevant docs and base-branch code;
4. publish one comment with bugs, architecture issues and recommendations;
5. update the same comment after another push.

The workflow must not execute pull-request code with the privileged token.
The manual fallback is `workflow_dispatch` with the pull request number.

## Recommended presentation order

Use the Telegram trip flow first, then `/help`, support, file operations and
AI review. This moves from the real product to progressively more developer-
oriented automation and makes the production boundary clear.

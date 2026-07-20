# Demonstration guide

This guide covers the Telegram product and the four AI homework scenarios found
in the neighboring `ai` and `project-assistant` tasks.

## Before the demo

1. Confirm `tg-agent.service` and `open-meteo-mcp.service` are active.
2. Confirm exactly one Telegram bot process is polling.
3. Open the Telegram bot and confirm `/support` is visible in its command menu.
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

In the Telegram bot, send:

```text
/support Почему не работает авторизация?
```

The bot adds the authenticated Telegram user ID. The isolated support backend
uses its JSON CRM MCP to resolve that user's active ticket and linked profile,
then retrieves relevant FAQ and user documentation before answering.

Success criteria:

- no ticket ID, password or access key is entered by the Telegram user;
- the answer reflects the active ticket linked to the sender's Telegram ID;
- FAQ and product documentation supply the user-safe resolution steps;
- internal prompts, tokens, stack and infrastructure are never revealed;
- an unknown Telegram user does not receive another user's ticket context.

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

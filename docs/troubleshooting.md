# Troubleshooting

## Telegram `/start` rejects the password

Typical ticket text includes “не работает авторизация”, “пароль неверный” or
`AUTH_PASSWORD_REJECTED`.

1. Confirm that the user sent `/start <password>` to the expected bot.
2. Confirm that the production `BOT_PASSWORD` is set and does not contain
   accidental surrounding whitespace.
3. Do not request or paste the password into a ticket, log or AI prompt.
4. Check whether the chat is already authorized in persisted access state.
5. If the password was rotated, ask the user to retry with the new value through
   Telegram; never include it in the support response.
6. Inspect recent `tg-agent` logs for an authorization failure without dumping
   the environment file.

The support assistant should answer from this checklist and the ticket context.
It must not invent a password or disclose the configured value.

## Bot service is active but Telegram does not answer

1. Confirm `tg-agent.service` has one stable `MainPID`.
2. Inspect recent logs for Telegram timeouts, rate limits or another poller.
3. Confirm outbound access to `api.telegram.org`.
4. Verify the token with Telegram `getMe` without printing the token.
5. Do not start the binary manually while systemd is polling.

Short transient `RetryAfter` or network timeout records do not by themselves
prove an outage. Repeated current errors and a stale dispatcher require action.

## Natural-language requests do not work

The command-only mode is expected when no LLM key is configured. Confirm that
one of `LLM_API_KEY` or `DEEPSEEK_API_KEY` is present, then verify
`LLM_BASE_URL` and `LLM_MODEL`. Do not log the key.

## Weather or places are missing

Use `/mcps` and `/tools` as the root user. Confirm the weather and geographic
MCP servers are connected and return real data. The agent must report missing
capability instead of fabricating forecasts, coordinates or routes.

## Support console

- `GET /support/health` returns `200` when the service is ready.
- `GET /support/` returns the operator page.
- Ticket APIs return `401` without `x-support-password`.
- A ticket answer needs both ticket/user MCP context and relevant README/docs
  chunks before the LLM is called.

## Web admin

The admin listener is optional and is disabled when `ADMIN_PASSWORD` is absent.
If nginx returns `502`, compare its upstream with `ADMIN_ADDR` and confirm the
Rust process is listening on that exact loopback port.

## AI review

Run the deterministic tests locally first. A GitHub Models `413` means the
combined prompt is too large; the current implementation applies separate
budgets, a 20,000-character GitHub Models limit and a 48,000-character custom
provider limit. Use `workflow_dispatch` with a PR number for a production smoke
rerun.

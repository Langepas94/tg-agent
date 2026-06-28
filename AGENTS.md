# tg-agent — Codex context

Telegram-бот = **MCP-клиент + LLM-агент**. Юзер общается ТОЛЬКО через телеграм-чат.
MCP-серверы подключаются **в чате** (команды или self-connect агентом), НЕ через
внешние конфиги (Codex Desktop / VS Code `.mcp.json` — НЕ относятся к этому проекту).

## Что это

- Telegram-бот на `teloxide`. Подключает MCP-серверы в рантайме, отвечает на
  свободный текст через LLM tool-loop поверх их tools, гоняет периодическиеджобы.
- MCP-клиент: `rmcp` 1.8, два транспорта (HTTP Streamable + stdio child).
- Агент-рантайм портирован из ai-playground: sticky-facts память, профиль,
  инварианты, PromptBuilder, multi-agent travel-weather FSM.

## Карта кода

- `src/main.rs` — старт: env → Config → restore_state (реконнект MCP/watches) → dispatcher.
- `src/config.rs` — env-конфиг (см. ниже).
- `src/bot.rs` — команды (enum `Command`), парсинг `/connect` (`parse_connect`,
  `parse_connect_stdio`).
- `src/mcp_client.rs` — `ConnectParams`, `connect()` → `spawn_stdio` / `connect_http`.
- `src/llm.rs` — LLM tool-loop + meta-tools `mcp_connect` / `mcp_disconnect` /
  `schedule_summary`; defs и хендлеры здесь.
- `src/agent/flow.rs` — travel-weather FSM (Planning→Execution→Validation→Done), `/trip`.
- `src/agent/prompt.rs` — `BASE_SYSTEM` + `build_system_prompt` (слои промпта).
- `src/persist.rs`, `src/state.rs` — персист серверов/подписок/watch, общий стейт.
- `src/scheduler.rs` — периодические watch-джобы и push-summaries.

## Конфиг (env, `src/config.rs`)

- `TELEGRAM_BOT_TOKEN` — **обязателен**.
- `LLM_API_KEY` или `DEEPSEEK_API_KEY` — **без него free-text отключён**, работают
  только команды. Для агент-флоу нужен.
- `LLM_BASE_URL` — default `https://api.deepseek.com`.
- `LLM_MODEL` — default `deepseek-chat`.
- `BOT_PASSWORD` — пароль для `/start`, default `202020`.
- `ADMIN_ADDR` — web-админка, default `127.0.0.1:8080` (`/admin`).
- `ADMIN_USERNAME` — логин web-админки, default `admin`.
- `ADMIN_PASSWORD` — пароль web-админки, default = `BOT_PASSWORD`.
- `DIGEST_INTERVAL_MINUTES` — default 360.
- `STATE_FILE` (default `state.json`), `SESSIONS_DIR` (default `sessions`).

## Команды (`src/bot.rs` enum `Command`)

`/start` `/help` `/connect` `/mcps` `/tools` `/call` `/watch` `/unwatch`
`/watches` `/disconnect` `/profile` `/info` `/facts` `/trip` `/reset`

## Подключение MCP — ТОЧНЫЙ синтаксис (`parse_connect`)

HTTP (URL-first, порядок свободный):
```
/connect <url> [name=NAME] [auth=TOKEN] [Header:Value ...]
```
stdio (бот спавнит child-процесс; `program` обязан быть в PATH хоста бота):
```
/connect stdio <program> [args...] [name=NAME] [env=KEY=VALUE ...]
```
- HTTP `auth=` → шлётся как `Authorization: Bearer <token>`.
- stdio `env=KEY=VALUE` → переменные окружения дочернего процесса.
- `name` опционален — выводится из URL/команды.

## Self-connect агентом (`mcp_connect` meta-tool, `src/llm.rs`)

Агент сам зовёт `mcp_connect`, когда нужна способность, которой нет у подключённых
серверов. Args: `transport` (http|stdio, required), `url`, `command` (argv array),
`auth`, `headers`, `env`, `name`. Нужны креды → агент **сначала спрашивает их в
чате**, потом передаёт. **Никогда не печатает секрет обратно.** После connect tools
доступны в том же turn.

## Инварианты

- Юзер взаимодействует только через бота — НЕ предлагать внешние конфиг-файлы.
- Токены/секреты не печатать в чат; агент их только запрашивает и передаёт в connect.
- Подключённые серверы персистятся и реконнектятся при рестарте (`restore_state`).
- HTTP-транспорт = ничего не ставить на хост бота; stdio = бинарь обязан быть в PATH.

## Сборка / тест

```bash
cargo build --release
cargo test                            # unit
cargo test -- --ignored --nocapture   # live (нужны MCP + LLM key)
```

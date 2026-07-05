# tg-agent — Codex context

Telegram-бот = **MCP-клиент + LLM-агент + RAG-фронт**. Юзер общается ТОЛЬКО через
телеграм-чат. MCP-серверы подключаются **в чате** (команды или self-connect
агентом), НЕ через внешние конфиги (Codex Desktop / VS Code `.mcp.json` — НЕ
относятся к этому проекту).

Проектные скиллы: `deploy-vps` (безопасный деплой + верификация),
`rag-smoke` (проверка RAG на проде). Демо для заказчика:
`docs/demo-rag-tasks-4-5.md`.

## Deployment Autonomy

- После правок не оставлять работу локальной: прогнать релевантные проверки,
  закоммитить intended changes, сделать `git fetch`, влить актуальный upstream
  (`main`) без переписывания истории, сразу `git push` и задеплоить.
- Деплой бота: синхронизировать исходники на VPS, собрать `cargo build --release`,
  перезапустить `tg-agent.service`, проверить `systemctl is-active` и короткий
  journal. Если менялся RAG/Telegram runtime, проверить `/opt/tg-agent/.env`
  `LLM_MODEL` без вывода секретов.
- Если менялись исходники бота, синхронизировать их копию в RAG corpus и
  запустить RAG deploy, чтобы live-индекс отвечал по актуальному коду.
- В handoff указывать commit, branch, push/deploy status и важный smoke-output.

## Что это

- Telegram-бот на `teloxide`. Подключает MCP-серверы в рантайме, отвечает на
  свободный текст через LLM tool-loop поверх их tools, гоняет периодическиеджобы.
- MCP-клиент: `rmcp` 1.8, два транспорта (HTTP Streamable + stdio child).
- Агент-рантайм портирован из ai-playground: sticky-facts память, профиль,
  инварианты, PromptBuilder, динамический multi-agent рой планирования поездок.

## Карта кода

- `src/main.rs` — старт: env → Config → restore_state (реконнект MCP/watches) → dispatcher.
- `src/config.rs` — env-конфиг (см. ниже).
- `src/bot.rs` — команды (enum `Command`), парсинг `/connect` (`parse_connect`,
  `parse_connect_stdio`).
- `src/mcp_client.rs` — `ConnectParams`, `connect()` → `spawn_stdio` / `connect_http`.
- `src/llm.rs` — LLM tool-loop + meta-tools `mcp_connect` / `mcp_disconnect` /
  `schedule_summary`; defs и хендлеры здесь.
- `src/agent/flow.rs` — динамический рой агентов (BriefAgent → OptionsAgent →
  SwarmPlanner → WorkerAgents → VerifierAgent → ArtifactsAgent → FinalAgent),
  `/trip`. План задач строит SwarmPlanner из живого инвентаря MCP-tools, а не
  фиксированный stage-граф. Реестр агентов — `SwarmAgentRegistry`: каждый агент
  отдельная сущность (`SwarmAgentSpec`: имя, роль, модель, права на tools и
  side-effects). Старые `Stage`-метки остались только для serde-совместимости
  персиста, в логику флоу не входят.
- `src/agent/router.rs` — семантический роутер (LLM, без keyword-списков):
  trip / chat / offtopic.
- `src/agent/prompt.rs` — `BASE_SYSTEM` + `build_system_prompt` (слои промпта).
- `src/persist.rs`, `src/state.rs` — персист серверов/подписок/watch, общий стейт.
- `src/scheduler.rs` — периодические watch-джобы и push-summaries.
- `src/rag_client.rs` — RAG-режим (`/rag on`): сабпроцесс
  `rag-indexer answer --mode rag --json` (репо `Rag/ollama-rag-indexer`).
  Каждый ход: `--history` (память сессии) + `--task-state` + `--rewrite`;
  парсит sources с `chunk_id`/`quote`, рендерит «цитаты»; `relevant:false` =
  честный «не знаю» без источников (отказ зашит в код индексера).
- `src/agent/rag_task.rs` — task state RAG-диалога (цель / уточнено /
  ограничения): LLM-экстракция перед каждым ответом, живёт в
  `ChatSession.rag_task`, видна в `/rag status`, чистится `/reset`.

## Конфиг (env, `src/config.rs`)

- `TELEGRAM_BOT_TOKEN` — **обязателен**.
- `LLM_API_KEY` или `DEEPSEEK_API_KEY` — **без него free-text отключён**, работают
  только команды. Для агент-флоу нужен.
- `LLM_BASE_URL` — default `https://api.deepseek.com`.
- `LLM_MODEL` — default `deepseek-chat`. Базовая модель всех агентов роя.
- `SWARM_MODEL_<AGENT>` — переопределение модели одного агента роя (опционально).
  Суффикс — имя агента в UPPER_SNAKE, не-буквенно-цифровое → `_`. Примеры:
  `SWARM_MODEL_VERIFIERAGENT`, `SWARM_MODEL_ARTIFACTSAGENT`, `SWARM_MODEL_BRIEFAGENT`.
  Не задано → агент берёт `LLM_MODEL`. Плюс план может задать модель на отдельную
  задачу через поле `model` в task-графе (приоритетнее env).
- `BOT_PASSWORD` — пароль для `/start`, default `202020`.
- `ADMIN_ADDR` — web-админка, default `127.0.0.1:8080` (`/admin`).
- `ADMIN_USERNAME` — логин web-админки, default `admin`.
- `ADMIN_PASSWORD` — пароль web-админки; без него `/admin` отключён. Должен отличаться от `BOT_PASSWORD`.
- `DIGEST_INTERVAL_MINUTES` — default 360.
- `STATE_FILE` (default `state.json`), `SESSIONS_DIR` (default `sessions`).
- `RAG_*` (см. `.env.example`): `RAG_INDEX` включает режим; ключевые —
  `RAG_EMBED_MODEL` (прод: bge-m3), `RAG_CHAT_PROVIDER` (прод: openai =
  DeepSeek, ключ `RAG_CHAT_API_KEY` только через env), `RAG_MIN_SCORE`
  (прод: 0.5, откалиброван), `RAG_REWRITE`, `RAG_TIMEOUT_SECS`.

## Команды (`src/bot.rs` enum `Command`)

`/start` `/help` `/connect` `/mcps` `/tools` `/call` `/watch` `/unwatch`
`/watches` `/disconnect` `/profile` `/info` `/facts` `/trip` `/rag` `/compact`
`/reset`

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
- Рой настоящий, не стейт-машина: задачи строит планировщик-LLM, каждый агент —
  отдельная сущность со своей моделью/ролью/правами. В систем-промпт агента
  попадает только его identity/role/model + контекст его задачи; working-память и
  чужие identity не протекают (см. `build_swarm_worker_system`/`swarm_agent_system`).
- Side-effects (внешние артефакты) разрешены только `ArtifactsAgent` и только
  после гейта `VerifierAgent`; обычный worker не может эскалироваться до side-effects.

## Сборка / тест

```bash
cargo build --release
cargo test                            # unit (рой: реестр агентов, изоляция промптов, гейты)
cargo test -- --ignored --nocapture   # live (нужны MCP + LLM key): kayak / cycling / walk
```

Live-сценарии роя — `tests/live_trip_flow.rs` (байдарки, велопоход, прогулка):
прогоняются на VPS с `LLM_API_KEY` + MCP (weather/osm/google).

Live RAG-диалоги — `tests/live_rag_dialog.rs`: 2 сценария (12+10 сообщений),
нужны `RAG_INDEX` (локальный qwen-индекс) + Ollama; проверяют источники+цитаты
на каждом ходе, refusal без источников, удержание цели.

## Деплой

`./deploy.sh` (rsync → remote release build → restart → prune). Правила и
верификация — скилл `deploy-vps`. ГЛАВНОЕ: никогда не запускать бинарь
`tg-agent` руками на VPS (`--version` не парсится → второй поллер →
`TerminatedByOtherGetUpdates`). Прод-раскладка: бот `/opt/tg-agent`,
RAG-движок `/opt/ollama-rag-indexer`, box 2GB RAM + 2GB swap — модели >1.5GB
не тянуть.

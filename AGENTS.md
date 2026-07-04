# tg-agent — Codex context

Telegram-бот = **MCP-клиент + LLM-агент**. Юзер общается ТОЛЬКО через телеграм-чат.
MCP-серверы подключаются **в чате** (команды или self-connect агентом), НЕ через
внешние конфиги (Codex Desktop / VS Code `.mcp.json` — НЕ относятся к этому проекту).

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
- `src/rag_client.rs` — тонкий клиент к локальному `rag-indexer answer --mode rag`;
  включается в чате через `/rag on`, выключается через `/rag off`. С 2026-07
  `rag-indexer` всегда возвращает `relevant: bool`, `retrieval` (кандидаты
  до/после фильтра, порог), `rewritten_query` и `quote` в каждом source —
  подробности в `Rag/ollama-rag-indexer/AGENTS.md`. `RagReply`/`RagSource`
  парсят всё это; `render()` печатает источники как
  `[n] source / section #chunk_id score=…` + «цитату» под каждым. Рефьюзл при
  низкой релевантности долетает автоматически (`relevant: false`, фиксированный
  "не знаю", `sources` пустой). Каждый ход бот передаёт `--history` (последние
  сообщения сессии) и `--task-state`; `--rewrite` включается при непустой
  истории (`RAG_REWRITE=0` отключает).
- `src/agent/rag_task.rs` — task state RAG-диалога (цель / что уточнено /
  ограничения-термины): LLM-экстракция перед каждым RAG-ответом (fallback —
  цель из первого вопроса), хранится в `ChatSession.rag_task`, показывается в
  `/rag status`, чистится `/reset`. Живые длинные сценарии:
  `tests/live_rag_dialog.rs` (2 диалога 12+10 сообщений, `--ignored`, нужны
  RAG_INDEX + Ollama).
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
- `RAG_INDEX` — путь к готовому индексу `rag-indexer` (`.../structural`). Если
  не задан, `/rag on` недоступен.
- `RAG_INDEXER_BIN` — путь к CLI `rag-indexer`, default `rag-indexer`.
- `RAG_EMBED_MODEL`, `RAG_CHAT_MODEL`, `RAG_OLLAMA_URL`, `RAG_CHAT_URL`,
  `RAG_SEARCH_MODE`, `RAG_TOP_K` — параметры RAG-клиента.

## Команды (`src/bot.rs` enum `Command`)

`/start` `/help` `/connect` `/mcps` `/tools` `/call` `/watch` `/unwatch`
`/watches` `/disconnect` `/profile` `/info` `/facts` `/trip` `/rag`
`/compact` `/reset`

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

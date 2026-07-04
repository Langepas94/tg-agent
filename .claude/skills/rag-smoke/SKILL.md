---
name: rag-smoke
description: Smoke-test the RAG pipeline on the production VPS — warm-up, sourced answer, off-topic refusal. Use before a customer demo, after deploy/reindex/env changes, or when the user reports RAG answers look wrong on the server.
---

# RAG smoke on the VPS

Runs the exact subprocess call the bot makes per /rag turn. All commands go
over ssh (`~/.ssh/id_ed25519_vps`, `root@5.129.234.9`).

## 1. Warm up the embedding model (~4s warm, 30-60s cold)

```bash
ssh -i ~/.ssh/id_ed25519_vps root@5.129.234.9 \
  'curl -s http://localhost:11434/api/embed -d "{\"model\":\"bge-m3\",\"input\":\"warmup\"}" \
     -o /dev/null -w "embed: %{time_total}s\n"'
```

## 2. Relevant question — expect sources + quotes

```bash
ssh -i ~/.ssh/id_ed25519_vps root@5.129.234.9 \
  'set -a; source /opt/tg-agent/.env; set +a; \
   $RAG_INDEXER_BIN answer --mode rag --json --index $RAG_INDEX \
     --query "Как бот подключает MCP-серверы в рантайме?" \
     --model $RAG_EMBED_MODEL --chat-model $RAG_CHAT_MODEL \
     --chat-url $RAG_CHAT_URL --chat-provider $RAG_CHAT_PROVIDER \
     --search-mode $RAG_SEARCH_MODE --top-k $RAG_TOP_K \
     --min-dense-score $RAG_MIN_SCORE' \
  | python3 -c "import json,sys; d=json.load(sys.stdin); \
      assert d['relevant'] and d['sources'], d; \
      assert all(s['quote'] for s in d['sources']), 'missing quotes'; \
      print('OK:', len(d['sources']), 'sources, quotes present')"
```

## 3. Off-topic — expect code-enforced refusal, zero sources

Same command with `--query "Кто выиграл чемпионат мира по футболу 2018 года?"`;
assert `relevant == False` and `sources == []`. The refusal text is the fixed
"Не знаю…" message — if instead you get an answer with sources, the floor is
miscalibrated (see the `rebuild-index` skill in the Rag repo).

## 4. Bot side

```bash
ssh -i ~/.ssh/id_ed25519_vps root@5.129.234.9 \
  'systemctl is-active tg-agent; grep -c "^RAG_" /opt/tg-agent/.env; \
   journalctl -u tg-agent --since "-10 min" | grep -i "rag\|error" | tail -5'
```

Expect: `active`, 12 RAG vars, no RAG errors. In Telegram the flow is
`/rag on` → question → answer ends with an «Источники:» block; `/rag status`
shows the client config and the dialog task state.

Known-good timings: warm turn ~9-12s, cold first turn 30-60s (2GB box unloads
bge-m3 after idle; keep_alive tuned via ollama systemd override).

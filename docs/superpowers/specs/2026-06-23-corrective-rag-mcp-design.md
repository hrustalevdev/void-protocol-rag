# Corrective RAG MCP Server — Design Spec

**Date:** 2026-06-23
**Stack:** TypeScript, Node.js
**Status:** Approved

---

## Overview

MCP-сервер, превращающий локальную папку с документами в поисковую базу знаний. Разработчик подключает сервер к IDE, индексирует документы и задаёт вопросы на русском языке — сервер ищет по английским документам и возвращает ответ на русском. Внутри — Corrective RAG пайплайн на LangGraph с локальной LLM через Ollama.

---

## Architecture

### Docker Compose (3 сервиса)

```
mcp-server  (Node.js/TS, HTTP/SSE, port 3000)
  ↕ HTTP
chromadb    (port 8000) — векторное хранилище
ollama      (port 11434) — LLM + эмбеддинги
  └── nomic-embed-text  (эмбеддинги, мультиязычная)
  └── phi3:mini         (LLM, по умолчанию; настраивается через конфиг)
```

### Project Structure

```
src/
  index.ts              ← точка входа, MCP SSE сервер
  config.ts             ← все настройки из env переменных
  tools/
    index-folder.ts     ← MCP tool: index_folder
    ask-question.ts     ← MCP tool: ask_question
    find-relevant-docs.ts ← MCP tool: find_relevant_docs
    index-status.ts     ← MCP tool: index_status
  rag/
    graph.ts            ← LangGraph Corrective RAG граф
    state.ts            ← тип RAGState
    nodes/
      rewrite-query.ts  ← перевод RU→EN + переформулировка
      retrieve.ts       ← BM25 + vector → RRF
      grade-chunks.ts   ← LLM оценивает релевантность чанков
      broaden-query.ts  ← расширение запроса при retry
      generate.ts       ← генерация ответа на русском
  indexer/
    indexer.ts          ← сканирование файлов, оркестрация
    chunker.ts          ← стратегии чанкинга (текст vs код)
    loaders.ts          ← загрузчики форматов файлов
  retriever/
    hybrid.ts           ← гибридный поиск, RRF fusion
    bm25.ts             ← in-memory BM25 индекс
tests/
  unit/
    graph.test.ts
    indexer.test.ts
    bm25.test.ts
    rrf.test.ts
  e2e/
    mcp-tools.test.ts
sample_docs/            ← демо-документы (игра Void Protocol, ~500KB)
docker-compose.yml
Dockerfile
.env.example
mcp-config.json         ← пример конфига для VSCode Copilot
```

---

## Key Libraries

| Задание (Python) | TypeScript-эквивалент |
|---|---|
| FastMCP | `@modelcontextprotocol/sdk` (SSE transport) |
| LangGraph | `@langchain/langgraph` |
| LangChain loaders/splitters | `langchain` + `@langchain/community` |
| ChromaDB in-process | `chromadb` JS-клиент + ChromaDB в Docker |
| rank_bm25 | `wink-bm25-text-search` (in-memory) |
| Ollama | `ollama` JS-клиент |
| — | `js-yaml` (парсинг YAML) |
| — | `glob` (сканирование файлов) |
| — | `vitest` (тесты) |

---

## Corrective RAG Graph (LangGraph.js)

### State

```typescript
interface RAGState {
  originalQuery: string      // русский запрос от пользователя
  rewrittenQuery: string     // английский запрос для поиска
  chunks: Chunk[]            // все retrieved чанки
  relevantChunks: Chunk[]    // после grading
  answer: string             // ответ на русском
  sources: string[]          // пути к исходным файлам
  retryCount: number         // счётчик повторов, max 2
}
```

### Graph Flow

```
START
  │
  ▼
rewriteQuery   ← LLM: переводит RU→EN, переформулирует для поиска
  │
  ▼
retrieve       ← BM25(EN query) + vector(EN query) → RRF → top-6 чанков
  │
  ▼
gradeChunks    ← LLM: для каждого чанка — relevant / not_relevant
                 промпт учитывает: "вопрос на RU, контекст на EN"
  │
  ├─ [relevantChunks.length >= 2 OR retryCount >= 2]
  │     ▼
  │   generate  ← LLM: генерирует ответ на русском по релевантным чанкам
  │     ▼
  │   END
  │
  └─ [relevantChunks.length < 2 AND retryCount < 2]
        ▼
      broadenQuery  ← LLM: расширяет запрос синонимами/смежными понятиями
        │
        └──▶ retrieve (retryCount++)
```

### Thresholds

- `top-K = 6` чанков при каждом поиске
- `enough relevant` = минимум 2 чанка прошли grading
- `max retries = 2`

---

## Chunking Strategies

**Текстовые файлы** (`.md`, `.txt`, `.yaml`, `.json`):
- `RecursiveCharacterTextSplitter`
- chunk size: 512 токенов, overlap: 64

**Код** (`.py`, `.js`, `.ts`):
- `RecursiveCharacterTextSplitter` с разделителями кода: `\nclass `, `\nfunction `, `\nconst `, `\nexport `, `\n\n`
- chunk size: 256 токенов, overlap: 32

**Метаданные каждого чанка:**
```typescript
{ source: "sample_docs/docs/world-lore.md", chunkIndex: 3, totalChunks: 12 }
```

---

## Hybrid Search (BM25 + Vector → RRF)

1. BM25 поиск по переведённому EN запросу → ранжированный список чанков
2. Векторный поиск через ChromaDB (Ollama `nomic-embed-text` эмбеддинги) → ранжированный список
3. Reciprocal Rank Fusion: `score(d) = Σ 1 / (k + rank(d))`, где `k=60`
4. Итоговый top-K по RRF score

**BM25 хранение:** in-memory индекс, пересобирается из ChromaDB при старте сервера.

---

## MCP Tools

### `index_folder`
```
Вход: folderPath (string), globPattern? (string, default "**/*")
Выход: { filesIndexed, chunksCreated, duration }
Description: "Index documents from a local folder into the RAG knowledge base.
Use when the user wants to add, update or reindex documents from a directory.
Supports .md, .txt, .py, .js, .ts, .json, .yaml files."
```

### `ask_question`
```
Вход: question (string) — на любом языке, включая русский
Выход: { answer, sources[], confidence: "high" | "low" }
  confidence: "high" — граф нашёл ≥2 релевантных чанка
  confidence: "low"  — retries исчерпаны, ответ на основе того что есть
Description: "Answer a question using the indexed knowledge base via RAG pipeline.
Use when the user asks anything about the content of indexed documents —
project docs, wiki, notes, code. Returns answer in the query language + sources."
```

### `find_relevant_docs`
```
Вход: query (string), topK? (number, default 5)
Выход: { chunks: [{ content, source, score }] }
Description: "Search the knowledge base and return ranked document chunks
without generating an answer. Use for exploring what's in the index,
finding specific passages, or when raw search results are needed."
```

### `index_status`
```
Вход: —
Выход: { fileCount, chunkCount, lastIndexed: ISO8601 | null, status }
Description: "Return statistics about the current knowledge base index:
file count, chunk count, last indexed time. Use to check if documents
are indexed before querying."
```

---

## Error Handling

| Ситуация | Поведение |
|---|---|
| ChromaDB недоступна | Ошибка: "Knowledge base unavailable. Is ChromaDB running?" |
| Ollama недоступна | Ошибка: "LLM service unavailable. Is Ollama running?" |
| Папка не найдена | Ошибка: "Folder not found: {path}" |
| Нет файлов по паттерну | Предупреждение + `filesIndexed: 0`, не падает |
| Индекс пустой при запросе | "No documents indexed yet. Run index_folder first." |
| Retries исчерпаны (0 релевантных) | Генерирует ответ с пометкой low confidence, не падает |

---

## Configuration (`config.ts`, из `.env`)

```
CHROMADB_URL=http://chromadb:8000
OLLAMA_URL=http://ollama:11434
LLM_MODEL=phi3:mini
EMBED_MODEL=nomic-embed-text
MCP_PORT=3000
CHROMA_COLLECTION=rag_knowledge_base
RETRIEVAL_TOP_K=6
MIN_RELEVANT_CHUNKS=2
MAX_RETRIES=2
```

---

## Tests (≥10)

### Unit (mock LLM + mock ChromaDB)

1. `rewriteQuery` — переводит русский запрос в английский
2. `gradeChunks` — фильтрует нерелевантные чанки, оставляет релевантные
3. `generate` — возвращает ответ на русском при английском контексте
4. граф выполняет retry при `relevantChunks.length < 2`
5. граф останавливается после 2 retries независимо от результата
6. RRF fusion — корректно ранжирует по позициям из двух списков
7. BM25 — находит документ по ключевому слову EN запроса
8. `chunker` — `.md` и `.ts` файлы режутся разными стратегиями

### E2E (реальный сервер, тестовая ChromaDB коллекция)

9. `index_folder` → `index_status` показывает корректное число файлов и чанков
10. `index_folder` → `ask_question` (RU) → ответ содержит факт из документа

E2E тесты используют отдельную ChromaDB коллекцию `rag_test` (не `rag_knowledge_base`), чтобы не затирать рабочий индекс.

---

## Infrastructure

### Docker Compose

```yaml
services:
  mcp-server:
    build: .
    ports: ["3000:3000"]
    depends_on: [chromadb, ollama]
    env_file: .env

  chromadb:
    image: chromadb/chroma:latest
    ports: ["8000:8000"]
    volumes: [chroma_data:/chroma/chroma]

  ollama:
    image: ollama/ollama:latest
    ports: ["11434:11434"]
    volumes: [ollama_data:/root/.ollama]
    # init-скрипт pull-ит модели (phi3:mini + nomic-embed-text) при первом старте
    # через healthcheck: ollama pull ${LLM_MODEL} && ollama pull ${EMBED_MODEL}

volumes:
  chroma_data:
  ollama_data:
```

### CI Pipeline (GitHub Actions)

- `lint` — ESLint + TypeScript type check
- `test` — vitest unit + e2e (ChromaDB + Ollama через docker в CI)
- `build` — `tsc`, проверка что компилируется

### VSCode Copilot конфиг (`mcp-config.json`)

```json
{
  "mcpServers": {
    "void-protocol-rag": {
      "url": "http://localhost:3000/sse",
      "type": "sse"
    }
  }
}
```

# Development Report

## История создания

Проект начинался как Python + FastMCP — именно так написано в задании. Но решил попробовать сделать на node.js + TS, т.к. это мой основной стек.

Первый этап — выбор технологий через brainstorming с Claude Code:

- **LangGraph vs ручная реализация графа** — выбрал LangGraph: он даёт встроенные условные переходы, аккумулирующий стейт и retry-циклы из коробки, что идеально для Corrective RAG.
- **ChromaDB** — JS-клиент работает только с внешним сервером (нет in-process режима как в Python), поэтому ChromaDB запускается отдельным Docker-контейнером.
- **SSE-транспорт** — Express + `@modelcontextprotocol/sdk`, проще чем stdio для отладки через браузер.
- **Эмбеддинги** — `nomic-embed-text` через Ollama: поддерживает многоязычные запросы, что критично для кросс-языкового поиска.

Ключевое требование обнаружилось в середине планирования: документы в `sample_docs/` на английском, а запросы будут на русском. Наивный поиск по русскому тексту в английских документах даёт нулевые результаты. Это потребовало выделить отдельный узел `rewriteQuery` в начале графа.

Реализация шла по методике Subagent-Driven Development: 10 задач, каждая выполнялась отдельным субагентом с последующим code review. Это позволило параллелизовать работу и изолировать контекст каждой задачи.

**Несколько итераций потребовали следующие баги:**

- **js-yaml ESM** — при обновлении до v5 перестал работать default import (`import yaml from "js-yaml"`), нужны именованные: `import { load, dump } from "js-yaml"`. Обнаружено на e2e тестах с реальными YAML-файлами.
- **Unbounded ChromaDB get()** — `collection.get({})` без `limit` молча обрезает результат на размере страницы. При реиндексации удалял не все документы. Исправлено на `collection.delete({ where: { "chunkIndex": { "$gte": 0 } } })`.
- **Потеря bestRelevantChunks между ретраями** — при исчерпании ретраев с пустым последним раундом `generate` падал на неотфильтрованные чанки. Добавлено аккумулирующее поле `bestRelevantChunks` с reducer `(a, b) => b.length > 0 ? b : a`.
- **chromadb v3 breaking changes** — после обновления зависимостей оказалось, что `{ path: url }` в конструкторе `ChromaClient` устарел, а `getOrCreateCollection` без явного `embeddingFunction` падает с ошибкой. Исправлено на `{ host, port, ssl }` и явный `embeddingFunction`.

## Ключевые проблемы и решения

### Кросс-языковой поиск (RU запросы → EN документы)

Документы в `sample_docs/` написаны на английском языке, но пользователи задают вопросы на русском.
Наивный поиск по русскому тексту в английских документах даёт нулевые или нерелевантные результаты.

**Решение**: узел `rewriteQuery` в LangGraph-графе переводит русский запрос в английский и переформулирует его для максимальной релевантности при поиске. Это происходит до обращения к BM25 и ChromaDB. Узел `generate` явно инструктирует LLM отвечать на русском языке, независимо от языка найденных чанков.

**Узел rewriteQuery** (`src/rag/nodes/rewrite-query.ts`):
```
You are a search query optimizer. The user asked a question in Russian.
Your task: translate it to English and rewrite it to maximize document retrieval quality.
Output ONLY the English search query — no explanations, no quotes, just the query text.

Russian question: ${state.originalQuery}
```

**Узел gradeChunks** использует `state.originalQuery` (русский оригинал), чтобы оценить релевантность английских чанков:
```
You are a relevance grader. The user asked a question (in Russian) and you have a document chunk (in English).
Determine if the chunk contains information relevant to answering the question.
Respond with ONLY "yes" or "no".

Question (Russian): ${state.originalQuery}
Document chunk (English): ${chunk.content}
```

## Примеры промптов

### Удачный пример

**Промпт для gradeChunks** (файл `src/rag/nodes/grade-chunks.ts`):
```
You are a relevance grader. The user asked a question (in Russian) and you have a document chunk (in English).
Determine if the chunk contains information relevant to answering the question.
Respond with ONLY "yes" or "no".

Question (Russian): ${state.originalQuery}
Document chunk (English): ${chunk.content}
```
**Почему сработало:** Промпт явно указывает языковой контекст («in Russian», «in English»), не требует перевода от LLM — только yes/no, что снижает вероятность галлюцинаций и ускоряет обработку.

### Неудачный пример

**Первая версия промпта для gradeChunks** (без явного указания языков):
```
You are a relevance grader. Does the following document chunk contain information
relevant to answering the user's question?
Respond with ONLY "yes" or "no".

Question: ${state.originalQuery}
Chunk: ${chunk.content}
```
**Почему не сработало:** При русском вопросе и английском чанке LLM начинала «помогать» — переводила вопрос или чанк прямо в ответе вместо yes/no, либо отвечала «no» на релевантные чанки просто потому что языки не совпадают. Модель не понимала, что языковое несоответствие — это норма, а не признак нерелевантности. Исправлено явным указанием языкового контекста в промпте.

## Использованные инструменты

- Claude Code (claude-sonnet-4-6) — архитектура, план, код
- Subagent-driven development — параллельная реализация независимых задач

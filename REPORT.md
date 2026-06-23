# Development Report

## История создания

_Добавляйте записи по мере работы._

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

_[Добавьте свой пример]_

## Использованные инструменты

- Claude Code (claude-sonnet-4-6) — архитектура, план, код
- Subagent-driven development — параллельная реализация независимых задач

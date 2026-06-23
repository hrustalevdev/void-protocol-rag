# Agent Instructions — void-protocol-rag

## Project Overview

MCP server implementing a Corrective RAG knowledge base over SSE transport.
Stack: TypeScript (ESM), LangGraph, ChromaDB, Ollama, Express.

**Critical cross-lingual requirement:** sample documents are in **English**; user queries arrive in **Russian**. The `rewriteQuery` node translates RU→EN before retrieval; the `generate` node always answers in **Russian**.

## Module System

This project uses ESM (`"type": "module"`, `"module": "NodeNext"`).
All local imports **must** use `.js` extensions even for `.ts` source files:

```ts
import { BM25Index } from "../retriever/bm25.js"  // correct
import { BM25Index } from "../retriever/bm25"      // wrong — will break at runtime
```

## Repository Layout

```
src/
  config.ts              — getConfig(): Config, reads 9 env vars with defaults
  index.ts               — Express server, SSEServerTransport, startup, tool registration
  indexer/
    loaders.ts           — loadFile(), scanFolder()
    chunker.ts           — chunkDocument(), Chunk type
    indexer.ts           — Indexer class (index, status, BM25 rebuild from ChromaDB)
  retriever/
    bm25.ts              — BM25Index class (in-memory, k1=1.5, b=0.75)
    hybrid.ts            — hybridSearch(), reciprocalRankFusion(k=60)
  rag/
    state.ts             — RAGStateAnnotation (LangGraph Annotation.Root)
    graph.ts             — createRAGGraph(), runRAGQuery()
    nodes/
      rewrite-query.ts   — translate RU→EN + reformulate; output ONLY the English query
      retrieve.ts        — hybridSearch with rewrittenQuery
      grade-chunks.ts    — LLM yes/no per chunk; populates relevantChunks + bestRelevantChunks
      broaden-query.ts   — expand query, increment retryCount
      generate.ts        — prefer bestRelevantChunks → relevantChunks → chunks; answer in RU
  tools/
    index-folder.ts      — register index_folder MCP tool
    ask-question.ts      — register ask_question MCP tool
    find-relevant-docs.ts— register find_relevant_docs MCP tool
    index-status.ts      — register index_status MCP tool
tests/
  unit/                  — chunker, bm25, rrf, graph (nodes + retry), indexer
  e2e/                   — mcp-tools (requires ChromaDB on localhost:8000)
```

## Key Invariants

- **BM25 is in-memory only.** On startup `indexer.loadBM25FromChroma()` rebuilds it with `limit: 100000`.
- **Re-index deletes via filter:** `collection.delete({ where: { "chunkIndex": { "$gte": 0 } } })` — never use unbounded `collection.get({})` which silently truncates at ChromaDB page size.
- **`bestRelevantChunks` accumulates across retries** with reducer `(a, b) => b.length > 0 ? b : a`. The `generate` node uses this so a good first-round result isn't lost if a retry round comes back empty.
- **Retry routing:** `gradeChunks → generate` if `relevantChunks.length >= minRelevantChunks || retryCount >= maxRetries`; otherwise `→ broadenQuery → retrieve`.
- **Confidence:** `"high"` if `bestRelevantChunks.length >= 2 || relevantChunks.length >= 2`, else `"low"`.
- **js-yaml imports:** use named exports — `import { dump, load } from "js-yaml"` — not the default import (js-yaml v4 ESM doesn't export a default).

## Build and Test

```bash
npm run build          # tsc → dist/
npx tsc --noEmit       # type-check without emit
npx vitest run         # all tests (unit + e2e)
npx vitest run tests/unit   # unit only
npx vitest run tests/e2e    # e2e only — needs ChromaDB on localhost:8000
```

E2E tests use a real ChromaDB collection `rag_test` and a class-based Ollama mock.

## Environment Variables

Copy `.env.example` to `.env`. All variables have defaults so the server starts without `.env`.

| Variable | Default | Notes |
|---|---|---|
| `CHROMADB_URL` | `http://chromadb:8000` | Use `http://localhost:8000` locally |
| `OLLAMA_URL` | `http://ollama:11434` | Use `http://localhost:11434` locally |
| `LLM_MODEL` | `phi3:mini` | Any Ollama model |
| `EMBED_MODEL` | `nomic-embed-text` | Multilingual, required for cross-lingual retrieval |
| `MCP_PORT` | `3000` | SSE endpoint port |
| `CHROMA_COLLECTION` | `rag_knowledge_base` | ChromaDB collection name |
| `RETRIEVAL_TOP_K` | `6` | Chunks returned per hybrid search |
| `MIN_RELEVANT_CHUNKS` | `2` | Threshold to skip retry and generate |
| `MAX_RETRIES` | `2` | Max broadenQuery iterations |

## Running Locally

ChromaDB must be running before starting the server or running e2e tests:

```bash
docker run -p 8000:8000 chromadb/chroma
```

Full stack via Docker Compose:

```bash
docker compose up
```

The `ollama` service pulls `LLM_MODEL` and `EMBED_MODEL` on first start (slow).

## MCP Transport

- `GET /sse` — opens SSE stream, returns `sessionId`
- `POST /messages?sessionId=<id>` — sends a tool call
- `GET /health` — liveness probe

MCP config for Claude Desktop / IDE: see `mcp-config.json`.

## Supported File Types for Indexing

`.md`, `.txt`, `.py`, `.js`, `.ts`, `.json`, `.yaml`

Code files (`.py`, `.js`, `.ts`) use smaller chunks: 256 chars / 32 overlap.
Text/doc files use 512 chars / 64 overlap.

## What Not to Do

- Do not add a default export to `js-yaml` imports.
- Do not call `collection.get({})` without a `limit` — ChromaDB silently paginates.
- Do not omit `.js` on local imports.
- Do not answer in English — the `generate` node prompt explicitly requires Russian output.
- Do not skip the `bestRelevantChunks` accumulation when editing retry logic.

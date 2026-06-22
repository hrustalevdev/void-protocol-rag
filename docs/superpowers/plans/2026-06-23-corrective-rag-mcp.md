# Corrective RAG MCP Server — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a TypeScript MCP server with Corrective RAG pipeline that indexes local documents and answers Russian-language questions about them using a local Ollama LLM and hybrid BM25+vector search.

**Architecture:** Express HTTP server with SSE MCP transport wraps a LangGraph Corrective RAG pipeline. Documents are indexed into ChromaDB (vector store) and an in-memory BM25 index. Russian queries are translated to English by the rewriteQuery node, retrieved via hybrid search with RRF fusion, graded per-chunk by LLM, and answered in Russian.

**Tech Stack:** TypeScript 5.5+, `@modelcontextprotocol/sdk`, `@langchain/langgraph`, `@langchain/textsplitters`, `chromadb`, `ollama`, `express`, `js-yaml`, `glob`, `zod`, `vitest`

## Global Constraints

- Node.js ≥ 20
- TypeScript strict mode, ESM output (`"module": "NodeNext"`)
- All config values come from env vars via `src/config.ts` — zero hardcoded values elsewhere
- E2E tests use collection name `rag_test`, never `rag_knowledge_base`
- Run tests: `npx vitest run`
- Build: `npm run build` → `tsc`
- Commit after every task

---

## File Map

```
src/
  index.ts                     ← Express + SSE MCP server entry point
  config.ts                    ← getConfig() reads env vars, returns Config
  tools/
    index-folder.ts            ← registerIndexFolderTool(server, indexer)
    ask-question.ts            ← registerAskQuestionTool(server, graph, embed)
    find-relevant-docs.ts      ← registerFindRelevantDocsTool(server, collection, bm25, embed)
    index-status.ts            ← registerIndexStatusTool(server, indexer)
  rag/
    state.ts                   ← RAGStateAnnotation, RetrievedChunk type
    graph.ts                   ← createRAGGraph(), runRAGQuery()
    nodes/
      rewrite-query.ts         ← createRewriteQueryNode(ollama, config)
      retrieve.ts              ← createRetrieveNode(collection, bm25, embed)
      grade-chunks.ts          ← createGradeChunksNode(ollama, config)
      broaden-query.ts         ← createBroadenQueryNode(ollama, config)
      generate.ts              ← createGenerateNode(ollama, config)
  indexer/
    loaders.ts                 ← loadFile(), scanFolder()
    chunker.ts                 ← chunkDocument()
    indexer.ts                 ← Indexer class
  retriever/
    bm25.ts                    ← BM25Index class (in-memory, from-scratch)
    hybrid.ts                  ← hybridSearch(), reciprocalRankFusion()
tests/
  unit/
    chunker.test.ts
    bm25.test.ts
    rrf.test.ts
    graph.test.ts
    indexer.test.ts
  e2e/
    mcp-tools.test.ts          ← requires ChromaDB running on localhost:8000
docker-compose.yml
Dockerfile
.env.example
mcp-config.json
.github/workflows/ci.yml
```

---

### Task 1: Project Setup & Configuration

**Files:**
- Modify: `package.json`
- Create: `tsconfig.json`
- Create: `vitest.config.ts`
- Create: `.env.example`
- Create: `src/config.ts`

**Interfaces:**
- Produces: `Config` interface, `getConfig(): Config`

- [ ] **Step 1: Install dependencies**

```bash
npm install @modelcontextprotocol/sdk @langchain/langgraph @langchain/textsplitters chromadb ollama express js-yaml glob zod
npm install -D typescript @types/node @types/express @types/js-yaml vitest
```

- [ ] **Step 2: Replace `package.json`**

```json
{
  "name": "void-protocol-rag",
  "version": "1.0.0",
  "type": "module",
  "main": "dist/index.js",
  "scripts": {
    "build": "tsc",
    "dev": "node --watch dist/index.js",
    "test": "vitest run",
    "test:watch": "vitest"
  },
  "dependencies": {
    "@langchain/langgraph": "^0.2.0",
    "@langchain/textsplitters": "^0.1.0",
    "@modelcontextprotocol/sdk": "^1.0.0",
    "chromadb": "^1.9.0",
    "express": "^4.18.0",
    "glob": "^11.0.0",
    "js-yaml": "^4.1.0",
    "ollama": "^0.5.0",
    "zod": "^3.22.0"
  },
  "devDependencies": {
    "@types/express": "^4.17.0",
    "@types/js-yaml": "^4.0.0",
    "@types/node": "^20.0.0",
    "typescript": "^5.5.3",
    "vitest": "^2.0.0"
  }
}
```

- [ ] **Step 3: Create `tsconfig.json`**

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "NodeNext",
    "moduleResolution": "NodeNext",
    "outDir": "dist",
    "rootDir": "src",
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "declaration": true,
    "sourceMap": true
  },
  "include": ["src"],
  "exclude": ["node_modules", "dist"]
}
```

- [ ] **Step 4: Create `vitest.config.ts`**

```typescript
import { defineConfig } from "vitest/config"

export default defineConfig({
  test: {
    globals: true,
    environment: "node",
    include: ["tests/**/*.test.ts"],
  },
})
```

- [ ] **Step 5: Create `.env.example`**

```env
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

- [ ] **Step 6: Create `src/config.ts`**

```typescript
export interface Config {
  chromadbUrl: string
  ollamaUrl: string
  llmModel: string
  embedModel: string
  mcpPort: number
  chromaCollection: string
  retrievalTopK: number
  minRelevantChunks: number
  maxRetries: number
}

export function getConfig(): Config {
  return {
    chromadbUrl: process.env.CHROMADB_URL ?? "http://localhost:8000",
    ollamaUrl: process.env.OLLAMA_URL ?? "http://localhost:11434",
    llmModel: process.env.LLM_MODEL ?? "phi3:mini",
    embedModel: process.env.EMBED_MODEL ?? "nomic-embed-text",
    mcpPort: parseInt(process.env.MCP_PORT ?? "3000", 10),
    chromaCollection: process.env.CHROMA_COLLECTION ?? "rag_knowledge_base",
    retrievalTopK: parseInt(process.env.RETRIEVAL_TOP_K ?? "6", 10),
    minRelevantChunks: parseInt(process.env.MIN_RELEVANT_CHUNKS ?? "2", 10),
    maxRetries: parseInt(process.env.MAX_RETRIES ?? "2", 10),
  }
}
```

- [ ] **Step 7: Verify compilation**

```bash
npm run build
```
Expected: `dist/` folder created, no errors.

- [ ] **Step 8: Commit**

```bash
git add package.json tsconfig.json vitest.config.ts .env.example src/config.ts
git commit -m "chore: project setup — deps, tsconfig, vitest, config"
```

---

### Task 2: Document Loaders & Chunker

**Files:**
- Create: `src/indexer/loaders.ts`
- Create: `src/indexer/chunker.ts`
- Create: `tests/unit/chunker.test.ts`

**Interfaces:**
- Produces:
  - `DocumentContent { content: string; filePath: string }`
  - `loadFile(filePath: string): Promise<DocumentContent>`
  - `scanFolder(folderPath: string, globPattern?: string): Promise<string[]>`
  - `Chunk { content: string; metadata: { source: string; chunkIndex: number; totalChunks: number } }`
  - `chunkDocument(doc: DocumentContent): Promise<Chunk[]>`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/chunker.test.ts`:

```typescript
import { describe, it, expect } from "vitest"
import path from "path"
import { chunkDocument } from "../../src/indexer/chunker.js"
import type { DocumentContent } from "../../src/indexer/loaders.js"

describe("chunkDocument", () => {
  it("splits a long markdown file into multiple chunks", async () => {
    const content = "# Heading\n\n" + "word ".repeat(300)
    const doc: DocumentContent = { content, filePath: "/docs/test.md" }
    const chunks = await chunkDocument(doc)
    expect(chunks.length).toBeGreaterThan(1)
  })

  it("sets correct metadata on each chunk", async () => {
    const doc: DocumentContent = {
      content: "word ".repeat(200),
      filePath: "/docs/lore.md",
    }
    const chunks = await chunkDocument(doc)
    expect(chunks[0].metadata.source).toBe("/docs/lore.md")
    expect(chunks[0].metadata.chunkIndex).toBe(0)
    expect(chunks[0].metadata.totalChunks).toBe(chunks.length)
  })

  it("uses smaller chunk size for .ts code files", async () => {
    const mdDoc: DocumentContent = {
      content: "word ".repeat(300),
      filePath: "/src/file.md",
    }
    const tsDoc: DocumentContent = {
      content: "word ".repeat(300),
      filePath: "/src/file.ts",
    }
    const mdChunks = await chunkDocument(mdDoc)
    const tsChunks = await chunkDocument(tsDoc)
    // code splitter has smaller chunk size → more chunks
    expect(tsChunks.length).toBeGreaterThanOrEqual(mdChunks.length)
  })
})
```

- [ ] **Step 2: Run to verify it fails**

```bash
npx vitest run tests/unit/chunker.test.ts
```
Expected: FAIL — `Cannot find module '../../src/indexer/chunker.js'`

- [ ] **Step 3: Create `src/indexer/loaders.ts`**

```typescript
import fs from "fs/promises"
import path from "path"
import { glob } from "glob"
import yaml from "js-yaml"

export interface DocumentContent {
  content: string
  filePath: string
}

const SUPPORTED_EXT = new Set([".md", ".txt", ".py", ".js", ".ts", ".json", ".yaml", ".yml"])

export async function loadFile(filePath: string): Promise<DocumentContent> {
  const ext = path.extname(filePath).toLowerCase()
  if (!SUPPORTED_EXT.has(ext)) throw new Error(`Unsupported file type: ${ext}`)

  const raw = await fs.readFile(filePath, "utf-8")
  let content = raw

  if (ext === ".json") {
    content = JSON.stringify(JSON.parse(raw), null, 2)
  } else if (ext === ".yaml" || ext === ".yml") {
    content = yaml.dump(yaml.load(raw))
  }

  return { content, filePath }
}

export async function scanFolder(folderPath: string, globPattern = "**/*"): Promise<string[]> {
  const files = await glob(globPattern, { cwd: folderPath, absolute: true, nodir: true })
  return files.filter(f => SUPPORTED_EXT.has(path.extname(f).toLowerCase()))
}
```

- [ ] **Step 4: Create `src/indexer/chunker.ts`**

```typescript
import path from "path"
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters"
import type { DocumentContent } from "./loaders.js"

export interface Chunk {
  content: string
  metadata: {
    source: string
    chunkIndex: number
    totalChunks: number
  }
}

const CODE_EXTS = new Set([".py", ".js", ".ts"])

const textSplitter = new RecursiveCharacterTextSplitter({ chunkSize: 512, chunkOverlap: 64 })

const codeSplitter = new RecursiveCharacterTextSplitter({
  chunkSize: 256,
  chunkOverlap: 32,
  separators: ["\nclass ", "\nfunction ", "\nconst ", "\nexport ", "\n\n", "\n", " ", ""],
})

export async function chunkDocument(doc: DocumentContent): Promise<Chunk[]> {
  const ext = path.extname(doc.filePath).toLowerCase()
  const splitter = CODE_EXTS.has(ext) ? codeSplitter : textSplitter
  const texts = await splitter.splitText(doc.content)
  return texts.map((content, i) => ({
    content,
    metadata: { source: doc.filePath, chunkIndex: i, totalChunks: texts.length },
  }))
}
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
npx vitest run tests/unit/chunker.test.ts
```
Expected: 3 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/indexer/loaders.ts src/indexer/chunker.ts tests/unit/chunker.test.ts
git commit -m "feat: document loaders and chunker with tests"
```

---

### Task 3: BM25 Index

**Files:**
- Create: `src/retriever/bm25.ts`
- Create: `tests/unit/bm25.test.ts`

**Interfaces:**
- Produces:
  - `BM25Index` class with `.add(id, text)`, `.buildIndex()`, `.search(query, topK): {id, score}[]`, `.rebuild(docs)`, `.size`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/bm25.test.ts`:

```typescript
import { describe, it, expect, beforeEach } from "vitest"
import { BM25Index } from "../../src/retriever/bm25.js"

describe("BM25Index", () => {
  let bm25: BM25Index

  beforeEach(() => {
    bm25 = new BM25Index()
    bm25.add("doc1", "the void protocol is a space station game")
    bm25.add("doc2", "combat mechanics include dodge and parry")
    bm25.add("doc3", "void station orbits the planet kedraxis")
    bm25.buildIndex()
  })

  it("returns the most relevant document for a keyword query", () => {
    const results = bm25.search("void protocol", 3)
    expect(results[0].id).toBe("doc1")
  })

  it("returns higher score for documents with more matching terms", () => {
    const results = bm25.search("void station", 3)
    // doc1 has "void", doc3 has "void" and "station" → doc3 scores higher
    expect(results[0].id).toBe("doc3")
  })

  it("returns empty array for query with no matching terms", () => {
    const results = bm25.search("xyzzy nonexistent", 3)
    expect(results).toHaveLength(0)
  })

  it("rebuild replaces the index", () => {
    bm25.rebuild([{ id: "new1", text: "completely different content about dragons" }])
    const results = bm25.search("dragons", 3)
    expect(results[0].id).toBe("new1")
    expect(bm25.size).toBe(1)
  })
})
```

- [ ] **Step 2: Run to verify it fails**

```bash
npx vitest run tests/unit/bm25.test.ts
```
Expected: FAIL — `Cannot find module '../../src/retriever/bm25.js'`

- [ ] **Step 3: Create `src/retriever/bm25.ts`**

```typescript
export class BM25Index {
  private readonly k1 = 1.5
  private readonly b = 0.75
  private docs = new Map<string, string[]>()
  private df = new Map<string, number>()
  private avgdl = 0

  add(id: string, text: string): void {
    const tokens = this.tokenize(text)
    this.docs.set(id, tokens)
    new Set(tokens).forEach(t => this.df.set(t, (this.df.get(t) ?? 0) + 1))
  }

  buildIndex(): void {
    const total = [...this.docs.values()].reduce((s, t) => s + t.length, 0)
    this.avgdl = this.docs.size > 0 ? total / this.docs.size : 0
  }

  search(query: string, topK: number): Array<{ id: string; score: number }> {
    const qTerms = this.tokenize(query)
    const N = this.docs.size
    const scores = new Map<string, number>()

    for (const term of qTerms) {
      const n = this.df.get(term) ?? 0
      if (n === 0) continue
      const idf = Math.log((N - n + 0.5) / (n + 0.5) + 1)
      for (const [id, tokens] of this.docs) {
        const tf = tokens.filter(t => t === term).length
        if (tf === 0) continue
        const dl = tokens.length
        const tfs = (tf * (this.k1 + 1)) / (tf + this.k1 * (1 - this.b + (this.b * dl) / this.avgdl))
        scores.set(id, (scores.get(id) ?? 0) + idf * tfs)
      }
    }

    return [...scores.entries()]
      .map(([id, score]) => ({ id, score }))
      .sort((a, b) => b.score - a.score)
      .slice(0, topK)
  }

  rebuild(documents: Array<{ id: string; text: string }>): void {
    this.docs.clear()
    this.df.clear()
    documents.forEach(d => this.add(d.id, d.text))
    this.buildIndex()
  }

  get size(): number {
    return this.docs.size
  }

  private tokenize(text: string): string[] {
    return text.toLowerCase().match(/\b[a-z0-9]+\b/g) ?? []
  }
}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
npx vitest run tests/unit/bm25.test.ts
```
Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/retriever/bm25.ts tests/unit/bm25.test.ts
git commit -m "feat: in-memory BM25 index with tests"
```

---

### Task 4: Hybrid Retriever & RRF

**Files:**
- Create: `src/retriever/hybrid.ts`
- Create: `tests/unit/rrf.test.ts`

**Interfaces:**
- Consumes: `BM25Index` from Task 3
- Produces:
  - `EmbedFn = (text: string) => Promise<number[]>`
  - `reciprocalRankFusion(list1, list2, k?): {id, score}[]`
  - `hybridSearch(query, topK, collection, bm25, embed): Promise<RetrievedChunk[]>`
    where `RetrievedChunk` is imported from `src/rag/state.ts` (created in Task 6 — define it here first as a local type, move import in Task 6)

- [ ] **Step 1: Write failing RRF tests**

Create `tests/unit/rrf.test.ts`:

```typescript
import { describe, it, expect } from "vitest"
import { reciprocalRankFusion } from "../../src/retriever/hybrid.js"

describe("reciprocalRankFusion", () => {
  it("scores a document appearing in both lists higher than one in only one list", () => {
    const list1 = [{ id: "a" }, { id: "b" }, { id: "c" }]
    const list2 = [{ id: "b" }, { id: "d" }, { id: "a" }]
    const result = reciprocalRankFusion(list1, list2)
    const scores = Object.fromEntries(result.map(r => [r.id, r.score]))
    // "b" is rank 1 in list1, rank 0 in list2 → highest score
    expect(scores["b"]).toBeGreaterThan(scores["c"])
    expect(scores["b"]).toBeGreaterThan(scores["d"])
  })

  it("returns results sorted by score descending", () => {
    const list1 = [{ id: "x" }, { id: "y" }]
    const list2 = [{ id: "x" }, { id: "z" }]
    const result = reciprocalRankFusion(list1, list2)
    for (let i = 1; i < result.length; i++) {
      expect(result[i - 1].score).toBeGreaterThanOrEqual(result[i].score)
    }
  })

  it("includes documents appearing in only one list", () => {
    const list1 = [{ id: "a" }]
    const list2 = [{ id: "b" }]
    const result = reciprocalRankFusion(list1, list2)
    const ids = result.map(r => r.id)
    expect(ids).toContain("a")
    expect(ids).toContain("b")
  })
})
```

- [ ] **Step 2: Run to verify it fails**

```bash
npx vitest run tests/unit/rrf.test.ts
```
Expected: FAIL — `Cannot find module '../../src/retriever/hybrid.js'`

- [ ] **Step 3: Create `src/retriever/hybrid.ts`**

```typescript
import type { Collection } from "chromadb"
import type { BM25Index } from "./bm25.js"

export type EmbedFn = (text: string) => Promise<number[]>

export interface RetrievedChunk {
  id: string
  content: string
  source: string
  chunkIndex: number
  score: number
}

export function reciprocalRankFusion(
  list1: Array<{ id: string }>,
  list2: Array<{ id: string }>,
  k = 60
): Array<{ id: string; score: number }> {
  const scores = new Map<string, number>()
  list1.forEach(({ id }, rank) => scores.set(id, (scores.get(id) ?? 0) + 1 / (k + rank + 1)))
  list2.forEach(({ id }, rank) => scores.set(id, (scores.get(id) ?? 0) + 1 / (k + rank + 1)))
  return [...scores.entries()]
    .map(([id, score]) => ({ id, score }))
    .sort((a, b) => b.score - a.score)
}

export async function hybridSearch(
  query: string,
  topK: number,
  collection: Collection,
  bm25: BM25Index,
  embed: EmbedFn
): Promise<RetrievedChunk[]> {
  const totalDocs = await collection.count()
  if (totalDocs === 0) return []

  const fetchK = Math.min(topK * 2, totalDocs)
  const [embedding, bm25Results] = await Promise.all([
    embed(query),
    Promise.resolve(bm25.search(query, fetchK)),
  ])

  const vectorResults = await collection.query({
    queryEmbeddings: [embedding],
    nResults: fetchK,
    include: ["documents", "metadatas"] as any,
  })

  const vectorIds = (vectorResults.ids[0] ?? []).map(id => ({ id }))
  const ranked = reciprocalRankFusion(vectorIds, bm25Results).slice(0, topK)

  // Build lookup from vector results (they have the content)
  const docMap = new Map<string, { content: string; metadata: Record<string, unknown> }>()
  ;(vectorResults.ids[0] ?? []).forEach((id, i) => {
    docMap.set(id, {
      content: vectorResults.documents[0]?.[i] ?? "",
      metadata: (vectorResults.metadatas[0]?.[i] as Record<string, unknown>) ?? {},
    })
  })

  // Fetch any ids that only appeared in BM25 results
  const missing = ranked.filter(r => !docMap.has(r.id)).map(r => r.id)
  if (missing.length > 0) {
    const fetched = await collection.get({
      ids: missing,
      include: ["documents", "metadatas"] as any,
    })
    ;(fetched.ids ?? []).forEach((id, i) => {
      docMap.set(id, {
        content: fetched.documents?.[i] ?? "",
        metadata: (fetched.metadatas?.[i] as Record<string, unknown>) ?? {},
      })
    })
  }

  return ranked
    .map(({ id, score }) => {
      const doc = docMap.get(id)
      if (!doc) return null
      return {
        id,
        content: doc.content,
        source: (doc.metadata.source as string) ?? "",
        chunkIndex: (doc.metadata.chunkIndex as number) ?? 0,
        score,
      }
    })
    .filter((c): c is RetrievedChunk => c !== null)
}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
npx vitest run tests/unit/rrf.test.ts
```
Expected: 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/retriever/hybrid.ts tests/unit/rrf.test.ts
git commit -m "feat: hybrid retriever with RRF fusion and tests"
```

---

### Task 5: Indexer

**Files:**
- Create: `src/indexer/indexer.ts`
- Create: `tests/unit/indexer.test.ts`

**Interfaces:**
- Consumes: `loadFile`, `scanFolder` (Task 2), `chunkDocument` (Task 2), `BM25Index` (Task 3), `EmbedFn` (Task 4)
- Produces:
  - `IndexResult { filesIndexed: number; chunksCreated: number; duration: number }`
  - `IndexStatus { fileCount: number; chunkCount: number; lastIndexed: string | null; status: "empty" | "ready" }`
  - `Indexer` class with `.indexFolder(path, glob?)`, `.getStatus()`, `.loadBM25FromChroma()`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/indexer.test.ts`:

```typescript
import { describe, it, expect, vi, beforeEach } from "vitest"
import { Indexer } from "../../src/indexer/indexer.js"
import { BM25Index } from "../../src/retriever/bm25.js"

// Mock loaders and chunker
vi.mock("../../src/indexer/loaders.js", () => ({
  scanFolder: vi.fn().mockResolvedValue(["/docs/test.md"]),
  loadFile: vi.fn().mockResolvedValue({ content: "test content about dragons", filePath: "/docs/test.md" }),
}))
vi.mock("../../src/indexer/chunker.js", () => ({
  chunkDocument: vi.fn().mockResolvedValue([
    { content: "test content about dragons", metadata: { source: "/docs/test.md", chunkIndex: 0, totalChunks: 1 } },
  ]),
}))

function makeCollection(count = 0) {
  return {
    count: vi.fn().mockResolvedValue(count),
    add: vi.fn().mockResolvedValue(undefined),
    delete: vi.fn().mockResolvedValue(undefined),
    get: vi.fn().mockResolvedValue({ ids: [], documents: [], metadatas: [] }),
  }
}

describe("Indexer", () => {
  it("returns correct IndexResult after indexing", async () => {
    const collection = makeCollection(0)
    const bm25 = new BM25Index()
    const embed = vi.fn().mockResolvedValue([0.1, 0.2, 0.3])
    const indexer = new Indexer(collection as any, bm25, embed)

    const result = await indexer.indexFolder("/sample_docs")

    expect(result.filesIndexed).toBe(1)
    expect(result.chunksCreated).toBe(1)
    expect(result.duration).toBeGreaterThanOrEqual(0)
  })

  it("updates BM25 index with indexed chunks", async () => {
    const collection = makeCollection(0)
    const bm25 = new BM25Index()
    const embed = vi.fn().mockResolvedValue([0.1, 0.2, 0.3])
    const indexer = new Indexer(collection as any, bm25, embed)

    await indexer.indexFolder("/sample_docs")

    const results = bm25.search("dragons", 3)
    expect(results.length).toBeGreaterThan(0)
  })

  it("getStatus returns empty status when nothing is indexed", async () => {
    const collection = makeCollection(0)
    const bm25 = new BM25Index()
    const embed = vi.fn()
    const indexer = new Indexer(collection as any, bm25, embed)

    const status = await indexer.getStatus()

    expect(status.status).toBe("empty")
    expect(status.chunkCount).toBe(0)
    expect(status.lastIndexed).toBeNull()
  })

  it("getStatus returns ready status after indexing", async () => {
    const collection = makeCollection(1)
    const bm25 = new BM25Index()
    const embed = vi.fn().mockResolvedValue([0.1])
    const indexer = new Indexer(collection as any, bm25, embed)

    await indexer.indexFolder("/sample_docs")
    const status = await indexer.getStatus()

    expect(status.status).toBe("ready")
    expect(status.lastIndexed).not.toBeNull()
  })
})
```

- [ ] **Step 2: Run to verify it fails**

```bash
npx vitest run tests/unit/indexer.test.ts
```
Expected: FAIL — `Cannot find module '../../src/indexer/indexer.js'`

- [ ] **Step 3: Create `src/indexer/indexer.ts`**

```typescript
import crypto from "crypto"
import type { Collection } from "chromadb"
import { scanFolder, loadFile } from "./loaders.js"
import { chunkDocument } from "./chunker.js"
import type { BM25Index } from "../retriever/bm25.js"
import type { EmbedFn } from "../retriever/hybrid.js"

export interface IndexResult {
  filesIndexed: number
  chunksCreated: number
  duration: number
}

export interface IndexStatus {
  fileCount: number
  chunkCount: number
  lastIndexed: string | null
  status: "empty" | "ready"
}

export class Indexer {
  private lastIndexed: string | null = null
  private fileCount = 0

  constructor(
    private collection: Collection,
    private bm25: BM25Index,
    private embed: EmbedFn
  ) {}

  async indexFolder(folderPath: string, globPattern?: string): Promise<IndexResult> {
    const start = Date.now()
    const files = await scanFolder(folderPath, globPattern)

    // Clear existing data
    const existing = await this.collection.count()
    if (existing > 0) {
      const all = await this.collection.get({})
      if (all.ids.length > 0) await this.collection.delete({ ids: all.ids })
    }

    const bm25Docs: Array<{ id: string; text: string }> = []
    let chunksCreated = 0

    for (const filePath of files) {
      const doc = await loadFile(filePath)
      const chunks = await chunkDocument(doc)
      const ids = chunks.map(() => crypto.randomUUID())
      const embeddings = await Promise.all(chunks.map(c => this.embed(c.content)))

      await this.collection.add({
        ids,
        documents: chunks.map(c => c.content),
        embeddings,
        metadatas: chunks.map(c => c.metadata as Record<string, string | number | boolean>),
      })

      ids.forEach((id, i) => bm25Docs.push({ id, text: chunks[i].content }))
      chunksCreated += chunks.length
    }

    this.bm25.rebuild(bm25Docs)
    this.fileCount = files.length
    this.lastIndexed = new Date().toISOString()

    return { filesIndexed: files.length, chunksCreated, duration: Date.now() - start }
  }

  async getStatus(): Promise<IndexStatus> {
    const chunkCount = await this.collection.count()
    return {
      fileCount: this.fileCount,
      chunkCount,
      lastIndexed: this.lastIndexed,
      status: chunkCount > 0 ? "ready" : "empty",
    }
  }

  async loadBM25FromChroma(): Promise<void> {
    const all = await this.collection.get({ include: ["documents", "metadatas"] as any })
    const docs = (all.ids ?? []).map((id, i) => ({
      id,
      text: all.documents?.[i] ?? "",
    }))
    this.bm25.rebuild(docs)
    this.fileCount = new Set(
      (all.metadatas ?? []).map(m => (m as Record<string, unknown>)?.source as string)
    ).size
  }
}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
npx vitest run tests/unit/indexer.test.ts
```
Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/indexer/indexer.ts tests/unit/indexer.test.ts
git commit -m "feat: indexer with ChromaDB + BM25 sync and tests"
```

---

### Task 6: RAG State & Nodes

**Files:**
- Create: `src/rag/state.ts`
- Create: `src/rag/nodes/rewrite-query.ts`
- Create: `src/rag/nodes/retrieve.ts`
- Create: `src/rag/nodes/grade-chunks.ts`
- Create: `src/rag/nodes/broaden-query.ts`
- Create: `src/rag/nodes/generate.ts`
- Create: `tests/unit/graph.test.ts` (node-level tests, graph-level tests added in Task 7)

**Interfaces:**
- Consumes: `hybridSearch`, `EmbedFn` (Task 4), `BM25Index` (Task 3), `Config` (Task 1)
- Produces:
  - `RAGStateAnnotation` (LangGraph state)
  - `RAGState` type alias
  - `RetrievedChunk` type (re-exported from `hybrid.ts`)
  - `createRewriteQueryNode(ollama, config): NodeFn`
  - `createRetrieveNode(collection, bm25, embed, config): NodeFn`
  - `createGradeChunksNode(ollama, config): NodeFn`
  - `createBroadenQueryNode(ollama, config): NodeFn`
  - `createGenerateNode(ollama, config): NodeFn`

- [ ] **Step 1: Create `src/rag/state.ts`**

```typescript
import { Annotation } from "@langchain/langgraph"
import type { RetrievedChunk } from "../retriever/hybrid.js"

export type { RetrievedChunk }

export const RAGStateAnnotation = Annotation.Root({
  originalQuery: Annotation<string>(),
  rewrittenQuery: Annotation<string>({ reducer: (_, b) => b, default: () => "" }),
  chunks: Annotation<RetrievedChunk[]>({ reducer: (_, b) => b, default: () => [] }),
  relevantChunks: Annotation<RetrievedChunk[]>({ reducer: (_, b) => b, default: () => [] }),
  answer: Annotation<string>({ reducer: (_, b) => b, default: () => "" }),
  sources: Annotation<string[]>({ reducer: (_, b) => b, default: () => [] }),
  retryCount: Annotation<number>({ reducer: (_, b) => b, default: () => 0 }),
})

export type RAGState = typeof RAGStateAnnotation.State
```

- [ ] **Step 2: Create `src/rag/nodes/rewrite-query.ts`**

```typescript
import type { Ollama } from "ollama"
import type { Config } from "../../config.js"
import type { RAGState } from "../state.js"

export function createRewriteQueryNode(ollama: Ollama, config: Config) {
  return async (state: RAGState): Promise<Partial<RAGState>> => {
    const prompt = `You are a search query optimizer. The user asked a question in Russian.
Your task: translate it to English and rewrite it to maximize document retrieval quality.
Output ONLY the English search query — no explanations, no quotes, just the query text.

Russian question: ${state.originalQuery}`

    const response = await ollama.generate({ model: config.llmModel, prompt, stream: false })
    return { rewrittenQuery: response.response.trim() }
  }
}
```

- [ ] **Step 3: Create `src/rag/nodes/retrieve.ts`**

```typescript
import type { Collection } from "chromadb"
import type { BM25Index } from "../../retriever/bm25.js"
import type { EmbedFn } from "../../retriever/hybrid.js"
import { hybridSearch } from "../../retriever/hybrid.js"
import type { Config } from "../../config.js"
import type { RAGState } from "../state.js"

export function createRetrieveNode(
  collection: Collection,
  bm25: BM25Index,
  embed: EmbedFn,
  config: Config
) {
  return async (state: RAGState): Promise<Partial<RAGState>> => {
    const chunks = await hybridSearch(
      state.rewrittenQuery,
      config.retrievalTopK,
      collection,
      bm25,
      embed
    )
    return { chunks }
  }
}
```

- [ ] **Step 4: Create `src/rag/nodes/grade-chunks.ts`**

```typescript
import type { Ollama } from "ollama"
import type { Config } from "../../config.js"
import type { RAGState, RetrievedChunk } from "../state.js"

export function createGradeChunksNode(ollama: Ollama, config: Config) {
  return async (state: RAGState): Promise<Partial<RAGState>> => {
    const relevantChunks: RetrievedChunk[] = []

    for (const chunk of state.chunks) {
      const prompt = `You are a relevance grader. The user asked a question (in Russian) and you have a document chunk (in English).
Determine if the chunk contains information relevant to answering the question.
Respond with ONLY "yes" or "no".

Question (Russian): ${state.originalQuery}
Document chunk (English): ${chunk.content}`

      const response = await ollama.generate({ model: config.llmModel, prompt, stream: false })
      if (response.response.trim().toLowerCase().startsWith("yes")) {
        relevantChunks.push(chunk)
      }
    }

    return { relevantChunks }
  }
}
```

- [ ] **Step 5: Create `src/rag/nodes/broaden-query.ts`**

```typescript
import type { Ollama } from "ollama"
import type { Config } from "../../config.js"
import type { RAGState } from "../state.js"

export function createBroadenQueryNode(ollama: Ollama, config: Config) {
  return async (state: RAGState): Promise<Partial<RAGState>> => {
    const prompt = `You are a search query optimizer. The previous search query did not find enough relevant results.
Broaden the query by adding synonyms, related terms, or alternative phrasings in English.
Output ONLY the new broader English search query — no explanations.

Original question (Russian): ${state.originalQuery}
Previous search query (English): ${state.rewrittenQuery}`

    const response = await ollama.generate({ model: config.llmModel, prompt, stream: false })
    return {
      rewrittenQuery: response.response.trim(),
      retryCount: state.retryCount + 1,
    }
  }
}
```

- [ ] **Step 6: Create `src/rag/nodes/generate.ts`**

```typescript
import type { Ollama } from "ollama"
import type { Config } from "../../config.js"
import type { RAGState } from "../state.js"

export function createGenerateNode(ollama: Ollama, config: Config) {
  return async (state: RAGState): Promise<Partial<RAGState>> => {
    const chunksToUse = state.relevantChunks.length > 0 ? state.relevantChunks : state.chunks
    const context = chunksToUse.map(c => c.content).join("\n\n---\n\n")
    const sources = [...new Set(chunksToUse.map(c => c.source))]

    const prompt = `You are a helpful assistant. Answer the user's question in Russian based on the provided context.
If the context does not fully answer the question, say so honestly in Russian.
Do not use external knowledge — only the context provided.

Context (English documents):
${context}

Question (Russian): ${state.originalQuery}

Answer in Russian:`

    const response = await ollama.generate({ model: config.llmModel, prompt, stream: false })
    return { answer: response.response.trim(), sources }
  }
}
```

- [ ] **Step 7: Write node-level tests**

Add to `tests/unit/graph.test.ts`:

```typescript
import { describe, it, expect, vi } from "vitest"
import { createRewriteQueryNode } from "../../src/rag/nodes/rewrite-query.js"
import { createGradeChunksNode } from "../../src/rag/nodes/grade-chunks.js"
import { createGenerateNode } from "../../src/rag/nodes/generate.js"
import type { Config } from "../../src/config.js"
import type { RAGState } from "../../src/rag/state.js"

const config: Config = {
  chromadbUrl: "", ollamaUrl: "", llmModel: "test-model", embedModel: "",
  mcpPort: 3000, chromaCollection: "test", retrievalTopK: 6,
  minRelevantChunks: 2, maxRetries: 2,
}

function makeOllama(responseText: string) {
  return { generate: vi.fn().mockResolvedValue({ response: responseText }) } as any
}

const baseState: RAGState = {
  originalQuery: "что такое Void Protocol?",
  rewrittenQuery: "",
  chunks: [],
  relevantChunks: [],
  answer: "",
  sources: [],
  retryCount: 0,
}

describe("rewriteQueryNode", () => {
  it("sets rewrittenQuery from LLM response", async () => {
    const node = createRewriteQueryNode(makeOllama("What is Void Protocol?"), config)
    const result = await node(baseState)
    expect(result.rewrittenQuery).toBe("What is Void Protocol?")
  })
})

describe("gradeChunksNode", () => {
  it("keeps relevant chunks and discards irrelevant ones", async () => {
    const ollama = { generate: vi.fn()
      .mockResolvedValueOnce({ response: "yes" })
      .mockResolvedValueOnce({ response: "no" })
    } as any
    const node = createGradeChunksNode(ollama, config)
    const state: RAGState = {
      ...baseState,
      rewrittenQuery: "Void Protocol game",
      chunks: [
        { id: "1", content: "Void Protocol is a space station game", source: "a.md", chunkIndex: 0, score: 1 },
        { id: "2", content: "Recipe for chocolate cake", source: "b.md", chunkIndex: 0, score: 0.5 },
      ],
    }
    const result = await node(state)
    expect(result.relevantChunks).toHaveLength(1)
    expect(result.relevantChunks![0].id).toBe("1")
  })
})

describe("generateNode", () => {
  it("returns answer in the format provided by LLM", async () => {
    const node = createGenerateNode(makeOllama("Void Protocol — это игра о космической станции."), config)
    const state: RAGState = {
      ...baseState,
      relevantChunks: [
        { id: "1", content: "Void Protocol is a space station game", source: "docs/lore.md", chunkIndex: 0, score: 1 },
      ],
    }
    const result = await node(state)
    expect(result.answer).toBe("Void Protocol — это игра о космической станции.")
    expect(result.sources).toEqual(["docs/lore.md"])
  })
})
```

- [ ] **Step 8: Run tests to verify they pass**

```bash
npx vitest run tests/unit/graph.test.ts
```
Expected: 3 tests PASS.

- [ ] **Step 9: Commit**

```bash
git add src/rag/state.ts src/rag/nodes/ tests/unit/graph.test.ts
git commit -m "feat: RAG state type and LangGraph nodes with unit tests"
```

---

### Task 7: RAG Graph (LangGraph Assembly)

**Files:**
- Create: `src/rag/graph.ts`
- Modify: `tests/unit/graph.test.ts` (add retry-cycle tests)

**Interfaces:**
- Consumes: all nodes from Task 6, `RAGStateAnnotation` (Task 6), `Config` (Task 1)
- Produces:
  - `createRAGGraph(ollama, collection, bm25, embed, config): CompiledGraph`
  - `runRAGQuery(graph, question): Promise<{ answer: string; sources: string[]; confidence: "high" | "low" }>`

- [ ] **Step 1: Write failing retry-logic tests**

Append to `tests/unit/graph.test.ts`:

```typescript
import { createRAGGraph, runRAGQuery } from "../../src/rag/graph.js"
import { BM25Index } from "../../src/retriever/bm25.js"

describe("RAG graph retry logic", () => {
  function makeGraph(gradeResponses: string[], generateResponse = "Ответ на русском.") {
    const generateMock = vi.fn().mockResolvedValue({ response: generateResponse })
    const gradeGen = (function* () { for (const r of gradeResponses) yield r })()

    const ollama = {
      generate: vi.fn().mockImplementation(({ prompt }: { prompt: string }) => {
        if (prompt.includes("relevance grader")) {
          return Promise.resolve({ response: gradeGen.next().value ?? "no" })
        }
        if (prompt.includes("Answer in Russian")) {
          return generateMock()
        }
        // rewrite / broaden
        return Promise.resolve({ response: "Void Protocol game" })
      }),
    } as any

    const collection = {
      count: vi.fn().mockResolvedValue(3),
      query: vi.fn().mockResolvedValue({
        ids: [["id1", "id2", "id3"]],
        documents: [["chunk1 about void", "chunk2 about station", "chunk3 unrelated"]],
        metadatas: [[
          { source: "lore.md", chunkIndex: 0 },
          { source: "lore.md", chunkIndex: 1 },
          { source: "other.md", chunkIndex: 0 },
        ]],
      }),
      get: vi.fn().mockResolvedValue({ ids: [], documents: [], metadatas: [] }),
    } as any

    const bm25 = new BM25Index()
    const embed = vi.fn().mockResolvedValue([0.1, 0.2])

    return createRAGGraph(ollama, collection, bm25, embed, config)
  }

  it("returns confidence=high when enough relevant chunks found on first try", async () => {
    // All 3 chunks graded "yes"
    const graph = makeGraph(["yes", "yes", "yes"])
    const result = await runRAGQuery(graph, "что такое void protocol?")
    expect(result.confidence).toBe("high")
    expect(result.answer).toBeTruthy()
  })

  it("retries when too few relevant chunks and returns after retry", async () => {
    // First retrieve: 0 relevant → broaden → second retrieve: 2 relevant
    const graph = makeGraph(["no", "no", "no", "yes", "yes", "yes"])
    const result = await runRAGQuery(graph, "что такое void protocol?")
    expect(result.answer).toBeTruthy()
  })

  it("returns confidence=low when retries exhausted with 0 relevant chunks", async () => {
    // All grades "no" across all retries
    const graph = makeGraph(Array(18).fill("no"))
    const result = await runRAGQuery(graph, "что такое void protocol?")
    expect(result.confidence).toBe("low")
  })
})
```

- [ ] **Step 2: Run to verify new tests fail**

```bash
npx vitest run tests/unit/graph.test.ts
```
Expected: first 3 tests PASS, new 3 tests FAIL — `Cannot find module '../../src/rag/graph.js'`

- [ ] **Step 3: Create `src/rag/graph.ts`**

```typescript
import { StateGraph, START, END } from "@langchain/langgraph"
import type { Ollama } from "ollama"
import type { Collection } from "chromadb"
import type { BM25Index } from "../retriever/bm25.js"
import type { EmbedFn } from "../retriever/hybrid.js"
import type { Config } from "../config.js"
import { RAGStateAnnotation, type RAGState } from "./state.js"
import { createRewriteQueryNode } from "./nodes/rewrite-query.js"
import { createRetrieveNode } from "./nodes/retrieve.js"
import { createGradeChunksNode } from "./nodes/grade-chunks.js"
import { createBroadenQueryNode } from "./nodes/broaden-query.js"
import { createGenerateNode } from "./nodes/generate.js"

function routeAfterGrading(state: RAGState, config: Config): string {
  if (state.relevantChunks.length >= config.minRelevantChunks || state.retryCount >= config.maxRetries) {
    return "generate"
  }
  return "broadenQuery"
}

export function createRAGGraph(
  ollama: Ollama,
  collection: Collection,
  bm25: BM25Index,
  embed: EmbedFn,
  config: Config
) {
  const graph = new StateGraph(RAGStateAnnotation)
    .addNode("rewriteQuery", createRewriteQueryNode(ollama, config))
    .addNode("retrieve", createRetrieveNode(collection, bm25, embed, config))
    .addNode("gradeChunks", createGradeChunksNode(ollama, config))
    .addNode("broadenQuery", createBroadenQueryNode(ollama, config))
    .addNode("generate", createGenerateNode(ollama, config))
    .addEdge(START, "rewriteQuery")
    .addEdge("rewriteQuery", "retrieve")
    .addEdge("retrieve", "gradeChunks")
    .addConditionalEdges("gradeChunks", (state: RAGState) => routeAfterGrading(state, config))
    .addEdge("broadenQuery", "retrieve")
    .addEdge("generate", END)

  return graph.compile()
}

export async function runRAGQuery(
  graph: ReturnType<typeof createRAGGraph>,
  question: string
): Promise<{ answer: string; sources: string[]; confidence: "high" | "low" }> {
  const result = await graph.invoke({ originalQuery: question })
  const confidence: "high" | "low" =
    result.relevantChunks.length >= 2 ? "high" : "low"
  return { answer: result.answer, sources: result.sources, confidence }
}
```

- [ ] **Step 4: Run all tests**

```bash
npx vitest run tests/unit/graph.test.ts
```
Expected: all 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/rag/graph.ts tests/unit/graph.test.ts
git commit -m "feat: LangGraph Corrective RAG graph with retry logic and tests"
```

---

### Task 8: MCP Tool Handlers

**Files:**
- Create: `src/tools/index-folder.ts`
- Create: `src/tools/ask-question.ts`
- Create: `src/tools/find-relevant-docs.ts`
- Create: `src/tools/index-status.ts`

**Interfaces:**
- Consumes: `Indexer` (Task 5), `createRAGGraph`/`runRAGQuery` (Task 7), `hybridSearch` (Task 4)
- Produces: `register*Tool(server, ...)` functions that register tools on an `McpServer` instance

- [ ] **Step 1: Create `src/tools/index-folder.ts`**

```typescript
import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js"
import { z } from "zod"
import type { Indexer } from "../indexer/indexer.js"

export function registerIndexFolderTool(server: McpServer, indexer: Indexer): void {
  server.tool(
    "index_folder",
    "Index documents from a local folder into the RAG knowledge base. " +
    "Use when the user wants to add, update, or reindex documents from a directory. " +
    "Supports .md, .txt, .py, .js, .ts, .json, .yaml files. " +
    "Call this before using ask_question or find_relevant_docs.",
    {
      folderPath: z.string().describe("Absolute or relative path to the folder to index"),
      globPattern: z.string().optional().describe("Glob pattern to filter files, e.g. '**/*.md'"),
    },
    async ({ folderPath, globPattern }) => {
      try {
        const result = await indexer.indexFolder(folderPath, globPattern)
        return {
          content: [{
            type: "text" as const,
            text: JSON.stringify({
              success: true,
              filesIndexed: result.filesIndexed,
              chunksCreated: result.chunksCreated,
              durationMs: result.duration,
            }),
          }],
        }
      } catch (err) {
        return {
          content: [{
            type: "text" as const,
            text: JSON.stringify({ success: false, error: (err as Error).message }),
          }],
          isError: true,
        }
      }
    }
  )
}
```

- [ ] **Step 2: Create `src/tools/index-status.ts`**

```typescript
import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js"
import type { Indexer } from "../indexer/indexer.js"

export function registerIndexStatusTool(server: McpServer, indexer: Indexer): void {
  server.tool(
    "index_status",
    "Return statistics about the current knowledge base index: file count, chunk count, " +
    "last indexed time. Use to check whether documents are indexed before querying, " +
    "or to verify that indexing completed successfully.",
    {},
    async () => {
      const status = await indexer.getStatus()
      return {
        content: [{ type: "text" as const, text: JSON.stringify(status) }],
      }
    }
  )
}
```

- [ ] **Step 3: Create `src/tools/ask-question.ts`**

```typescript
import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js"
import { z } from "zod"
import type { createRAGGraph } from "../rag/graph.js"
import { runRAGQuery } from "../rag/graph.js"

export function registerAskQuestionTool(
  server: McpServer,
  graph: ReturnType<typeof createRAGGraph>
): void {
  server.tool(
    "ask_question",
    "Answer a question using the indexed knowledge base via the Corrective RAG pipeline. " +
    "Use when the user asks anything about the content of indexed documents — " +
    "project documentation, wiki, notes, or source code. " +
    "Accepts questions in any language (including Russian) and returns an answer in the same language with source references. " +
    "Requires documents to be indexed first via index_folder.",
    {
      question: z.string().describe("The question to answer, in any language"),
    },
    async ({ question }) => {
      try {
        const result = await runRAGQuery(graph, question)
        return {
          content: [{
            type: "text" as const,
            text: JSON.stringify({
              answer: result.answer,
              sources: result.sources,
              confidence: result.confidence,
            }),
          }],
        }
      } catch (err) {
        return {
          content: [{
            type: "text" as const,
            text: JSON.stringify({ error: (err as Error).message }),
          }],
          isError: true,
        }
      }
    }
  )
}
```

- [ ] **Step 4: Create `src/tools/find-relevant-docs.ts`**

```typescript
import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js"
import { z } from "zod"
import type { Collection } from "chromadb"
import type { BM25Index } from "../retriever/bm25.js"
import type { EmbedFn } from "../retriever/hybrid.js"
import { hybridSearch } from "../retriever/hybrid.js"

export function registerFindRelevantDocsTool(
  server: McpServer,
  collection: Collection,
  bm25: BM25Index,
  embed: EmbedFn,
  defaultTopK: number
): void {
  server.tool(
    "find_relevant_docs",
    "Search the knowledge base and return ranked document chunks without generating an answer. " +
    "Use for exploring what is in the index, finding specific passages, " +
    "or when raw search results are needed instead of a synthesized answer. " +
    "Returns chunks ranked by hybrid BM25 + vector similarity score.",
    {
      query: z.string().describe("Search query in any language"),
      topK: z.number().int().positive().optional().describe("Number of chunks to return (default 5)"),
    },
    async ({ query, topK }) => {
      try {
        const chunks = await hybridSearch(query, topK ?? defaultTopK, collection, bm25, embed)
        return {
          content: [{
            type: "text" as const,
            text: JSON.stringify({
              chunks: chunks.map(c => ({
                content: c.content,
                source: c.source,
                chunkIndex: c.chunkIndex,
                score: c.score,
              })),
            }),
          }],
        }
      } catch (err) {
        return {
          content: [{
            type: "text" as const,
            text: JSON.stringify({ error: (err as Error).message }),
          }],
          isError: true,
        }
      }
    }
  )
}
```

- [ ] **Step 5: Verify compilation**

```bash
npm run build
```
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add src/tools/
git commit -m "feat: 4 MCP tool handlers with rich descriptions for agent autodiscovery"
```

---

### Task 9: MCP Server & E2E Tests

**Files:**
- Create: `src/index.ts`
- Create: `tests/e2e/mcp-tools.test.ts`

**Interfaces:**
- Consumes: all tool handlers (Task 8), `Indexer` (Task 5), `createRAGGraph` (Task 7), `BM25Index` (Task 3), `getConfig` (Task 1)
- Produces: running Express+SSE MCP server, exportable `createApp()` for testing

**Pre-condition for e2e tests:** ChromaDB must be running on `localhost:8000`. Start it with `docker compose up -d chromadb`.

- [ ] **Step 1: Write failing e2e tests**

Create `tests/e2e/mcp-tools.test.ts`:

```typescript
import { describe, it, expect, beforeAll, afterAll, vi } from "vitest"
import { ChromaClient } from "chromadb"
import path from "path"
import { fileURLToPath } from "url"
import { BM25Index } from "../../src/retriever/bm25.js"
import { Indexer } from "../../src/indexer/indexer.js"
import { createRAGGraph, runRAGQuery } from "../../src/rag/graph.js"

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const SAMPLE_DOCS = path.resolve(__dirname, "../../sample_docs")
const TEST_COLLECTION = "rag_test"

// Mock Ollama — deterministic responses for e2e
vi.mock("ollama", () => ({
  Ollama: vi.fn().mockImplementation(() => ({
    embeddings: vi.fn().mockResolvedValue({ embedding: Array(384).fill(0.1) }),
    generate: vi.fn().mockImplementation(({ prompt }: { prompt: string }) => {
      if (prompt.includes("relevance grader")) return Promise.resolve({ response: "yes" })
      if (prompt.includes("Answer in Russian")) return Promise.resolve({ response: "Тестовый ответ из базы знаний." })
      return Promise.resolve({ response: "void protocol game station" })
    }),
  })),
}))

describe("MCP tool handlers (e2e)", () => {
  let indexer: Indexer
  let bm25: BM25Index
  let collection: any

  beforeAll(async () => {
    const { Ollama } = await import("ollama")
    const client = new ChromaClient({ path: "http://localhost:8000" })

    // Clean slate
    try { await client.deleteCollection({ name: TEST_COLLECTION }) } catch {}
    collection = await client.getOrCreateCollection({ name: TEST_COLLECTION })

    const ollama = new Ollama()
    const embed = (text: string) => ollama.embeddings({ model: "nomic-embed-text", prompt: text }).then(r => r.embedding)
    bm25 = new BM25Index()
    indexer = new Indexer(collection, bm25, embed)
  })

  afterAll(async () => {
    const client = new ChromaClient({ path: "http://localhost:8000" })
    try { await client.deleteCollection({ name: TEST_COLLECTION }) } catch {}
  })

  it("index_status returns empty before indexing", async () => {
    const status = await indexer.getStatus()
    expect(status.status).toBe("empty")
    expect(status.chunkCount).toBe(0)
  })

  it("index_folder indexes sample_docs and returns file/chunk counts", async () => {
    const result = await indexer.indexFolder(SAMPLE_DOCS)
    expect(result.filesIndexed).toBeGreaterThan(0)
    expect(result.chunksCreated).toBeGreaterThan(0)
  })

  it("index_status returns ready after indexing with correct counts", async () => {
    const status = await indexer.getStatus()
    expect(status.status).toBe("ready")
    expect(status.chunkCount).toBeGreaterThan(0)
    expect(status.lastIndexed).not.toBeNull()
  })
})
```

- [ ] **Step 2: Run to verify it fails (requires ChromaDB)**

```bash
docker compose up -d chromadb 2>/dev/null || true
npx vitest run tests/e2e/mcp-tools.test.ts
```
Expected: FAIL — `Cannot find module '../../src/rag/graph.js'` in e2e context, or connection error.

- [ ] **Step 3: Create `src/index.ts`**

```typescript
import express from "express"
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js"
import { SSEServerTransport } from "@modelcontextprotocol/sdk/server/sse.js"
import { ChromaClient } from "chromadb"
import { Ollama } from "ollama"
import { getConfig } from "./config.js"
import { BM25Index } from "./retriever/bm25.js"
import { Indexer } from "./indexer/indexer.js"
import { createRAGGraph } from "./rag/graph.js"
import { registerIndexFolderTool } from "./tools/index-folder.js"
import { registerIndexStatusTool } from "./tools/index-status.js"
import { registerAskQuestionTool } from "./tools/ask-question.js"
import { registerFindRelevantDocsTool } from "./tools/find-relevant-docs.js"

async function main() {
  const config = getConfig()

  const chromaClient = new ChromaClient({ path: config.chromadbUrl })
  const collection = await chromaClient.getOrCreateCollection({ name: config.chromaCollection })

  const ollama = new Ollama({ host: config.ollamaUrl })
  const embed = (text: string) =>
    ollama.embeddings({ model: config.embedModel, prompt: text }).then(r => r.embedding)

  const bm25 = new BM25Index()
  const indexer = new Indexer(collection, bm25, embed)

  // Restore BM25 from existing ChromaDB data on startup
  await indexer.loadBM25FromChroma()

  const graph = createRAGGraph(ollama, collection, bm25, embed, config)

  const server = new McpServer({ name: "void-protocol-rag", version: "1.0.0" })
  registerIndexFolderTool(server, indexer)
  registerIndexStatusTool(server, indexer)
  registerAskQuestionTool(server, graph)
  registerFindRelevantDocsTool(server, collection, bm25, embed, config.retrievalTopK)

  const app = express()
  const transports: Record<string, SSEServerTransport> = {}

  app.get("/sse", async (req, res) => {
    const transport = new SSEServerTransport("/messages", res)
    transports[transport.sessionId] = transport
    await server.connect(transport)
    req.on("close", () => { delete transports[transport.sessionId] })
  })

  app.post("/messages", express.json(), async (req, res) => {
    const sessionId = req.query.sessionId as string
    const transport = transports[sessionId]
    if (transport) {
      await transport.handlePostMessage(req, res)
    } else {
      res.status(404).json({ error: "Session not found" })
    }
  })

  app.get("/health", (_, res) => res.json({ status: "ok" }))

  app.listen(config.mcpPort, () => {
    console.log(`MCP server running on http://localhost:${config.mcpPort}/sse`)
  })
}

main().catch(err => { console.error(err); process.exit(1) })
```

- [ ] **Step 4: Run e2e tests**

```bash
npx vitest run tests/e2e/mcp-tools.test.ts
```
Expected: 3 tests PASS.

- [ ] **Step 5: Run all tests**

```bash
npx vitest run
```
Expected: all tests PASS.

- [ ] **Step 6: Verify build**

```bash
npm run build
```
Expected: no errors.

- [ ] **Step 7: Commit**

```bash
git add src/index.ts tests/e2e/mcp-tools.test.ts
git commit -m "feat: MCP server with SSE transport and e2e tests"
```

---

### Task 10: Docker, CI & Documentation

**Files:**
- Create: `Dockerfile`
- Create: `docker-compose.yml`
- Create: `.dockerignore`
- Create: `mcp-config.json`
- Create: `.github/workflows/ci.yml`
- Create: `ARCHITECTURE.md`
- Create: `README.md`
- Create: `REPORT.md`

- [ ] **Step 1: Create `Dockerfile`**

```dockerfile
FROM node:20-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY tsconfig.json ./
COPY src ./src
RUN npm run build

FROM node:20-alpine
WORKDIR /app
COPY package*.json ./
RUN npm ci --omit=dev
COPY --from=builder /app/dist ./dist
EXPOSE 3000
CMD ["node", "dist/index.js"]
```

- [ ] **Step 2: Create `.dockerignore`**

```
node_modules
dist
.env
*.test.ts
tests/
.git
```

- [ ] **Step 3: Create `docker-compose.yml`**

```yaml
services:
  mcp-server:
    build: .
    ports:
      - "3000:3000"
    env_file:
      - .env
    environment:
      CHROMADB_URL: http://chromadb:8000
      OLLAMA_URL: http://ollama:11434
    depends_on:
      chromadb:
        condition: service_healthy
      ollama:
        condition: service_healthy
    restart: unless-stopped

  chromadb:
    image: chromadb/chroma:latest
    ports:
      - "8000:8000"
    volumes:
      - chroma_data:/chroma/chroma
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/v1/heartbeat"]
      interval: 10s
      timeout: 5s
      retries: 5

  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:11434/api/tags"]
      interval: 10s
      timeout: 5s
      retries: 10
    entrypoint: >
      sh -c "ollama serve &
             sleep 5 &&
             ollama pull ${LLM_MODEL:-phi3:mini} &&
             ollama pull ${EMBED_MODEL:-nomic-embed-text} &&
             wait"

volumes:
  chroma_data:
  ollama_data:
```

- [ ] **Step 4: Create `mcp-config.json`**

```json
{
  "mcpServers": {
    "void-protocol-rag": {
      "type": "sse",
      "url": "http://localhost:3000/sse"
    }
  }
}
```

- [ ] **Step 5: Create `.github/workflows/ci.yml`**

```yaml
name: CI

on:
  push:
    branches: [master, main]
  pull_request:

jobs:
  lint-and-test:
    runs-on: ubuntu-latest

    services:
      chromadb:
        image: chromadb/chroma:latest
        ports:
          - 8000:8000
        options: >-
          --health-cmd "curl -f http://localhost:8000/api/v1/heartbeat"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-node@v4
        with:
          node-version: 20
          cache: npm

      - run: npm ci

      - name: Type check
        run: npx tsc --noEmit

      - name: Unit tests
        run: npx vitest run tests/unit

      - name: E2E tests
        run: npx vitest run tests/e2e
        env:
          CHROMADB_URL: http://localhost:8000
```

- [ ] **Step 6: Create `ARCHITECTURE.md`** (brief, factual)

```markdown
# Architecture

## Overview

MCP server built with TypeScript and Express. Exposes 4 tools over SSE transport.
Implements Corrective RAG pattern using LangGraph for orchestration.

## Components

```
┌─────────────────────────────────────────────┐
│             MCP Client (IDE)                │
└──────────────────┬──────────────────────────┘
                   │ SSE (port 3000)
┌──────────────────▼──────────────────────────┐
│             MCP Server (Express)            │
│  ┌─────────┐ ┌──────────┐ ┌─────────────┐  │
│  │index_   │ │ask_      │ │find_        │  │
│  │folder   │ │question  │ │relevant_docs│  │
│  └────┬────┘ └────┬─────┘ └──────┬──────┘  │
│       │           │               │         │
│  ┌────▼───────────▼───────────────▼──────┐  │
│  │              Indexer                  │  │
│  └────────────────┬───────────────────── ┘  │
│                   │                         │
│  ┌────────────────▼────────────────────┐    │
│  │         LangGraph RAG Graph         │    │
│  │  rewrite→retrieve→grade→generate    │    │
│  └──────────────────────────────────── ┘    │
└───────────────┬─────────────────────────────┘
                │
    ┌───────────┴────────────┐
    ▼                        ▼
ChromaDB (vector)      Ollama (LLM + embed)
BM25 (in-memory)
```

## Corrective RAG Flow

```
RU Query → rewriteQuery (RU→EN) → retrieve (BM25+vector→RRF)
  → gradeChunks (LLM: relevant? yes/no)
    ├─ ≥2 relevant OR retries≥2 → generate (answer in RU) → return
    └─ <2 relevant AND retries<2 → broadenQuery → retrieve (loop)
```

## Hybrid Search

BM25 (keyword) + ChromaDB vector search → Reciprocal Rank Fusion (k=60).
BM25 index is held in memory and rebuilt from ChromaDB on server startup.
```

- [ ] **Step 7: Create `README.md`**

```markdown
# Void Protocol RAG — MCP Server

MCP server that turns a local folder of documents into a searchable knowledge base.
Uses Corrective RAG (LangGraph) with a local Ollama LLM and hybrid BM25+vector search.

## Quick Start

```bash
cp .env.example .env
docker compose up
```

First start downloads Ollama models (~3GB). Wait for all services to be healthy.

## Connect to VSCode Copilot

Copy `mcp-config.json` contents into your VSCode `settings.json` under `"github.copilot.chat.mcp.servers"`.

## Usage

```
index_folder("./sample_docs")        # index the demo documents
ask_question("что такое Void Protocol?")  # ask in Russian, get answer in Russian
find_relevant_docs("combat mechanics")    # raw chunk search
index_status()                            # check index state
```

## Verification Facts

The following facts exist only in the `sample_docs` knowledge base:

- The Void Station orbits planet **Kedraxis** at altitude **7,432 km**
- Server timezone is **UTC-3** (Kedraxis Standard Time)
- Maximum player capacity: **247 simultaneous connections**
- Founder of the Solis Archive: **Dr. Maren Voss**, year **2387**

Ask `ask_question("сколько игроков может подключиться одновременно?")` to verify RAG is reading your documents, not using general knowledge.

## Development

```bash
npm install
npm run build
npx vitest run
```

E2E tests require ChromaDB: `docker compose up -d chromadb`
```

- [ ] **Step 8: Create `REPORT.md`** (shell — fill in while building)

```markdown
# Development Report

## История создания

_Добавляйте записи по мере работы._

## Ключевые проблемы и решения

### Кросс-языковой поиск (RU запросы → EN документы)

...

## Примеры промптов

### Удачный пример

**Промпт для gradeChunks:**
```
You are a relevance grader. The user asked a question (in Russian) and you have a document chunk (in English).
...
```
**Почему сработало:** ...

### Неудачный пример

...

## Использованные инструменты

- Claude Code (claude-sonnet-4-6) — архитектура, план, код
```

- [ ] **Step 9: Run final test suite**

```bash
npx vitest run
npm run build
```
Expected: all tests PASS, build succeeds.

- [ ] **Step 10: Commit**

```bash
git add Dockerfile docker-compose.yml .dockerignore mcp-config.json .github/ ARCHITECTURE.md README.md REPORT.md
git commit -m "chore: Docker, CI pipeline, and project documentation"
```

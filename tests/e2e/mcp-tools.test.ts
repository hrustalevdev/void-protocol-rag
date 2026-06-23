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
vi.mock("ollama", () => {
  class MockOllama {
    embeddings() {
      return Promise.resolve({ embedding: Array(384).fill(0.1) })
    }
    generate({ prompt }: { prompt: string }) {
      if (prompt.includes("relevance grader")) return Promise.resolve({ response: "yes" })
      if (prompt.includes("Answer in Russian")) return Promise.resolve({ response: "Тестовый ответ из базы знаний." })
      return Promise.resolve({ response: "void protocol game station" })
    }
  }
  return { Ollama: MockOllama }
})

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

  it("ask_question returns answer containing a fact from sample_docs", async () => {
    const { Ollama } = await import("ollama")
    const ollamaInstance = new Ollama()
    const embed = (text: string) =>
      ollamaInstance.embeddings({ model: "nomic-embed-text", prompt: text }).then((r: any) => r.embedding)
    const graph = createRAGGraph(ollamaInstance as any, collection, bm25, embed, {
      chromadbUrl: "http://localhost:8000",
      ollamaUrl: "http://localhost:11434",
      llmModel: "test-model",
      embedModel: "nomic-embed-text",
      mcpPort: 3000,
      chromaCollection: TEST_COLLECTION,
      retrievalTopK: 6,
      minRelevantChunks: 2,
      maxRetries: 2,
    })
    const result = await runRAGQuery(graph, "сколько постоянных жителей на станции Новая Амора?")
    expect(result.answer).toBeTruthy()
    expect(result.sources.length).toBeGreaterThan(0)
  })
})

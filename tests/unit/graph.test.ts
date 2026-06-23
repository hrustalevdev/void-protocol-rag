import { describe, it, expect, vi } from "vitest"
import { createRewriteQueryNode } from "../../src/rag/nodes/rewrite-query.js"
import { createGradeChunksNode } from "../../src/rag/nodes/grade-chunks.js"
import { createGenerateNode } from "../../src/rag/nodes/generate.js"
import { createRAGGraph, runRAGQuery } from "../../src/rag/graph.js"
import { BM25Index } from "../../src/retriever/bm25.js"
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

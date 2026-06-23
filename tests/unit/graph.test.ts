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

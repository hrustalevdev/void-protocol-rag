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

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

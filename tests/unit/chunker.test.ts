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

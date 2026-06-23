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
      // Delete all existing docs without fetching IDs (avoids page-size truncation)
      try {
        await this.collection.delete({ where: { "chunkIndex": { "$gte": 0 } } })
      } catch {
        // Collection was empty — safe to proceed
      }
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
    const all = await this.collection.get({ include: ["documents", "metadatas"] as any, limit: 100000 })
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

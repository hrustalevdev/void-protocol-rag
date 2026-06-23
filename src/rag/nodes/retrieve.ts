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

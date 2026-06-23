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

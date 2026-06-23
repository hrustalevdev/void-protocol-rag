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

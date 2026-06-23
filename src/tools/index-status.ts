import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js"
import type { Indexer } from "../indexer/indexer.js"

export function registerIndexStatusTool(server: McpServer, indexer: Indexer): void {
  server.tool(
    "index_status",
    "Return statistics about the current knowledge base index: file count, chunk count, " +
    "last indexed time. Use to check whether documents are indexed before querying, " +
    "or to verify that indexing completed successfully.",
    {},
    async () => {
      try {
        const status = await indexer.getStatus()
        return {
          content: [{ type: "text" as const, text: JSON.stringify(status) }],
        }
      } catch (err) {
        return {
          content: [{ type: "text" as const, text: JSON.stringify({ error: (err as Error).message }) }],
          isError: true,
        }
      }
    }
  )
}

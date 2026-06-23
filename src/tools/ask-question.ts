import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js"
import { z } from "zod"
import type { createRAGGraph } from "../rag/graph.js"
import { runRAGQuery } from "../rag/graph.js"

export function registerAskQuestionTool(
  server: McpServer,
  graph: ReturnType<typeof createRAGGraph>
): void {
  server.tool(
    "ask_question",
    "Answer a question using the indexed knowledge base via the Corrective RAG pipeline. " +
    "Use when the user asks anything about the content of indexed documents — " +
    "project documentation, wiki, notes, or source code. " +
    "Accepts questions in any language (including Russian) and returns an answer in the same language with source references. " +
    "Requires documents to be indexed first via index_folder.",
    {
      question: z.string().describe("The question to answer, in any language"),
    },
    async ({ question }) => {
      try {
        const result = await runRAGQuery(graph, question)
        return {
          content: [{
            type: "text" as const,
            text: JSON.stringify({
              answer: result.answer,
              sources: result.sources,
              confidence: result.confidence,
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

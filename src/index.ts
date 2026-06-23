import express from "express"
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js"
import { SSEServerTransport } from "@modelcontextprotocol/sdk/server/sse.js"
import { ChromaClient } from "chromadb"
import { Ollama } from "ollama"
import { getConfig } from "./config.js"
import { BM25Index } from "./retriever/bm25.js"
import { Indexer } from "./indexer/indexer.js"
import { createRAGGraph } from "./rag/graph.js"
import { registerIndexFolderTool } from "./tools/index-folder.js"
import { registerIndexStatusTool } from "./tools/index-status.js"
import { registerAskQuestionTool } from "./tools/ask-question.js"
import { registerFindRelevantDocsTool } from "./tools/find-relevant-docs.js"

async function main() {
  const config = getConfig()

  const ollama = new Ollama({ host: config.ollamaUrl })
  const embed = (text: string) =>
    ollama.embeddings({ model: config.embedModel, prompt: text }).then(r => r.embedding)

  const chromaUrl = new URL(config.chromadbUrl)
  const chromaClient = new ChromaClient({
    host: chromaUrl.hostname,
    port: parseInt(chromaUrl.port) || 8000,
    ssl: chromaUrl.protocol === "https:",
  })
  const collection = await chromaClient.getOrCreateCollection({
    name: config.chromaCollection,
    embeddingFunction: { generate: (texts: string[]) => Promise.all(texts.map(embed)) },
  })

  const bm25 = new BM25Index()
  const indexer = new Indexer(collection, bm25, embed)

  // Restore BM25 from existing ChromaDB data on startup
  await indexer.loadBM25FromChroma()

  const graph = createRAGGraph(ollama, collection, bm25, embed, config)

  const server = new McpServer({ name: "void-protocol-rag", version: "1.0.0" })
  registerIndexFolderTool(server, indexer)
  registerIndexStatusTool(server, indexer)
  registerAskQuestionTool(server, graph)
  registerFindRelevantDocsTool(server, collection, bm25, embed, config.retrievalTopK)

  const app = express()
  const transports: Record<string, SSEServerTransport> = {}

  app.get("/sse", async (req, res) => {
    const transport = new SSEServerTransport("/messages", res)
    transports[transport.sessionId] = transport
    await server.connect(transport)
    req.on("close", () => { delete transports[transport.sessionId] })
  })

  app.post("/messages", express.json(), async (req, res) => {
    const sessionId = req.query.sessionId as string
    const transport = transports[sessionId]
    if (transport) {
      await transport.handlePostMessage(req, res)
    } else {
      res.status(404).json({ error: "Session not found" })
    }
  })

  app.get("/health", (_, res) => res.json({ status: "ok" }))

  app.listen(config.mcpPort, () => {
    console.log(`MCP server running on http://localhost:${config.mcpPort}/sse`)
  })
}

main().catch(err => { console.error(err); process.exit(1) })

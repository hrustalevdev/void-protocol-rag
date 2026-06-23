export interface Config {
  chromadbUrl: string
  ollamaUrl: string
  llmModel: string
  embedModel: string
  mcpPort: number
  chromaCollection: string
  retrievalTopK: number
  minRelevantChunks: number
  maxRetries: number
}

export function getConfig(): Config {
  return {
    chromadbUrl: process.env.CHROMADB_URL ?? "http://localhost:8000",
    ollamaUrl: process.env.OLLAMA_URL ?? "http://localhost:11434",
    llmModel: process.env.LLM_MODEL ?? "phi3:mini",
    embedModel: process.env.EMBED_MODEL ?? "nomic-embed-text",
    mcpPort: parseInt(process.env.MCP_PORT ?? "3000", 10),
    chromaCollection: process.env.CHROMA_COLLECTION ?? "rag_knowledge_base",
    retrievalTopK: parseInt(process.env.RETRIEVAL_TOP_K ?? "6", 10),
    minRelevantChunks: parseInt(process.env.MIN_RELEVANT_CHUNKS ?? "2", 10),
    maxRetries: parseInt(process.env.MAX_RETRIES ?? "2", 10),
  }
}

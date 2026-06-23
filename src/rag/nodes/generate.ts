import type { Ollama } from "ollama"
import type { Config } from "../../config.js"
import type { RAGState } from "../state.js"

export function createGenerateNode(ollama: Ollama, config: Config) {
  return async (state: RAGState): Promise<Partial<RAGState>> => {
    const chunksToUse = (state.bestRelevantChunks?.length ?? 0) > 0
      ? state.bestRelevantChunks
      : state.relevantChunks.length > 0
      ? state.relevantChunks
      : state.chunks
    const context = chunksToUse.map(c => c.content).join("\n\n---\n\n")
    const sources = [...new Set(chunksToUse.map(c => c.source))]

    const prompt = `You are a helpful assistant. Answer the user's question in Russian based on the provided context.
If the context does not fully answer the question, say so honestly in Russian.
Do not use external knowledge — only the context provided.

Context (English documents):
${context}

Question (Russian): ${state.originalQuery}

Answer in Russian:`

    const response = await ollama.generate({ model: config.llmModel, prompt, stream: false })
    return { answer: response.response.trim(), sources }
  }
}

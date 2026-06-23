import type { Ollama } from "ollama"
import type { Config } from "../../config.js"
import type { RAGState } from "../state.js"

export function createRewriteQueryNode(ollama: Ollama, config: Config) {
  return async (state: RAGState): Promise<Partial<RAGState>> => {
    const prompt = `You are a search query optimizer. The user asked a question in Russian.
Your task: translate it to English and rewrite it to maximize document retrieval quality.
Output ONLY the English search query — no explanations, no quotes, just the query text.

Russian question: ${state.originalQuery}`

    const response = await ollama.generate({ model: config.llmModel, prompt, stream: false })
    return { rewrittenQuery: response.response.trim() }
  }
}

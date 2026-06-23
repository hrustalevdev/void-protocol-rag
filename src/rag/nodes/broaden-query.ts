import type { Ollama } from "ollama"
import type { Config } from "../../config.js"
import type { RAGState } from "../state.js"

export function createBroadenQueryNode(ollama: Ollama, config: Config) {
  return async (state: RAGState): Promise<Partial<RAGState>> => {
    const prompt = `You are a search query optimizer. The previous search query did not find enough relevant results.
Broaden the query by adding synonyms, related terms, or alternative phrasings in English.
Output ONLY the new broader English search query — no explanations.

Original question (Russian): ${state.originalQuery}
Previous search query (English): ${state.rewrittenQuery}`

    const response = await ollama.generate({ model: config.llmModel, prompt, stream: false })
    return {
      rewrittenQuery: response.response.trim(),
      retryCount: state.retryCount + 1,
    }
  }
}

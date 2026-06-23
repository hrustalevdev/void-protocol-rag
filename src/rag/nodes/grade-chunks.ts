import type { Ollama } from "ollama"
import type { Config } from "../../config.js"
import type { RAGState, RetrievedChunk } from "../state.js"

export function createGradeChunksNode(ollama: Ollama, config: Config) {
  return async (state: RAGState): Promise<Partial<RAGState>> => {
    const relevantChunks: RetrievedChunk[] = []

    for (const chunk of state.chunks) {
      const prompt = `You are a relevance grader. The user asked a question (in Russian) and you have a document chunk (in English).
Determine if the chunk contains information relevant to answering the question.
Respond with ONLY "yes" or "no".

Question (Russian): ${state.originalQuery}
Document chunk (English): ${chunk.content}`

      const response = await ollama.generate({ model: config.llmModel, prompt, stream: false })
      if (response.response.trim().toLowerCase().startsWith("yes")) {
        relevantChunks.push(chunk)
      }
    }

    return { relevantChunks }
  }
}

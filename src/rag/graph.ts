import { StateGraph, START, END } from "@langchain/langgraph"
import type { Ollama } from "ollama"
import type { Collection } from "chromadb"
import type { BM25Index } from "../retriever/bm25.js"
import type { EmbedFn } from "../retriever/hybrid.js"
import type { Config } from "../config.js"
import { RAGStateAnnotation, type RAGState } from "./state.js"
import { createRewriteQueryNode } from "./nodes/rewrite-query.js"
import { createRetrieveNode } from "./nodes/retrieve.js"
import { createGradeChunksNode } from "./nodes/grade-chunks.js"
import { createBroadenQueryNode } from "./nodes/broaden-query.js"
import { createGenerateNode } from "./nodes/generate.js"

function routeAfterGrading(state: RAGState, config: Config): string {
  if (state.relevantChunks.length >= config.minRelevantChunks || state.retryCount >= config.maxRetries) {
    return "generate"
  }
  return "broadenQuery"
}

export function createRAGGraph(
  ollama: Ollama,
  collection: Collection,
  bm25: BM25Index,
  embed: EmbedFn,
  config: Config
) {
  const graph = new StateGraph(RAGStateAnnotation)
    .addNode("rewriteQuery", createRewriteQueryNode(ollama, config))
    .addNode("retrieve", createRetrieveNode(collection, bm25, embed, config))
    .addNode("gradeChunks", createGradeChunksNode(ollama, config))
    .addNode("broadenQuery", createBroadenQueryNode(ollama, config))
    .addNode("generate", createGenerateNode(ollama, config))
    .addEdge(START, "rewriteQuery")
    .addEdge("rewriteQuery", "retrieve")
    .addEdge("retrieve", "gradeChunks")
    .addConditionalEdges("gradeChunks", (state: RAGState) => routeAfterGrading(state, config))
    .addEdge("broadenQuery", "retrieve")
    .addEdge("generate", END)

  return graph.compile()
}

export async function runRAGQuery(
  graph: ReturnType<typeof createRAGGraph>,
  question: string
): Promise<{ answer: string; sources: string[]; confidence: "high" | "low" }> {
  const result = await graph.invoke({ originalQuery: question })
  const confidence: "high" | "low" =
    result.relevantChunks.length >= 2 ? "high" : "low"
  return { answer: result.answer, sources: result.sources, confidence }
}

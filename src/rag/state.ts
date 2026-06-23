import { Annotation } from "@langchain/langgraph"
import type { RetrievedChunk } from "../retriever/hybrid.js"

export type { RetrievedChunk }

export const RAGStateAnnotation = Annotation.Root({
  originalQuery: Annotation<string>(),
  rewrittenQuery: Annotation<string>({ reducer: (_, b) => b, default: () => "" }),
  chunks: Annotation<RetrievedChunk[]>({ reducer: (_, b) => b, default: () => [] }),
  relevantChunks: Annotation<RetrievedChunk[]>({ reducer: (_, b) => b, default: () => [] }),
  answer: Annotation<string>({ reducer: (_, b) => b, default: () => "" }),
  sources: Annotation<string[]>({ reducer: (_, b) => b, default: () => [] }),
  retryCount: Annotation<number>({ reducer: (_, b) => b, default: () => 0 }),
})

export type RAGState = typeof RAGStateAnnotation.State

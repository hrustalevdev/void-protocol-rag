import type { Collection } from "chromadb"
import type { BM25Index } from "./bm25.js"

export type EmbedFn = (text: string) => Promise<number[]>

export interface RetrievedChunk {
  id: string
  content: string
  source: string
  chunkIndex: number
  score: number
}

export function reciprocalRankFusion(
  list1: Array<{ id: string }>,
  list2: Array<{ id: string }>,
  k = 60
): Array<{ id: string; score: number }> {
  const scores = new Map<string, number>()
  list1.forEach(({ id }, rank) => scores.set(id, (scores.get(id) ?? 0) + 1 / (k + rank + 1)))
  list2.forEach(({ id }, rank) => scores.set(id, (scores.get(id) ?? 0) + 1 / (k + rank + 1)))
  return [...scores.entries()]
    .map(([id, score]) => ({ id, score }))
    .sort((a, b) => b.score - a.score)
}

export async function hybridSearch(
  query: string,
  topK: number,
  collection: Collection,
  bm25: BM25Index,
  embed: EmbedFn
): Promise<RetrievedChunk[]> {
  const totalDocs = await collection.count()
  if (totalDocs === 0) return []

  const fetchK = Math.min(topK * 2, totalDocs)
  const [embedding, bm25Results] = await Promise.all([
    embed(query),
    Promise.resolve(bm25.search(query, fetchK)),
  ])

  const vectorResults = await collection.query({
    queryEmbeddings: [embedding],
    nResults: fetchK,
    include: ["documents", "metadatas"] as any,
  })

  const vectorIds = (vectorResults.ids[0] ?? []).map(id => ({ id }))
  const ranked = reciprocalRankFusion(vectorIds, bm25Results).slice(0, topK)

  // Build lookup from vector results (they have the content)
  const docMap = new Map<string, { content: string; metadata: Record<string, unknown> }>()
  ;(vectorResults.ids[0] ?? []).forEach((id, i) => {
    docMap.set(id, {
      content: vectorResults.documents[0]?.[i] ?? "",
      metadata: (vectorResults.metadatas[0]?.[i] as Record<string, unknown>) ?? {},
    })
  })

  // Fetch any ids that only appeared in BM25 results
  const missing = ranked.filter(r => !docMap.has(r.id)).map(r => r.id)
  if (missing.length > 0) {
    const fetched = await collection.get({
      ids: missing,
      include: ["documents", "metadatas"] as any,
    })
    ;(fetched.ids ?? []).forEach((id, i) => {
      docMap.set(id, {
        content: fetched.documents?.[i] ?? "",
        metadata: (fetched.metadatas?.[i] as Record<string, unknown>) ?? {},
      })
    })
  }

  return ranked
    .map(({ id, score }) => {
      const doc = docMap.get(id)
      if (!doc) return null
      return {
        id,
        content: doc.content,
        source: (doc.metadata.source as string) ?? "",
        chunkIndex: (doc.metadata.chunkIndex as number) ?? 0,
        score,
      }
    })
    .filter((c): c is RetrievedChunk => c !== null)
}

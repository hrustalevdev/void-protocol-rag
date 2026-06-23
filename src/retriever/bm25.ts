export class BM25Index {
  private readonly k1 = 1.5
  private readonly b = 0.75
  private docs = new Map<string, string[]>()
  private df = new Map<string, number>()
  private avgdl = 0

  add(id: string, text: string): void {
    const tokens = this.tokenize(text)
    this.docs.set(id, tokens)
    new Set(tokens).forEach(t => this.df.set(t, (this.df.get(t) ?? 0) + 1))
  }

  buildIndex(): void {
    const total = [...this.docs.values()].reduce((s, t) => s + t.length, 0)
    this.avgdl = this.docs.size > 0 ? total / this.docs.size : 0
  }

  search(query: string, topK: number): Array<{ id: string; score: number }> {
    const qTerms = this.tokenize(query)
    const N = this.docs.size
    const scores = new Map<string, number>()

    for (const term of qTerms) {
      const n = this.df.get(term) ?? 0
      if (n === 0) continue
      const idf = Math.log((N - n + 0.5) / (n + 0.5) + 1)
      for (const [id, tokens] of this.docs) {
        const tf = tokens.filter(t => t === term).length
        if (tf === 0) continue
        const dl = tokens.length
        const tfs = (tf * (this.k1 + 1)) / (tf + this.k1 * (1 - this.b + (this.b * dl) / this.avgdl))
        scores.set(id, (scores.get(id) ?? 0) + idf * tfs)
      }
    }

    return [...scores.entries()]
      .map(([id, score]) => ({ id, score }))
      .sort((a, b) => b.score - a.score)
      .slice(0, topK)
  }

  rebuild(documents: Array<{ id: string; text: string }>): void {
    this.docs.clear()
    this.df.clear()
    documents.forEach(d => this.add(d.id, d.text))
    this.buildIndex()
  }

  get size(): number {
    return this.docs.size
  }

  private tokenize(text: string): string[] {
    return text.toLowerCase().match(/\b[a-z0-9]+\b/g) ?? []
  }
}

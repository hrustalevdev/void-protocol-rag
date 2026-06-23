import { describe, it, expect } from "vitest"
import { reciprocalRankFusion } from "../../src/retriever/hybrid.js"

describe("reciprocalRankFusion", () => {
  it("scores a document appearing in both lists higher than one in only one list", () => {
    const list1 = [{ id: "a" }, { id: "b" }, { id: "c" }]
    const list2 = [{ id: "b" }, { id: "d" }, { id: "a" }]
    const result = reciprocalRankFusion(list1, list2)
    const scores = Object.fromEntries(result.map(r => [r.id, r.score]))
    // "b" is rank 1 in list1, rank 0 in list2 → highest score
    expect(scores["b"]).toBeGreaterThan(scores["c"])
    expect(scores["b"]).toBeGreaterThan(scores["d"])
  })

  it("returns results sorted by score descending", () => {
    const list1 = [{ id: "x" }, { id: "y" }]
    const list2 = [{ id: "x" }, { id: "z" }]
    const result = reciprocalRankFusion(list1, list2)
    for (let i = 1; i < result.length; i++) {
      expect(result[i - 1].score).toBeGreaterThanOrEqual(result[i].score)
    }
  })

  it("includes documents appearing in only one list", () => {
    const list1 = [{ id: "a" }]
    const list2 = [{ id: "b" }]
    const result = reciprocalRankFusion(list1, list2)
    const ids = result.map(r => r.id)
    expect(ids).toContain("a")
    expect(ids).toContain("b")
  })
})

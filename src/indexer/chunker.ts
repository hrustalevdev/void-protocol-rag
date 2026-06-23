import path from "path"
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters"
import type { DocumentContent } from "./loaders.js"

export interface Chunk {
  content: string
  metadata: {
    source: string
    chunkIndex: number
    totalChunks: number
  }
}

const CODE_EXTS = new Set([".py", ".js", ".ts"])

const textSplitter = new RecursiveCharacterTextSplitter({ chunkSize: 512, chunkOverlap: 64 })

const codeSplitter = new RecursiveCharacterTextSplitter({
  chunkSize: 256,
  chunkOverlap: 32,
  separators: ["\nclass ", "\nfunction ", "\nconst ", "\nexport ", "\n\n", "\n", " ", ""],
})

export async function chunkDocument(doc: DocumentContent): Promise<Chunk[]> {
  const ext = path.extname(doc.filePath).toLowerCase()
  const splitter = CODE_EXTS.has(ext) ? codeSplitter : textSplitter
  const texts = await splitter.splitText(doc.content)
  return texts.map((content, i) => ({
    content,
    metadata: { source: doc.filePath, chunkIndex: i, totalChunks: texts.length },
  }))
}

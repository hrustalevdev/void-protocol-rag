import fs from "fs/promises"
import path from "path"
import { glob } from "glob"
import yaml from "js-yaml"

export interface DocumentContent {
  content: string
  filePath: string
}

const SUPPORTED_EXT = new Set([".md", ".txt", ".py", ".js", ".ts", ".json", ".yaml", ".yml"])

export async function loadFile(filePath: string): Promise<DocumentContent> {
  const ext = path.extname(filePath).toLowerCase()
  if (!SUPPORTED_EXT.has(ext)) throw new Error(`Unsupported file type: ${ext}`)

  const raw = await fs.readFile(filePath, "utf-8")
  let content = raw

  if (ext === ".json") {
    content = JSON.stringify(JSON.parse(raw), null, 2)
  } else if (ext === ".yaml" || ext === ".yml") {
    content = yaml.dump(yaml.load(raw))
  }

  return { content, filePath }
}

export async function scanFolder(folderPath: string, globPattern = "**/*"): Promise<string[]> {
  const files = await glob(globPattern, { cwd: folderPath, absolute: true, nodir: true })
  return files.filter(f => SUPPORTED_EXT.has(path.extname(f).toLowerCase()))
}

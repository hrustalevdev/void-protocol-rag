# Void Protocol RAG — MCP Server

MCP server that turns a local folder of documents into a searchable knowledge base.
Uses Corrective RAG (LangGraph) with a local Ollama LLM and hybrid BM25 + vector search.

## Prerequisites

- **Docker** 20+ with Compose plugin (`docker compose version`)
- **MCP client**: Claude Desktop **or** VSCode with GitHub Copilot

No Node.js required to just run the server — everything runs inside Docker.

## Quick Start

**1. Clone and configure**

```bash
git clone <repo-url>
cd void-protocol-rag
cp .env.example .env   # defaults work out of the box
```

**2. Start all services**

```bash
docker compose up -d
```

**First start takes 5–15 minutes** — Ollama downloads `phi3:mini` (~2.4 GB) and `nomic-embed-text` (~270 MB).
Subsequent starts are instant (models are cached in a Docker volume).

To follow logs while waiting: `docker compose logs -f`

**3. Verify services are healthy**

```bash
docker compose ps
```

All three services (`mcp-server`, `chromadb`, `ollama`) must show status `healthy` or `running`.

You can also check endpoints directly:

```bash
curl http://localhost:3000/health   # {"status":"ok"}
curl http://localhost:8000/api/v2/heartbeat  # {"nanosecond heartbeat":...}
curl http://localhost:11434/api/tags         # {"models":[...]}
```

## Connect MCP Client

### Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "void-protocol-rag": {
      "type": "sse",
      "url": "http://localhost:3000/sse"
    }
  }
}
```

Restart Claude Desktop. The four tools (`index_folder`, `ask_question`, `find_relevant_docs`, `index_status`) will appear in the tools panel.

### VSCode + GitHub Copilot

Add the contents of `mcp-config.json` to your VSCode `settings.json` under the key `"github.copilot.chat.mcp.servers"`.

## Verification

You can verify using your MCP client (Claude Desktop or Copilot Chat) or with **MCP Inspector**:

```bash
npx @modelcontextprotocol/inspector http://localhost:3000/sse
```

Run the steps below in order:

**Step 1 — Check status before indexing**

```
index_status()
```

Expected: `status: "empty", chunkCount: 0`

**Step 2 — Index the sample documents**

```
index_folder("./sample_docs")
```

Expected response: `filesIndexed: 14, chunksCreated: <number>`

**Step 3 — Check status after indexing**

```
index_status()
```

Expected: `status: "ready", chunkCount: <number>, lastIndexed: <timestamp>`

**Step 4 — Ask a question in Russian**

```
ask_question("сколько постоянных жителей на станции Новая Амора?")
```

Expected: an answer mentioning **47 000** — a fact that exists only in `sample_docs` and cannot be guessed from general knowledge.

**Step 5 — Find raw chunks**

```
find_relevant_docs("combat mechanics")
```

Expected: several chunks from `sample_docs/docs/combat-system.md` with relevance scores.

## Verification Facts

The following numbers exist only in `sample_docs` — if the RAG returns them, it is reading your documents, not hallucinating:

| Question | Expected answer |
|---|---|
| Постоянное население станции Новая Амора | **47 000** |
| Стартовый кредит нового игрока | **2 847** |
| Порт сервера игры | **51 847** |
| Базовая константа Void Resonance Factor (VRF) | **0.8847 Hz** |

## Development

```bash
npm install
npm run build          # compile TypeScript
npx tsc --noEmit       # type-check only
npx vitest run tests/unit   # unit tests (no Docker needed)
```

E2E tests require ChromaDB running locally:

```bash
docker compose up -d chromadb
npx vitest run tests/e2e
```

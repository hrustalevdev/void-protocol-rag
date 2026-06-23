# Void Protocol RAG — MCP Server

MCP server that turns a local folder of documents into a searchable knowledge base.
Uses Corrective RAG (LangGraph) with a local Ollama LLM and hybrid BM25+vector search.

## Quick Start

```bash
cp .env.example .env
docker compose up
```

First start downloads Ollama models (~3GB). Wait for all services to be healthy.

## Connect to VSCode Copilot

Copy `mcp-config.json` contents into your VSCode `settings.json` under `"github.copilot.chat.mcp.servers"`.

## Usage

```
index_folder("./sample_docs")        # index the demo documents
ask_question("что такое Void Protocol?")  # ask in Russian, get answer in Russian
find_relevant_docs("combat mechanics")    # raw chunk search
index_status()                            # check index state
```

## Verification Facts

The following facts exist only in the `sample_docs` knowledge base and cannot be guessed from general knowledge:

- New Amora Station's permanent resident population is exactly **47,000**
- Starting credits for new players is **2,847** (the cost of a one-way docking permit)
- The server port is **51847** (chosen deliberately; contains 1847, the session timeout reference)
- The Void Resonance Factor (VRF) base constant is **0.8847 Hz** (appears in combat formulas, world lore, NPC dialogue, and config)

Ask `ask_question("сколько постоянных жителей на станции Новая Амора?")` to verify RAG is reading your documents, not using general knowledge.

## Development

```bash
npm install
npm run build
npx vitest run
```

E2E tests require ChromaDB: `docker compose up -d chromadb`

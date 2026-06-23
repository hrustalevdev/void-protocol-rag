# Architecture

## Overview

MCP server built with TypeScript and Express. Exposes 4 tools over SSE transport.
Implements Corrective RAG pattern using LangGraph for orchestration.

## Components

```mermaid
graph TD
    Client["MCP Client (IDE)"]
    Server["MCP Server (Express)"]
    IF["index_folder"]
    AQ["ask_question"]
    FRD["find_relevant_docs"]
    IS["index_status"]
    Indexer["Indexer"]
    RAG["LangGraph RAG Graph\nrewrite → retrieve → grade → generate"]
    ChromaDB["ChromaDB\n(vector store)"]
    BM25["BM25\n(in-memory)"]
    Ollama["Ollama\n(LLM + embeddings)"]

    Client -->|"SSE port 3000"| Server
    Server --> IF
    Server --> AQ
    Server --> FRD
    Server --> IS
    IF --> Indexer
    AQ --> RAG
    FRD --> ChromaDB
    FRD --> BM25
    IS --> Indexer
    Indexer --> ChromaDB
    Indexer --> BM25
    RAG --> ChromaDB
    RAG --> BM25
    RAG --> Ollama
    Indexer --> Ollama
```

## Corrective RAG Flow

```mermaid
flowchart TD
    Start([RU Query]) --> RW[rewriteQuery\nRU → EN]
    RW --> RT[retrieve\nBM25 + vector → RRF]
    RT --> GR[gradeChunks\nLLM: relevant? yes/no]
    GR --> COND{≥2 relevant\nOR retries ≥ 2?}
    COND -->|Yes| GEN[generate\nanswer in RU]
    GEN --> End([Return answer + sources])
    COND -->|No| BQ[broadenQuery\nexpand with synonyms]
    BQ --> RT
```

## Hybrid Search

BM25 (keyword) + ChromaDB vector search → Reciprocal Rank Fusion (k=60).
BM25 index is held in memory and rebuilt from ChromaDB on server startup.

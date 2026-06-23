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
    RAG["LangGraph RAG Graph<br/>rewrite → retrieve → grade → generate"]
    ChromaDB["ChromaDB<br/>(vector store)"]
    BM25["BM25<br/>(in-memory)"]
    Ollama["Ollama<br/>(LLM + embeddings)"]

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
    Start([RU Query]) --> RW[rewriteQuery<br/>RU → EN]
    RW --> RT[retrieve<br/>BM25 + vector → RRF]
    RT --> GR[gradeChunks<br/>LLM: relevant? yes/no]
    GR --> COND{≥2 relevant<br/>OR retries ≥ 2?}
    COND -->|Yes| GEN[generate<br/>answer in RU]
    GEN --> End([Return answer + sources])
    COND -->|No| BQ[broadenQuery<br/>expand with synonyms]
    BQ --> RT
```

## Hybrid Search

```mermaid
flowchart LR
    Query["Search Query"]
    BM25["BM25<br/>(keyword match)"]
    Vector["ChromaDB<br/>(vector search)"]
    RRF["Reciprocal Rank Fusion<br/>k=60"]
    Result["Top-K Chunks"]

    Query --> BM25
    Query --> Vector
    BM25 -->|"ranked list"| RRF
    Vector -->|"ranked list"| RRF
    RRF --> Result
```

BM25 index is held in memory and rebuilt from ChromaDB on server startup.

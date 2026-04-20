# Architecture Overview

## System Design

```
User Question
    ↓
REST API (FastAPI)
    ↓
RAG Pipeline
├── Retrieve (search vector DB)
├── Rerank (score relevance)
├── Generate (LLM + context)
└── Return answer + sources
    ↓
Vector Database (ChromaDB)
    ↓
Embeddings (sentence-transformers)
```

## Data Flow

1. **Ingestion Phase**
   - Parse markdown files
   - Split into semantic chunks
   - Generate embeddings (384-dim)
   - Store in ChromaDB with metadata

2. **Query Phase**
   - Encode user question
   - Find similar chunks (cosine similarity)
   - Rerank by relevance
   - Prompt LLM with top chunks
   - Extract answer + citations

3. **Agent Phase** (complex questions)
   - Break down question into steps
   - Execute tools (search, recommend)
   - Aggregate results
   - Synthesize final answer

4. **Evaluation Phase**
   - Run 30 benchmark questions
   - Measure 4 metrics:
     - Faithfulness (no hallucination)
     - Answer Relevance (addresses Q)
     - Context Precision (relevant docs)
     - Context Recall (complete retrieval)

5. **Deployment Phase**
   - Docker container
   - Health checks
   - Rate limiting
   - Production logging

## Key Components

### ingest_handbook.py
- Clones/updates handbook
- Parses markdown hierarchically
- Generates embeddings locally
- Indexes in ChromaDB

### query_engine.py
- KnowledgeBase class (wraps ChromaDB)
- SimpleReranker (scores chunks)
- AnswerGenerator (LLM synthesis)
- RAGPipeline (orchestration)

### agent.py
- ToolExecutor (runs tools)
- AgentMemory (tracks reasoning)
- Agent (decision loop)
- MultiAgentOrchestrator (routing)

### evaluate.py
- RAGAS metrics implementation
- 30-question benchmark
- Regression testing
- Results persistence

### api_server.py
- FastAPI endpoints
- Pydantic validation
- Rate limiting
- Health checks
- Interactive docs

## Technologies

| Layer | Technology |
|-------|-----------|
| API | FastAPI + Uvicorn |
| Validation | Pydantic |
| LLM | OpenAI GPT-3.5 |
| Embeddings | sentence-transformers |
| Vector DB | ChromaDB |
| Deployment | Docker |
| CI/CD | GitHub Actions |

## Scaling

Current: 1 container, handles ~100 requests/hour
To scale: Multiple containers + load balancer + separate DB service

# OpsBot

A production AI system I built to demonstrate full-stack LLM engineering - from data ingestion to deployment.

## The Problem

Most companies have tons of internal docs (policies, handbooks, SOPs) scattered everywhere. When you need an answer, you waste time digging through them. OpsBot solves this by understanding questions and finding answers from your docs automatically.

## What I Built

7 phases across the full AI stack:

1. **Ingestion** - Parse docs, generate embeddings, store in vector DB
2. **Retrieval** - Find relevant docs using semantic search
3. **Generation** - Use LLM to synthesize answers from docs
4. **Agents** - Handle complex questions with multi-step reasoning
5. **Evaluation** - Actually measure if it works (0.81 quality score)
6. **API** - Expose as REST endpoints with validation & rate limiting
7. **Deployment** - Docker + CI/CD so it actually ships

## Quick Demo

```bash
pip install -r requirements.txt
python scripts/ingest_handbook.py
python scripts/api_server.py
```

Then visit http://localhost:8000/docs for interactive API docs.

Or run the RAG engine directly:
```bash
python scripts/query_engine.py
```

## Interactive Tests

Try each component:

```bash
# See the RAG system in action
python scripts/query_engine.py

# Test agent reasoning on complex questions  
python scripts/agent.py

# Run evaluation benchmark (30 questions)
python scripts/evaluate.py

# Start the API server
python scripts/api_server.py
```

## Docker

```bash
docker-compose up
```

## What's in Here

- **scripts/** - All 7 phases, ready to run
- **data/handbook/** - Sample company handbook (IT, HR, Engineering)
- **Dockerfile** - Multi-stage build, optimized for production
- **docker-compose.yml** - Local dev setup
- **requirements.txt** - All dependencies
- **docs/** - Architecture guide and detailed breakdowns

## Tech Stack

- **Python** - Core system
- **ChromaDB** - Vector database
- **sentence-transformers** - Embeddings
- **FastAPI** - REST API
- **OpenAI** - LLM (with mock fallback)
- **Docker** - Deployment

## Real Numbers

- 4,305 lines of code
- 30-question evaluation benchmark
- 0.81 quality score (faithfulness, relevance, precision, recall)
- Handles easy/medium/hard questions

## Why This Matters

Most AI projects stop at "it works." I went further:
- **Evaluation** - Measured quality with real metrics
- **Deployment** - Actually containerized and automated
- **Error handling** - Graceful fallbacks, real logging
- **Production patterns** - Rate limiting, health checks, tracing

## Learn More

See `docs/ARCHITECTURE.md` for system design and `docs/PROJECT_SUMMARY.md` for interview talking points.

---

Built to demonstrate skills for AI engineer roles at companies like Stripe, Anthropic, OpenAI.

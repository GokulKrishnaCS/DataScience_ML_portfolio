# Setup & Running OpsBot

## Prerequisites

- Python 3.10+
- pip
- (Optional) Docker & Docker Compose

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/opsbot.git
cd opsbot
pip install -r requirements.txt
```

## Running Each Phase

### Phase 1: Ingest Documents
```bash
python ingest_handbook.py
```
Creates vector database from handbook files.

### Phase 2: Interactive RAG
```bash
python query_engine.py
```
Try asking: "What is the password policy?"

### Phase 3: Agent Reasoning
```bash
python agent.py
```
Try: "I need to understand password policy and create a secure password"

### Phase 4: MCP Server (Demo)
```bash
python mcp_server.py --demo
```

### Phase 5: Evaluate System
```bash
python evaluate.py
```
Runs 30-question benchmark, reports metrics.

### Phase 6: REST API
```bash
python api_server.py
```
Visit http://localhost:8000/docs for interactive API docs.

Test:
```bash
curl -X POST http://localhost:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "password policy"}'
```

## With Docker

```bash
docker-compose up
curl http://localhost:8000/api/v1/health
```

## Troubleshooting

**"ModuleNotFoundError: No module named 'chromadb'"**
```bash
pip install -r requirements.txt
```

**"CUDA not available"** (if you see GPU warnings)
This is fine - embeddings run on CPU.

**API returns 503 "RAG pipeline not available"**
Run Phase 1 first:
```bash
python ingest_handbook.py
```

**Docker build fails**
Make sure you're in the right directory:
```bash
pwd  # should end with /opsbot
docker-compose up
```

## Configuration

Set OpenAI API key (optional, system has fallback):
```bash
export OPENAI_API_KEY="sk-..."
```

## Data

Sample handbook files are in `data/handbook/` - modify or replace with your own docs.

## Files

- `ingest_handbook.py` - Ingestion pipeline
- `query_engine.py` - RAG implementation  
- `agent.py` - Multi-step reasoning
- `mcp_server.py` - MCP protocol server
- `evaluate.py` - Evaluation framework
- `api_server.py` - REST API
- `data/handbook/` - Sample docs
- `requirements.txt` - Dependencies
- `Dockerfile` - Production container
- `docker-compose.yml` - Local setup

See `docs/ARCHITECTURE.md` for design details.

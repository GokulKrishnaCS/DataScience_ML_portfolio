# OpsBot

Production-grade AI knowledge assistant that reduces time spent searching internal documents by retrieving and generating answers from company knowledge bases.

---

## Business Impact

- Eliminates manual document search across policies, SOPs, and handbooks  
- Demonstrates end-to-end LLM system design beyond prototypes  
- Built with production patterns: evaluation, APIs, deployment, reliability  

---

## What It Does

- Converts internal docs → embeddings → vector store  
- Retrieves relevant context using semantic search  
- Generates grounded answers using LLMs  
- Handles complex queries via agent workflows  
- Measures quality with structured evaluation  

---

## System Coverage (7 Phases)

1. Ingestion — parsing, chunking, embeddings  
2. Retrieval — semantic search over vector DB  
3. Generation — LLM-based grounded responses  
4. Agents — multi-step reasoning workflows  
5. Evaluation — benchmark-driven scoring  
6. API — FastAPI with validation & rate limiting  
7. Deployment — Dockerized, CI/CD-ready  

---

## Key Results

- 30-question benchmark  
- 0.81 overall quality score  
- Evaluated on faithfulness, relevance, precision, recall  
- Supports easy → complex queries  

---

## Tech Stack

- Python  
- ChromaDB  
- sentence-transformers  
- FastAPI  
- OpenAI (with fallback)  
- Docker  

---

## Quick Start

```bash
pip install -r requirements.txt
python scripts/ingest_handbook.py
python scripts/api_server.py

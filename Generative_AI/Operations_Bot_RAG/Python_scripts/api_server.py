"""
================================================================================
PHASE 6: REST API BACKEND - FASTAPI & PRODUCTION INFRASTRUCTURE
================================================================================
OpsBot - Enterprise Knowledge Copilot

REST API exposes OpsBot as production-grade service:
1. Search endpoint - Query knowledge base with RAG
2. Agent endpoint - Multi-step reasoning and task execution
3. Evaluation endpoint - Run benchmarks and get metrics
4. Admin endpoints - System health, metadata, configuration

Production features:
- Request/response validation (Pydantic models)
- Error handling and standardized responses
- Rate limiting (by IP/user)
- Request logging and audit trail
- Structured logging (JSON format)
- CORS support for cross-origin requests
- OpenAPI documentation (Swagger UI)

SKILL SIGNALS THIS DEMONSTRATES:
- Backend engineering: FastAPI, REST design
- Production patterns: logging, error handling, validation
- API design: clean contracts, versioning
- Observability: request tracing, performance metrics
- Security: rate limiting, input validation

Endpoints:
- POST /api/v1/search     - Search knowledge base
- POST /api/v1/agent/ask  - Run agentic task
- GET  /api/v1/health     - System status
- GET  /api/v1/topics     - List available topics
- POST /api/v1/evaluate   - Run evaluation benchmark

================================================================================
"""

import json
import logging
import time
from datetime import datetime
from typing import Optional, List, Dict, Any
from pathlib import Path
import uuid

from fastapi import (
    FastAPI, HTTPException, Request, Response,
    BackgroundTasks
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

from query_engine import RAGPipeline
from agent import Agent
from evaluate import EvaluationEngine, BENCHMARK_DATASET

# ============================================================================
# LOGGING SETUP
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# REQUEST/RESPONSE MODELS (Pydantic)
# ============================================================================

class SearchRequest(BaseModel):
    """Search knowledge base request"""
    query: str = Field(..., min_length=1, max_length=500,
                       description="Search query")
    top_k: int = Field(3, ge=1, le=10,
                       description="Number of results to return")


class SearchResponse(BaseModel):
    """Search knowledge base response"""
    request_id: str
    query: str
    answer: str
    sources: List[str]
    confidence: float
    retrieved_chunks: int
    execution_time_ms: float


class AgentRequest(BaseModel):
    """Agent multi-step reasoning request"""
    question: str = Field(..., min_length=1, max_length=1000,
                          description="Complex question or task")
    max_steps: int = Field(5, ge=1, le=10,
                           description="Maximum reasoning steps")


class AgentResponse(BaseModel):
    """Agent response"""
    request_id: str
    question: str
    answer: str
    recommendation: str
    steps_taken: List[str]
    summary: str
    execution_time_ms: float


class HealthResponse(BaseModel):
    """System health status"""
    status: str
    version: str
    timestamp: str
    rag_ready: bool
    agent_ready: bool
    uptime_seconds: float


class TopicsResponse(BaseModel):
    """Available topics in knowledge base"""
    topics: List[str]
    total_chunks: int
    categories: Dict[str, int]


class EvaluateRequest(BaseModel):
    """Run evaluation benchmark"""
    subset: str = Field("all", regex="^(all|easy|medium|hard)$",
                        description="Question difficulty subset")


class EvaluateResponse(BaseModel):
    """Evaluation results"""
    request_id: str
    total_questions: int
    overall_score: float
    execution_time_ms: float
    metrics: Dict[str, Dict[str, float]]


class ErrorResponse(BaseModel):
    """Standardized error response"""
    request_id: str
    error: str
    status_code: int
    timestamp: str


# ============================================================================
# RATE LIMITING
# ============================================================================

class RateLimiter:
    """Simple in-memory rate limiter"""

    def __init__(self, requests_per_minute: int = 30):
        self.requests_per_minute = requests_per_minute
        self.requests = {}  # IP -> [(timestamp, request), ...]

    def is_allowed(self, client_ip: str) -> bool:
        """Check if request is allowed"""
        now = time.time()
        cutoff = now - 60  # Last 60 seconds

        # Clean old requests
        if client_ip in self.requests:
            self.requests[client_ip] = [
                ts for ts in self.requests[client_ip]
                if ts > cutoff
            ]
        else:
            self.requests[client_ip] = []

        # Check limit
        if len(self.requests[client_ip]) >= self.requests_per_minute:
            return False

        # Add this request
        self.requests[client_ip].append(now)
        return True


# ============================================================================
# APPLICATION STATE
# ============================================================================

class AppState:
    """Shared application state"""
    def __init__(self):
        self.rag: Optional[RAGPipeline] = None
        self.agent: Optional[Agent] = None
        self.start_time = time.time()
        self.request_count = 0
        self.rate_limiter = RateLimiter(requests_per_minute=30)
        self.request_log: List[Dict] = []  # Audit trail


app_state = AppState()


# ============================================================================
# LIFESPAN MANAGEMENT
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup and shutdown management
    """
    # Startup
    print("\n" + "="*70)
    print("PHASE 6: REST API SERVER - STARTING")
    print("="*70)

    logger.info("Initializing RAG pipeline...")
    try:
        app_state.rag = RAGPipeline()
        logger.info(f"RAG ready: {app_state.rag.ready}")
    except Exception as e:
        logger.error(f"Failed to initialize RAG: {e}")
        app_state.rag = None

    if app_state.rag and app_state.rag.ready:
        try:
            app_state.agent = Agent(app_state.rag)
            logger.info("Agent initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize agent: {e}")
            app_state.agent = None

    logger.info("REST API ready at http://localhost:8000")
    logger.info("Documentation at http://localhost:8000/docs")

    yield

    # Shutdown
    logger.info("Shutting down REST API")


# ============================================================================
# CREATE APP
# ============================================================================

app = FastAPI(
    title="OpsBot REST API",
    description="Enterprise Knowledge Copilot - Production API",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# MIDDLEWARE
# ============================================================================

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Rate limiting middleware"""
    client_ip = request.client.host if request.client else "unknown"

    if not app_state.rate_limiter.is_allowed(client_ip):
        return JSONResponse(
            status_code=429,
            content={
                "error": "Rate limit exceeded",
                "status_code": 429,
                "timestamp": datetime.utcnow().isoformat()
            }
        )

    return await call_next(request)


@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    """Request logging and tracing"""
    request_id = str(uuid.uuid4())
    start_time = time.time()

    # Add request ID to state
    request.state.request_id = request_id

    response = await call_next(request)
    duration_ms = (time.time() - start_time) * 1000

    # Log request
    log_entry = {
        "request_id": request_id,
        "method": request.method,
        "path": request.url.path,
        "status_code": response.status_code,
        "duration_ms": duration_ms,
        "timestamp": datetime.utcnow().isoformat(),
        "client_ip": request.client.host if request.client else "unknown"
    }

    app_state.request_log.append(log_entry)
    logger.info(f"{request.method} {request.url.path} - {response.status_code} ({duration_ms:.1f}ms)")

    return response


# ============================================================================
# EXCEPTION HANDLERS
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "request_id": request_id,
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.utcnow().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions"""
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    logger.error(f"Unhandled exception: {exc}", exc_info=True)

    return JSONResponse(
        status_code=500,
        content={
            "request_id": request_id,
            "error": "Internal server error",
            "status_code": 500,
            "timestamp": datetime.utcnow().isoformat()
        }
    )


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/api/v1/health", response_model=HealthResponse)
async def health_check(request: Request) -> HealthResponse:
    """System health and status"""
    uptime = time.time() - app_state.start_time

    return HealthResponse(
        status="healthy" if app_state.rag and app_state.rag.ready else "degraded",
        version="1.0.0",
        timestamp=datetime.utcnow().isoformat(),
        rag_ready=bool(app_state.rag and app_state.rag.ready),
        agent_ready=bool(app_state.agent is not None),
        uptime_seconds=uptime
    )


@app.post("/api/v1/search", response_model=SearchResponse)
async def search_knowledge_base(
    request: Request,
    search_req: SearchRequest
) -> SearchResponse:
    """Search the knowledge base using RAG"""
    request_id = request.state.request_id
    start_time = time.time()

    if not app_state.rag or not app_state.rag.ready:
        raise HTTPException(
            status_code=503,
            detail="RAG pipeline not available"
        )

    try:
        # Query RAG pipeline
        result = app_state.rag.query(search_req.query)

        duration_ms = (time.time() - start_time) * 1000

        return SearchResponse(
            request_id=request_id,
            query=search_req.query,
            answer=result.answer,
            sources=result.sources[:search_req.top_k],
            confidence=result.confidence,
            retrieved_chunks=result.retrieved_chunks,
            execution_time_ms=duration_ms
        )

    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )


@app.post("/api/v1/agent/ask", response_model=AgentResponse)
async def ask_agent(
    request: Request,
    agent_req: AgentRequest
) -> AgentResponse:
    """Run agent for multi-step reasoning"""
    request_id = request.state.request_id
    start_time = time.time()

    if not app_state.agent:
        raise HTTPException(
            status_code=503,
            detail="Agent not available"
        )

    try:
        # Run agent
        result = app_state.agent.run(agent_req.question)

        duration_ms = (time.time() - start_time) * 1000

        return AgentResponse(
            request_id=request_id,
            question=agent_req.question,
            answer=result["answer"],
            recommendation=result["recommendation"],
            steps_taken=result["steps_taken"],
            summary=result["summary"],
            execution_time_ms=duration_ms
        )

    except Exception as e:
        logger.error(f"Agent failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Agent failed: {str(e)}"
        )


@app.get("/api/v1/topics", response_model=TopicsResponse)
async def list_topics(request: Request) -> TopicsResponse:
    """List available topics in knowledge base"""
    if not app_state.rag or not app_state.rag.ready:
        raise HTTPException(
            status_code=503,
            detail="RAG pipeline not available"
        )

    try:
        info = app_state.rag.kb.get_collection_info()

        # Extract topics
        topics = set()
        for file_info in info.get("metadata", {}).get("files", []):
            path = file_info.get("relative_path", "")
            topic = path.split("\\")[0] if "\\" in path else path.split("/")[0]
            if topic:
                topics.add(topic)

        # Count by category
        categories = {}
        for file_info in info.get("metadata", {}).get("files", []):
            path = file_info.get("relative_path", "")
            topic = path.split("\\")[0] if "\\" in path else path.split("/")[0]
            categories[topic] = categories.get(topic, 0) + 1

        return TopicsResponse(
            topics=sorted(list(topics)),
            total_chunks=info.get("metadata", {}).get("total_chunks", 0),
            categories=categories
        )

    except Exception as e:
        logger.error(f"Failed to list topics: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list topics: {str(e)}"
        )


@app.post("/api/v1/evaluate", response_model=EvaluateResponse)
async def run_evaluation(
    request: Request,
    eval_req: EvaluateRequest,
    background_tasks: BackgroundTasks
) -> EvaluateResponse:
    """Run evaluation benchmark"""
    request_id = request.state.request_id
    start_time = time.time()

    if not app_state.rag or not app_state.rag.ready:
        raise HTTPException(
            status_code=503,
            detail="RAG pipeline not available"
        )

    try:
        # Run evaluation
        evaluator = EvaluationEngine(app_state.rag)
        stats = evaluator.run_benchmark(subset=eval_req.subset)

        duration_ms = (time.time() - start_time) * 1000

        return EvaluateResponse(
            request_id=request_id,
            total_questions=stats.get("total_questions", 0),
            overall_score=stats.get("overall_score", 0.0),
            execution_time_ms=duration_ms,
            metrics=stats.get("metrics", {})
        )

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Evaluation failed: {str(e)}"
        )


# ============================================================================
# ADMIN ENDPOINTS
# ============================================================================

@app.get("/api/v1/admin/stats")
async def admin_stats(request: Request) -> Dict[str, Any]:
    """Admin statistics (in production, add authentication)"""
    return {
        "total_requests": len(app_state.request_log),
        "uptime_seconds": time.time() - app_state.start_time,
        "request_log": app_state.request_log[-10:]  # Last 10 requests
    }


@app.get("/")
async def root():
    """Root endpoint - documentation redirect"""
    return {
        "message": "OpsBot REST API v1.0",
        "documentation": "http://localhost:8000/docs",
        "openapi": "http://localhost:8000/openapi.json"
    }


# ============================================================================
# MAIN - RUN SERVER
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    print("\n" + "="*70)
    print("PHASE 6: REST API SERVER")
    print("="*70)
    print("\nStarting FastAPI server...")
    print("API Documentation: http://localhost:8000/docs")
    print("ReDoc: http://localhost:8000/redoc")
    print("\nExample requests:")
    print("  curl -X POST http://localhost:8000/api/v1/search \\")
    print("    -H 'Content-Type: application/json' \\")
    print("    -d '{\"query\": \"What is the password policy?\"}'")
    print("\n  curl http://localhost:8000/api/v1/health")
    print("\n" + "="*70 + "\n")

    # Run uvicorn server
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )

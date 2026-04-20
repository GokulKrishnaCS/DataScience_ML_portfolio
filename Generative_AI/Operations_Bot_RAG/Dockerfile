# ============================================================================
# DOCKERFILE - OpsBot Production Container
# ============================================================================
# Multi-stage build for optimal image size
# Stage 1: Build (dependencies compilation)
# Stage 2: Runtime (lean production image)
#
# Build: docker build -t opsbot:latest .
# Run:   docker run -p 8000:8000 opsbot:latest
# ============================================================================

# ============================================================================
# STAGE 1: BUILDER
# ============================================================================
FROM python:3.11-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Create wheels for all dependencies (faster installation in runtime stage)
RUN pip install --no-cache-dir --user --no-warn-script-location \
    --upgrade pip setuptools wheel && \
    pip wheel --no-cache-dir --no-deps --wheel-dir /build/wheels -r requirements.txt

# ============================================================================
# STAGE 2: RUNTIME
# ============================================================================
FROM python:3.11-slim

WORKDIR /app

# Create non-root user for security
RUN useradd -m -u 1000 opsbot

# Install only runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy wheels from builder
COPY --from=builder /build/wheels /wheels
COPY --from=builder /build/requirements.txt .

# Install Python dependencies from wheels
RUN pip install --no-cache-dir --no-index --find-links /wheels -r requirements.txt && \
    rm -rf /wheels

# Copy application
COPY --chown=opsbot:opsbot opsbot ./opsbot

# Switch to non-root user
USER opsbot

# Expose API port
EXPOSE 8000

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV OPENAI_API_KEY=""

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Run API server by default
CMD ["python", "-m", "uvicorn", "opsbot.scripts.api_server:app", \
     "--host", "0.0.0.0", \
     "--port", "8000"]

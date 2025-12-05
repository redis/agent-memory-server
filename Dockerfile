# ============================================
# BUILDER BASE - Build tools for compilation
# ============================================
FROM python:3.12-slim-bookworm AS builder-base

WORKDIR /app

# Copy uv binary from official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

# Install build tools (only needed for compilation)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# ============================================
# BUILDER STANDARD - Compile standard deps
# ============================================
FROM builder-base AS builder-standard

# Create virtual environment explicitly
RUN uv venv .venv

# Copy dependency files first for better layer caching
COPY pyproject.toml uv.lock ./
COPY agent-memory-client ./agent-memory-client

# Install dependencies into the venv (without the project)
RUN --mount=type=cache,target=/root/.cache/uv \
    VIRTUAL_ENV=/app/.venv uv sync --frozen --no-install-project --no-dev

# Copy source code
COPY . /app

# Install the project itself
RUN --mount=type=cache,target=/root/.cache/uv \
    . .venv/bin/activate && \
    uv pip install --no-deps .

# ============================================
# BUILDER AWS - Compile AWS deps
# ============================================
FROM builder-base AS builder-aws

# Create virtual environment explicitly
RUN uv venv .venv

# Copy dependency files first for better layer caching
COPY pyproject.toml uv.lock ./
COPY agent-memory-client ./agent-memory-client

# Install dependencies into the venv (without the project)
RUN --mount=type=cache,target=/root/.cache/uv \
    VIRTUAL_ENV=/app/.venv uv sync --frozen --no-install-project --no-dev --extra aws

# Copy source code
COPY . /app

# Install the project itself
RUN --mount=type=cache,target=/root/.cache/uv \
    . .venv/bin/activate && \
    uv pip install --no-deps .

# ============================================
# RUNTIME BASE - Slim image without build tools
# ============================================
FROM python:3.12-slim-bookworm AS runtime-base

# OCI labels for Docker Hub and container registries
LABEL org.opencontainers.image.title="Redis Agent Memory Server"
LABEL org.opencontainers.image.description="A memory layer for AI agents using Redis as the vector database. Provides REST API and MCP server interfaces with semantic search, topic extraction, and conversation summarization."
LABEL org.opencontainers.image.url="https://github.com/redis/agent-memory-server"
LABEL org.opencontainers.image.source="https://github.com/redis/agent-memory-server"
LABEL org.opencontainers.image.documentation="https://redis.github.io/agent-memory-server/"
LABEL org.opencontainers.image.vendor="Redis"
LABEL org.opencontainers.image.licenses="Apache-2.0"

WORKDIR /app

# Install only runtime dependencies (curl for healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r agentmemory && useradd -r -g agentmemory agentmemory

# ============================================
# STANDARD VARIANT - OpenAI/Anthropic only
# ============================================
FROM runtime-base AS standard

# Copy the virtual environment and app from builder
COPY --chown=agentmemory:agentmemory --from=builder-standard /app /app

ENV PATH="/app/.venv/bin:$PATH"

# Switch to non-root user
USER agentmemory

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/v1/health || exit 1

# Enable authentication by default.
# You may override with DISABLE_AUTH=true in development.
ENV DISABLE_AUTH=false

# Default to development mode (no separate worker needed).
# For production, override the command to remove --no-worker and run a separate task-worker container.
# Examples:
#   Development: docker run -p 8000:8000 redislabs/agent-memory-server
#   Production API: docker run -p 8000:8000 redislabs/agent-memory-server agent-memory api --host 0.0.0.0 --port 8000
#   Production Worker: docker run redislabs/agent-memory-server agent-memory task-worker --concurrency 10
CMD ["agent-memory", "api", "--host", "0.0.0.0", "--port", "8000", "--no-worker"]

# ============================================
# AWS VARIANT - Includes AWS Bedrock support
# ============================================
FROM runtime-base AS aws

# Copy the virtual environment and app from builder
COPY --chown=agentmemory:agentmemory --from=builder-aws /app /app

ENV PATH="/app/.venv/bin:$PATH"

# Switch to non-root user
USER agentmemory

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/v1/health || exit 1

# Enable authentication by default.
# You may override with DISABLE_AUTH=true in development.
ENV DISABLE_AUTH=false

# Default to development mode (no separate worker needed).
# For production, override the command to remove --no-worker and run a separate task-worker container.
# Examples:
#   Development: docker run -p 8000:8000 redislabs/agent-memory-server:aws
#   Production API: docker run -p 8000:8000 redislabs/agent-memory-server:aws agent-memory api --host 0.0.0.0 --port 8000
#   Production Worker: docker run redislabs/agent-memory-server:aws agent-memory task-worker --concurrency 10
CMD ["agent-memory", "api", "--host", "0.0.0.0", "--port", "8000", "--no-worker"]

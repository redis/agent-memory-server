# ============================================
# BASE STAGE - Common setup for all variants
# ============================================
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS base

# OCI labels for Docker Hub and container registries
LABEL org.opencontainers.image.title="Redis Agent Memory Server"
LABEL org.opencontainers.image.description="A memory layer for AI agents using Redis as the vector database. Provides REST API and MCP server interfaces with semantic search, topic extraction, and conversation summarization."
LABEL org.opencontainers.image.url="https://github.com/redis/agent-memory-server"
LABEL org.opencontainers.image.source="https://github.com/redis/agent-memory-server"
LABEL org.opencontainers.image.documentation="https://redis.github.io/agent-memory-server/"
LABEL org.opencontainers.image.vendor="Redis"
LABEL org.opencontainers.image.licenses="Apache-2.0"

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

# Install system dependencies including build tools
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r agentmemory && useradd -r -g agentmemory agentmemory

# ============================================
# STANDARD VARIANT - OpenAI/Anthropic only
# ============================================
FROM base AS standard

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=agent-memory-client,target=agent-memory-client \
    uv sync --frozen --no-install-project --no-dev

ADD . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

RUN chown -R agentmemory:agentmemory /app

ENV PATH="/app/.venv/bin:$PATH"

# Switch to non-root user
USER agentmemory

ENTRYPOINT []

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/v1/health || exit 1

# Disable auth by default for easier local development.
# Override with DISABLE_AUTH=false in production.
ENV DISABLE_AUTH=true

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
FROM base AS aws

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=agent-memory-client,target=agent-memory-client \
    uv sync --frozen --no-install-project --no-dev --extra aws

ADD . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra aws

RUN chown -R agentmemory:agentmemory /app

ENV PATH="/app/.venv/bin:$PATH"

# Switch to non-root user
USER agentmemory

ENTRYPOINT []

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/v1/health || exit 1

# Disable auth by default for easier local development.
# Override with DISABLE_AUTH=false in production.
ENV DISABLE_AUTH=true

# Default to development mode (no separate worker needed).
# For production, override the command to remove --no-worker and run a separate task-worker container.
# Examples:
#   Development: docker run -p 8000:8000 redislabs/agent-memory-server:aws
#   Production API: docker run -p 8000:8000 redislabs/agent-memory-server:aws agent-memory api --host 0.0.0.0 --port 8000
#   Production Worker: docker run redislabs/agent-memory-server:aws agent-memory task-worker --concurrency 10
CMD ["agent-memory", "api", "--host", "0.0.0.0", "--port", "8000", "--no-worker"]

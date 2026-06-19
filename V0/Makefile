.PHONY: help setup sync sync-dev lint format test test-api test-unit test-integration test-cov pre-commit verify clean server mcp mcp-sse worker rebuild-index migrate

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Setup and dependencies
setup:  ## Initial setup: create virtualenv and install dependencies
	pip install uv
	uv venv
	$(MAKE) sync
	uv run pre-commit install

sync:  ## Sync dependencies from lock file
	uv sync --all-extras

sync-dev:  ## Sync development dependencies only
	uv sync --only-dev

# Code quality
lint:  ## Run linting checks (ruff)
	uv run ruff check .

format:  ## Format code (ruff)
	uv run ruff format .
	uv run ruff check --fix .

pre-commit:  ## Run all pre-commit hooks
	uv run pre-commit run --all-files

verify:  ## Run the full local verification flow used by CI (requires OPENAI_API_KEY for API tests)
	$(MAKE) pre-commit
	$(MAKE) test-api

# Testing
test:  ## Run tests (matches the general CI test job; excludes API-key-dependent tests)
	uv run pytest

test-api:  ## Run all tests including API tests (matches the CI service-tests job; requires OPENAI_API_KEY)
	uv run pytest --run-api-tests

test-unit:  ## Run only unit tests
	uv run pytest tests/unit/

test-integration:  ## Run only integration tests
	uv run pytest tests/integration/

test-cov:  ## Run tests with coverage report
	uv run pytest --cov

# Running services
server:  ## Start the REST API server
	uv run agent-memory api

mcp:  ## Start the MCP server (stdio mode)
	uv run agent-memory mcp

mcp-sse:  ## Start the MCP server (SSE mode on port 9000)
	uv run agent-memory mcp --mode sse --port 9000

worker:  ## Start the background task worker
	uv run agent-memory task-worker

# Database operations
rebuild-index:  ## Rebuild Redis search index
	uv run agent-memory rebuild-index

migrate:  ## Run memory migrations
	uv run agent-memory migrate-memories

# Cleanup
clean:  ## Clean up generated files and caches
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .coverage htmlcov/ 2>/dev/null || true

#!/bin/bash
set -e

# Ensure uv is in PATH
export PATH="$HOME/.local/bin:$PATH"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Agent Memory Workbench Startup${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Create logs directory
mkdir -p /tmp/ams-logs

# Load environment variables from workbench .env if it exists
if [ -f "/workspace/workbench/.env" ]; then
    echo -e "${YELLOW}Loading environment from /workspace/workbench/.env...${NC}"
    set -a  # automatically export all variables
    source /workspace/workbench/.env
    set +a
fi

# Also check for root .env file
if [ -f "/workspace/.env" ]; then
    echo -e "${YELLOW}Loading environment from /workspace/.env...${NC}"
    set -a
    source /workspace/.env
    set +a
fi

# Unset ANTHROPIC_API_KEY so Claude CLI can use its normal OAuth login flow
# The Memory Server will load its own keys from .env files via python-dotenv
unset ANTHROPIC_API_KEY

# Wait for Redis
echo -e "${YELLOW}Waiting for Redis...${NC}"
until redis-cli -h redis ping > /dev/null 2>&1; do
    sleep 1
done
echo -e "${GREEN}Redis is ready!${NC}"

# Start Memory Server
echo -e "${YELLOW}Starting Memory Server...${NC}"
cd /workspace
if [ -f "pyproject.toml" ]; then
    nohup uv run agent-memory api --host 0.0.0.0 --port 8000 --task-backend=asyncio > /tmp/ams-logs/memory-server.log 2>&1 &
    echo $! > /tmp/ams-logs/memory-server.pid
else
    echo -e "${RED}Error: pyproject.toml not found in /workspace${NC}"
    exit 1
fi

# Wait for Memory Server
echo -e "${YELLOW}Waiting for Memory Server to be ready...${NC}"
max_attempts=60
attempt=0
until curl -s http://localhost:8000/v1/health > /dev/null 2>&1; do
    attempt=$((attempt + 1))
    if [ $attempt -ge $max_attempts ]; then
        echo -e "${RED}Memory Server failed to start. Check logs at /tmp/ams-logs/memory-server.log${NC}"
        cat /tmp/ams-logs/memory-server.log
        exit 1
    fi
    sleep 1
done
echo -e "${GREEN}Memory Server is ready!${NC}"

# Start MCP Server (SSE mode for workbench browser client)
echo -e "${YELLOW}Starting MCP Server (SSE mode on port 9000)...${NC}"
cd /workspace
nohup uv run agent-memory mcp --mode sse --port 9000 --task-backend=asyncio > /tmp/ams-logs/mcp-server.log 2>&1 &
echo $! > /tmp/ams-logs/mcp-server.pid
# Give MCP server a moment to bind
sleep 3
echo -e "${GREEN}MCP Server started!${NC}"

# Start Workbench UI if it exists
if [ -d "/workspace/workbench" ] && [ -f "/workspace/workbench/package.json" ]; then
    echo -e "${YELLOW}Starting Workbench UI...${NC}"
    cd /workspace/workbench

    # Install dependencies if node_modules doesn't exist
    if [ ! -d "node_modules" ]; then
        echo -e "${YELLOW}Installing Workbench dependencies...${NC}"
        npm install
    fi

    # Run vite directly (not through npm) for better process management
    # Use setsid to fully detach the process
    setsid /usr/bin/node node_modules/vite/bin/vite.js --host 0.0.0.0 > /tmp/ams-logs/workbench-ui.log 2>&1 &
    echo $! > /tmp/ams-logs/workbench-ui.pid

    # Wait for Workbench UI to be ready
    echo -e "${YELLOW}Waiting for Workbench UI to be ready...${NC}"
    max_attempts=30
    attempt=0
    until curl -s http://localhost:5173 > /dev/null 2>&1; do
        attempt=$((attempt + 1))
        if [ $attempt -ge $max_attempts ]; then
            echo -e "${RED}Workbench UI failed to start. Check logs at /tmp/ams-logs/workbench-ui.log${NC}"
            echo -e "${YELLOW}You can start it manually: cd /workspace/workbench && npm run dev${NC}"
            break
        fi
        sleep 1
    done

    if [ $attempt -lt $max_attempts ]; then
        echo -e "${GREEN}Workbench UI is ready!${NC}"
    fi
else
    echo -e "${YELLOW}Workbench UI not found. Skipping...${NC}"
    echo -e "${YELLOW}To set up the workbench, run: cd /workspace/workbench && npm install${NC}"
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  All services started successfully!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "  ${BLUE}Workbench UI:${NC}    http://localhost:5173  (or check VS Code Ports panel)"
echo -e "  ${BLUE}Memory Server:${NC}   http://localhost:8000  (REST API)"
echo -e "  ${BLUE}MCP Server:${NC}      http://localhost:9000  (SSE transport)"
echo -e "  ${BLUE}API Docs:${NC}        http://localhost:8000/docs"
echo -e "  ${BLUE}Redis Insight:${NC}   http://localhost:28001"
echo -e "  ${BLUE}Redis:${NC}           localhost:26379"
echo ""
echo -e "${YELLOW}Note: VS Code forwards container ports. Check the 'Ports' panel for exact URLs.${NC}"
echo ""
echo -e "${YELLOW}To use Claude with memory tools:${NC}"
echo -e "  claude --mcp-config /workspace/.devcontainer/mcp-config.json"
echo ""
echo -e "${YELLOW}To view logs:${NC}"
echo -e "  tail -f /tmp/ams-logs/memory-server.log"
echo -e "  tail -f /tmp/ams-logs/mcp-server.log"
echo -e "  tail -f /tmp/ams-logs/workbench-ui.log"
echo ""

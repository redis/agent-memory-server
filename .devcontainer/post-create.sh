#!/bin/bash
set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Post-Create Setup${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Install Python dependencies for the memory server
echo -e "${YELLOW}Installing Memory Server dependencies...${NC}"
cd /workspace
uv sync --all-extras

# Install Workbench UI dependencies if the project exists
if [ -d "/workspace/workbench" ] && [ -f "/workspace/workbench/package.json" ]; then
    echo -e "${YELLOW}Installing Workbench UI dependencies...${NC}"
    cd /workspace/workbench
    npm install || {
        echo -e "${YELLOW}Warning: npm install had issues. You may need to run 'npm install' manually.${NC}"
    }
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Setup Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo ""
echo -e "  1. The services will start automatically via postStartCommand"
echo ""
echo -e "  2. Access the applications:"
echo -e "     - ${BLUE}Workbench UI:${NC}  http://localhost:25173"
echo -e "     - ${BLUE}Memory API:${NC}    http://localhost:28000/docs"
echo -e "     - ${BLUE}Redis Insight:${NC} http://localhost:28001"
echo -e "     - ${BLUE}Redis:${NC}         localhost:26379"
echo ""
echo -e "  3. To use Claude with memory tools:"
echo -e "     ${BLUE}claude --mcp-config /workspace/.devcontainer/mcp-config.json${NC}"
echo ""
echo -e "  4. Run examples:"
echo -e "     ${BLUE}cd /workspace && uv run python examples/travel_agent.py${NC}"
echo ""
echo -e "  5. Run tests:"
echo -e "     ${BLUE}cd /workspace && uv run pytest${NC}"
echo ""

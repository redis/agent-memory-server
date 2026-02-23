#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Stopping Agent Memory Workbench services...${NC}"

# Stop Memory Server
if [ -f /tmp/ams-logs/memory-server.pid ]; then
    PID=$(cat /tmp/ams-logs/memory-server.pid)
    if kill -0 $PID 2>/dev/null; then
        kill $PID
        echo -e "${GREEN}Stopped Memory Server (PID: $PID)${NC}"
    fi
    rm -f /tmp/ams-logs/memory-server.pid
fi

# Stop Workbench UI
if [ -f /tmp/ams-logs/workbench-ui.pid ]; then
    PID=$(cat /tmp/ams-logs/workbench-ui.pid)
    if kill -0 $PID 2>/dev/null; then
        kill $PID
        echo -e "${GREEN}Stopped Workbench UI (PID: $PID)${NC}"
    fi
    rm -f /tmp/ams-logs/workbench-ui.pid
fi

# Kill any remaining node processes for the workbench
pkill -f "vite.*workbench" 2>/dev/null && echo -e "${GREEN}Killed remaining Vite processes${NC}" || true

echo -e "${GREEN}All services stopped.${NC}"

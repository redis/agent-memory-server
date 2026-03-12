#!/bin/bash
# Run system scale tests with configurable parameters
#
# Usage:
#   ./run_scale_tests.sh [quick|standard|production]
#
# Profiles:
#   quick      - Fast smoke test (default)
#   standard   - Standard scale test
#   production - Production-like scale test

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Default profile
PROFILE="${1:-quick}"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Agent Memory Server - System Scale Tests${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Set scale parameters based on profile
case "$PROFILE" in
  quick)
    echo -e "${YELLOW}Profile: Quick Smoke Test${NC}"
    export SCALE_SHORT_MESSAGES=5
    export SCALE_MEDIUM_MESSAGES=20
    export SCALE_LONG_MESSAGES=50
    export SCALE_VERY_LARGE_MESSAGE_SIZE=2000
    export SCALE_PARALLEL_SESSIONS=3
    export SCALE_CONCURRENT_UPDATES=5
    ;;
  standard)
    echo -e "${YELLOW}Profile: Standard Scale Test${NC}"
    export SCALE_SHORT_MESSAGES=10
    export SCALE_MEDIUM_MESSAGES=50
    export SCALE_LONG_MESSAGES=200
    export SCALE_VERY_LARGE_MESSAGE_SIZE=5000
    export SCALE_PARALLEL_SESSIONS=5
    export SCALE_CONCURRENT_UPDATES=10
    ;;
  production)
    echo -e "${YELLOW}Profile: Production-Like Scale Test${NC}"
    export SCALE_SHORT_MESSAGES=20
    export SCALE_MEDIUM_MESSAGES=100
    export SCALE_LONG_MESSAGES=500
    export SCALE_VERY_LARGE_MESSAGE_SIZE=10000
    export SCALE_PARALLEL_SESSIONS=10
    export SCALE_CONCURRENT_UPDATES=20
    ;;
  *)
    echo -e "${RED}Unknown profile: $PROFILE${NC}"
    echo "Usage: $0 [quick|standard|production]"
    exit 1
    ;;
esac

echo ""
echo "Configuration:"
echo "  Short messages: $SCALE_SHORT_MESSAGES"
echo "  Medium messages: $SCALE_MEDIUM_MESSAGES"
echo "  Long messages: $SCALE_LONG_MESSAGES"
echo "  Large message size: $SCALE_VERY_LARGE_MESSAGE_SIZE chars"
echo "  Parallel sessions: $SCALE_PARALLEL_SESSIONS"
echo "  Concurrent updates: $SCALE_CONCURRENT_UPDATES"
echo ""

# Check prerequisites
echo -e "${YELLOW}Checking prerequisites...${NC}"

# Check if server is running
SERVER_URL="${MEMORY_SERVER_BASE_URL:-http://localhost:8001}"
if ! curl -s -f "$SERVER_URL/health" > /dev/null 2>&1; then
    echo -e "${RED}❌ Memory server not reachable at $SERVER_URL${NC}"
    echo "   Start the server with: uv run agent-memory api --port 8001"
    exit 1
fi
echo -e "${GREEN}✓ Memory server is running${NC}"

# Check for API keys
if [ -z "$OPENAI_API_KEY" ] && [ -z "$ANTHROPIC_API_KEY" ]; then
    echo -e "${RED}❌ No API keys found${NC}"
    echo "   Set OPENAI_API_KEY or ANTHROPIC_API_KEY"
    exit 1
fi
echo -e "${GREEN}✓ API keys configured${NC}"

echo ""
echo -e "${GREEN}Running tests...${NC}"
echo ""

# Run the tests
uv run pytest tests/system/test_long_conversation_scale.py \
    --run-api-tests \
    -v \
    -s \
    --tb=short

# Capture exit code
EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}✅ All system tests passed!${NC}"
    echo -e "${GREEN}========================================${NC}"
else
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}❌ Some tests failed${NC}"
    echo -e "${RED}========================================${NC}"
fi

exit $EXIT_CODE

